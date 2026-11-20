"""lm-evaluation-harness adapter for the custom BubblSpace Transformer.

This model is not a `transformers`-native architecture (no config.json,
no AutoModel registration), so lm-eval-harness's default HFLM wrapper can't
load it. This adapter implements EleutherAI's `LM` base class directly,
wrapping our own model.forward() for loglikelihood scoring.

Only loglikelihood() is exercised by the benchmark set we actually run
(HellaSwag, PIQA, ARC-Easy/Challenge, WinoGrande, LAMBADA are all
multiple-choice / loglikelihood-scored tasks — appropriate for a
base-pretrained, non-instruction-tuned model). loglikelihood_rolling()
and generate_until() are implemented for completeness/robustness in case
a task requests them, but are not on the critical path.

Everything runs with batch_size=1 (one request per forward pass). The
model's Attention module does unconditional causal SDPA with no padding
mask support (fine for training's fixed-length windows), so batching
variable-length eval requests together would need padding-mask support
that doesn't exist in the model code today. Correctness over speed here —
eval request counts (low thousands) are trivial next to a 19k-step
training run, so unbatched scoring is still fast in absolute terms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from src.config import RunConfig
from src.model.transformer import Transformer


class BubblSpaceLM(LM):
    def __init__(self, model: Transformer, tokenizer, cfg: RunConfig, device: str):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.real_vocab_size = len(tokenizer)
        self.max_seq_len = cfg.model.max_seq_len
        self.eot_token_id = tokenizer.eos_token_id
        self._autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def _score_continuation(self, context_enc: list[int], continuation_enc: list[int]) -> tuple[float, bool]:
        if len(context_enc) == 0:
            context_enc = [self.eot_token_id]

        total_len = len(context_enc) + len(continuation_enc)
        if total_len > self.max_seq_len:
            overflow = total_len - self.max_seq_len
            context_enc = context_enc[overflow:]
            # if the continuation alone still overflows, truncate it from
            # the right as a last resort (shouldn't happen for our task set,
            # continuations here are short phrases/words).
            if len(context_enc) + len(continuation_enc) > self.max_seq_len:
                continuation_enc = continuation_enc[: self.max_seq_len - len(context_enc)]

        full_enc = context_enc + continuation_enc
        inp = torch.tensor([full_enc[:-1]], dtype=torch.long, device=self.device)
        targets = torch.tensor(full_enc[1:], dtype=torch.long, device=self.device)

        with torch.autocast(device_type=self.device, dtype=self._autocast_dtype):
            logits, _, _, _ = self.model(inp)
        logits = logits[0, :, : self.real_vocab_size].float()

        cont_len = len(continuation_enc)
        cont_logits = logits[-cont_len:]
        cont_targets = targets[-cont_len:]

        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_logprobs = log_probs.gather(1, cont_targets.unsqueeze(-1)).squeeze(-1)
        greedy_tokens = cont_logits.argmax(dim=-1)
        is_greedy = bool((greedy_tokens == cont_targets).all().item())

        return token_logprobs.sum().item(), is_greedy

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        for req in requests:
            context, continuation = req.args
            context_enc = self._encode(context)
            continuation_enc = self._encode(continuation)
            results.append(self._score_continuation(context_enc, continuation_enc))
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        results = []
        for req in requests:
            (text,) = req.args
            enc = self._encode(text)
            if len(enc) < 2:
                results.append(0.0)
                continue

            total_logprob = 0.0
            # non-overlapping windows: condition each chunk only on itself
            # (the standard lm-eval-harness "rolling" simplification — a true
            # fully-overlapping sliding window is far more expensive and not
            # needed for the sanity-check perplexity use case here).
            stride = self.max_seq_len
            for start in range(0, len(enc), stride):
                chunk = enc[start : start + stride]
                if len(chunk) < 2:
                    continue
                inp = torch.tensor([chunk[:-1]], dtype=torch.long, device=self.device)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=self.device)
                with torch.autocast(device_type=self.device, dtype=self._autocast_dtype):
                    logits, _, _, _ = self.model(inp)
                logits = logits[0, :, : self.real_vocab_size].float()
                log_probs = F.log_softmax(logits, dim=-1)
                token_logprobs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
                total_logprob += token_logprobs.sum().item()

            results.append(total_logprob)
        return results

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            max_new_tokens = gen_kwargs.get("max_gen_toks", 128) if isinstance(gen_kwargs, dict) else 128
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []

            ids = self._encode(context)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            generated_text = ""

            for _ in range(max_new_tokens):
                seq_len = input_ids.shape[1]
                if seq_len > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len :]
                with torch.autocast(device_type=self.device, dtype=self._autocast_dtype):
                    logits, _, _, _ = self.model(input_ids)
                next_logits = logits[0, -1, : self.real_vocab_size]
                next_id = next_logits.argmax().view(1, 1)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                if next_id.item() == self.eot_token_id:
                    break
                generated_text = self.tokenizer.decode(input_ids[0, len(ids) :].tolist())
                if any(stop in generated_text for stop in until):
                    for stop in until:
                        idx = generated_text.find(stop)
                        if idx != -1:
                            generated_text = generated_text[:idx]
                    break

            results.append(generated_text)
        return results
