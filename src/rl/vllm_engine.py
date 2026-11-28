"""vLLM rollout engine for Phase 4 RL: fast sampling plus in-place weight sync from the trainer's model.

Colocated on one GPU, in ONE process with the trainer (vLLM's engine core runs in-process,
VLLM_ENABLE_V1_MULTIPROCESSING=0), so a weight sync is a device-to-device copy of the policy's parameters: no
serialization, no second copy of the model in another process.

One RL step:
    eng.wake()                       # weights + KV cache back on the GPU
    eng.sync_weights(policy)         # trainer params -> vLLM (our training-code names; the plugin maps them)
    outs = eng.generate(prompts, n=8, temperature=1.0, max_tokens=1024)
    eng.sleep()                      # free vLLM's GPU memory for the training step
    ... compute rewards, GRPO loss, optimizer step on `policy` ...

sync_weights goes through vLLM's reload_weights(), which calls the model's load_weights (vllm_plugin/bubblspace_vllm)
and then redoes any post-load weight processing the kernels need, so the result matches loading the same weights from
disk. Needs the vLLM env with the plugin installed (pip install -e vllm_plugin).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # must be set before vllm is imported

import torch


@dataclass
class Rollout:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]  # sampling log-prob of each completion token (behaviour policy)
    text: str
    finished: bool                    # stopped on a stop token (not the length limit)


class VLLMEngine:
    def __init__(self, model_dir: str, gpu_memory_utilization: float = 0.3, max_model_len: int = 16384,
                 enable_sleep_mode: bool = True, seed: int = 0):
        from vllm import LLM
        self.llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len, enable_sleep_mode=enable_sleep_mode, seed=seed,
                       enable_prefix_caching=True)
        self.tok = self.llm.get_tokenizer()
        self.stop_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id]
        self.asleep = False
        self.last_sync_s = None

    # ------------------------------------------------------------------ weights
    @torch.no_grad()
    def sync_weights(self, policy: torch.nn.Module | dict) -> float:
        """Copy the trainer's weights (a model or a state dict with our names) into vLLM. Returns seconds taken."""
        t0 = time.time()
        sd = policy if isinstance(policy, dict) else policy.state_dict()

        def it():
            for k, v in sd.items():
                k = k.removeprefix("_orig_mod.")  # torch.compile wrapper
                if k.startswith(("rope_", "_")):   # non-weight buffers
                    continue
                yield k, v.detach()

        self.llm.collective_rpc("reload_weights", kwargs={"weights_iterator": it(), "is_checkpoint_format": True})
        self.llm.reset_prefix_cache()  # cached KV was computed with the old weights
        torch.cuda.synchronize()
        self.last_sync_s = time.time() - t0
        return self.last_sync_s

    # ------------------------------------------------------------------ memory
    def sleep(self):
        """Free vLLM's GPU memory for the training step: level 1 drops the KV cache and moves the weights to CPU.
        (Level 2 would also drop non-weight buffers such as the RoPE cos/sin table, which a weight sync doesn't
        rewrite: tested 27 Nov, outputs broke after level 2 + sync.)"""
        if not self.asleep:
            self.llm.sleep(level=1)
            self.asleep = True

    def wake(self):
        if self.asleep:
            self.llm.wake_up()
            self.asleep = False

    # ------------------------------------------------------------------ sampling
    def generate(self, prompts: list[str] | list[list[int]], n: int = 1, temperature: float = 1.0, top_p: float = 1.0,
                 max_tokens: int = 1024, seed: int | None = None) -> list[list[Rollout]]:
        """n samples per prompt, with the sampling log-prob of every generated token (for importance ratios)."""
        from vllm import SamplingParams, TokensPrompt
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
                            stop_token_ids=self.stop_ids, logprobs=0, skip_special_tokens=False)
        inp = [TokensPrompt(prompt_token_ids=p) if isinstance(p, list) else p for p in prompts]
        res = []
        for o in self.llm.generate(inp, sp, use_tqdm=False):
            group = []
            for c in o.outputs:
                lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)] if c.logprobs else []
                group.append(Rollout(list(o.prompt_token_ids), list(c.token_ids), lps, c.text,
                                     c.finish_reason == "stop"))
            res.append(group)
        return res
