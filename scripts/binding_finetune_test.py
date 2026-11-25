"""Short fine-tune of a BubblSpace checkpoint on contrastive text documents (A10 test, not a training phase).

Used for the two-arm test in doc/binding-study-02-priors-and-word-order.md §6.1: the same checkpoint, steps and learning
rate, once on contrastive v3 (control) and once on v3 with order alternation (fix). Data: JSONL from
scripts/gen_contrastive_v4.py.

Each row of a batch is ONE document: <|endoftext|> doc <|endoftext|>, right-padded. With causal attention no document sees
another one or any padding, which matches the doc-masked training setup without needing a mask. Pad targets are ignored.
Optimizers are the project's Muon + AdamW (fresh state), with a short warmup and a linear decay.

    python scripts/binding_finetune_test.py --checkpoint step_22888.pt --data fix.jsonl --out fix.pt --steps 400
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from scripts.run_benchmarks import load_weights  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model.moe import probe_grouped_mm  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402
from src.optim.param_groups import build_optimizers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--micro_docs", type=int, default=32, help="documents per forward/backward pass")
    ap.add_argument("--accum", type=int, default=6, help="passes per optimizer step")
    ap.add_argument("--max_len", type=int, default=112, help="documents longer than this (in tokens) are skipped")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--end_lr", type=float, default=6e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="merge configs/debug.yaml (tiny model) for a smoke test")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    cfg = load_config(a.config, debug=a.debug)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda" if torch.cuda.is_available() else "cpu"  # cpu only for a tiny-model smoke test
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    eot, pad, vocab = tok.eos_token_id, tok.convert_tokens_to_ids("<|fim_pad|>"), len(tok)

    texts = [json.loads(line)["text"] for line in open(a.data)]
    docs = [[eot] + ids + [eot] for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    docs = [d for d in docs if len(d) <= a.max_len]
    need = a.steps * a.accum * a.micro_docs
    print(f"[data] {len(docs)} docs <= {a.max_len} tokens (of {len(texts)}), mean {sum(map(len, docs)) / len(docs):.1f} tokens; need {need}", flush=True)
    assert len(docs) >= need, "not enough documents for one pass; generate more or lower --steps"
    rng.shuffle(docs)

    model = Transformer(cfg.model).to(device)
    step0 = load_weights(model, a.checkpoint)
    model.train()
    muon_opt, adamw_opt = build_optimizers(model, cfg.optimizer.muon, cfg.optimizer.adamw)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        return a.lr + (a.end_lr - a.lr) * (s - a.warmup) / max(1, a.steps - a.warmup)

    def batch(i):
        chunk = docs[i * a.micro_docs:(i + 1) * a.micro_docs]
        L = max(map(len, chunk))
        x = torch.full((len(chunk), L - 1), pad, dtype=torch.long)
        y = torch.full((len(chunk), L - 1), -100, dtype=torch.long)
        for r, d in enumerate(chunk):
            x[r, :len(d) - 1] = torch.tensor(d[:-1])
            y[r, :len(d) - 1] = torch.tensor(d[1:])
        return x.to(device), y.to(device)

    t0, i, log = time.time(), 0, []
    for step in range(a.steps):
        for opt in (muon_opt, adamw_opt):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
        tot = 0.0
        for _ in range(a.accum):
            x, y = batch(i)
            i += 1
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits, lb_loss, z_loss, _ = model(x)
                ce = F.cross_entropy(logits[..., :vocab].reshape(-1, vocab).float(), y.view(-1), ignore_index=-100)
                loss = ce + cfg.model.moe.load_balance_coef * lb_loss + cfg.model.moe.router_z_loss_coef * z_loss
            (loss / a.accum).backward()
            tot += ce.item() / a.accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        muon_opt.step()
        adamw_opt.step()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        log.append(round(tot, 4))
        if step % 20 == 0 or step == a.steps - 1:
            print(f"step {step}/{a.steps} loss {tot:.4f} grad {gn:.3f} lr {lr_at(step):.2e} "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else 0:.1f}G {time.time() - t0:.0f}s", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": step0, "finetune": vars(a), "loss_log": log}, a.out)
    print(f"wrote {a.out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
