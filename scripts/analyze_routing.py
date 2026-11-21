"""Per-layer MoE routing analysis for a checkpoint, split by data domain.

W&B's `tokens_per_expert` histogram sums counts for the same expert INDEX
across all 16 layers -- but expert 37 in layer 0 and expert 37 in layer 9 are
unrelated, so that view hides per-layer problems. This script measures
routing per (layer, expert), separately for each data group (english, code,
agentic, contrastive), on held-out validation windows.

Reports, per layer:
  - imbalance: max / min / coefficient of variation of expert load vs uniform
  - dead / starved experts (load < 10% of uniform) and overloaded (> 3x)
  - normalized routing entropy (1.0 = perfectly uniform)
  - domain specialization: experts whose share of a domain's tokens is far
    above that domain's average (e.g. "layer 7 expert 12 is a code expert")

Usage:
    python scripts/analyze_routing.py --checkpoint ckpt.pt --pools_yaml pools.yaml \
        --windows_per_group 48 --out results/routing_step1000.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.eval.determinism import make_deterministic
from src.data.dataset import _Pool
from src.model.attention import build_doc_block_mask
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer


def main():
    make_deterministic()  # bit-reproducible scores (see src/eval/determinism.py)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pools_yaml", required=True, help="yaml with a `pools:` mapping like phase2.pools (paths for THIS machine)")
    ap.add_argument("--windows_per_group", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda"
    L_seq = cfg.model.max_seq_len
    n_layers, n_exp, top_k = cfg.model.n_layers, cfg.model.moe.n_experts, cfg.model.moe.top_k

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    eot, pad = tok.eos_token_id, tok.convert_tokens_to_ids("<|fim_pad|>")

    model = Transformer(cfg.model).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step")

    captured = {}

    def make_hook(i):
        def hook(module, inp, out):
            captured[i] = out.detach()
        return hook

    for i, b in enumerate(model.blocks):
        b.moe.router.register_forward_hook(make_hook(i))

    pools_cfg = yaml.safe_load(open(args.pools_yaml))["pools"]
    pools = [_Pool(n, s, L_seq, val_tokens=8 * 1024 * 1024) for n, s in pools_cfg.items()]
    groups = sorted({p.group for p in pools})

    # counts[group] -> (n_layers, n_exp) number of top-k selections
    counts = {g: np.zeros((n_layers, n_exp), dtype=np.int64) for g in groups}
    tokens = {g: 0 for g in groups}

    for g in groups:
        members = [p for p in pools if p.group == g]
        per_pool = max(1, args.windows_per_group // len(members))
        xs = []
        for p in members:
            xv, _ = p.val_batch(per_pool, pad)
            xs.append(xv)
        X = torch.from_numpy(np.concatenate(xs)).to(device)
        for s in range(0, X.shape[0], args.batch):
            xb = X[s : s + args.batch]
            keep = (xb != pad).reshape(-1)  # don't count routing of agentic pad tails
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(xb, None, build_doc_block_mask(xb, eot))
            for i in range(n_layers):
                logits = captured[i].float()[keep]
                idx = F.softmax(logits, dim=-1).topk(top_k, dim=-1).indices.reshape(-1)
                counts[g][i] += torch.bincount(idx, minlength=n_exp).cpu().numpy()
            tokens[g] += int(keep.sum())
        print(f"[routing] {g}: {tokens[g]:,} tokens from {len(members)} pools")

    total = sum(counts.values())
    uniform = total.sum(axis=1, keepdims=True) / n_exp  # per-layer uniform load

    layers = []
    for i in range(n_layers):
        load = total[i] / uniform[i, 0]  # 1.0 = exactly fair share
        p = total[i] / total[i].sum()
        entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(n_exp))
        spec = []
        for g in groups:
            share = counts[g][i] / max(1, counts[g][i].sum())  # this domain's distribution over experts
            other = (total[i] - counts[g][i]) / max(1, (total[i] - counts[g][i]).sum())
            lift = share / np.maximum(other, 1e-9)  # >1: expert preferred by this domain vs the rest
            for e in np.argsort(-lift)[:3]:
                if lift[e] >= 2.0 and share[e] >= 1.0 / n_exp:
                    spec.append({"group": g, "expert": int(e), "lift": round(float(lift[e]), 2), "share_of_group": round(float(share[e]), 4)})
        layers.append({
            "layer": i,
            "load_max": round(float(load.max()), 3),
            "load_min": round(float(load.min()), 3),
            "load_cv": round(float(load.std()), 3),
            "entropy": round(entropy, 4),
            "starved_lt_0.1x": [int(e) for e in np.flatnonzero(load < 0.1)],
            "overloaded_gt_3x": [int(e) for e in np.flatnonzero(load > 3.0)],
            "specialized": spec,
        })

    out = {
        "checkpoint": args.checkpoint,
        "step": step,
        "tokens_per_group": tokens,
        "layers": layers,
        "counts": {g: counts[g].tolist() for g in groups},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\n=== routing @ step {step} ===")
    print(f"{'layer':>5} {'max':>6} {'min':>6} {'cv':>6} {'entropy':>8}  starved  overloaded  specialized (lift>=2)")
    for r in layers:
        sp = ", ".join(f"{s['group'][:4]}:e{s['expert']}x{s['lift']}" for s in r["specialized"]) or "-"
        print(f"{r['layer']:>5} {r['load_max']:>6} {r['load_min']:>6} {r['load_cv']:>6} {r['entropy']:>8}  "
              f"{len(r['starved_lt_0.1x']):>7}  {len(r['overloaded_gt_3x']):>10}  {sp}")
    print(f"[routing] saved {args.out}")


if __name__ == "__main__":
    main()
