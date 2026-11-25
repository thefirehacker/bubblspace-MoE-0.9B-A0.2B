"""Free-generation check for the binding probes: does what the model WRITES agree with the probe?

The WSP lesson (doc/wsp-experiments.md §4): a two-way log-prob probe can say "fixed" while greedy generation
is still wrong, because the model's real top choice is a third continuation. This script measures that gap
automatically on a probe file (v4 / v5 are the held-out ones), for both contexts of every case:

  hit               greedy decoding (argmax over the full vocabulary) reproduces the correct completion up to and
                    including its first token that differs from the wrong completion ("too big" vs "too small":
                    must write "too big"; a shared first token alone doesn't count)
  flip_hit          hit holds for BOTH contexts (ctxA -> c1 and ctxB -> c2), the generation analogue
                    of the probe's flip criterion
  gap               logp(correct completion) - logp(greedy continuation of the same length), in nats; 0 means
                    the model would write the correct completion itself, very negative means the probe's
                    two-way choice says little about generation
It also saves the 12-token greedy continuation of every context for manual reading.

    python scripts/eval_binding_generation.py --checkpoint step_N.pt \
        --cases_file benchmark/data/probefinding-winograndestyle/probefinding-winogrande-v5-heldout-structures-20261123.json \
        --output binding_gen_v5_stepN.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoTokenizer  # noqa: E402

from scripts.run_benchmarks import load_weights  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval.determinism import make_deterministic  # noqa: E402
from src.eval.lm_eval_adapter import BubblSpaceLM  # noqa: E402
from src.model.moe import probe_grouped_mm  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402


@torch.no_grad()
def greedy(model, ids, n, device, real_vocab, eot):
    cache = model.new_kv_cache()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits, _, _, _ = model(x, kv_cache=cache)
        nid = int(logits[0, -1, :real_vocab].argmax())
        if nid == eot:
            break
        out.append(nid)
        x = torch.tensor([[nid]], dtype=torch.long, device=device)
    return out


def main():
    make_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cases_file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--show_tokens", type=int, default=12)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    model = Transformer(cfg.model).to(device)
    step = load_weights(model, a.checkpoint)
    model.eval()
    lm = BubblSpaceLM(model, tok, cfg, device)
    real_vocab = len(tok)
    eot = tok.convert_tokens_to_ids(cfg.tokenizer.doc_separator)

    cases = json.loads(Path(a.cases_file).read_text())["cases"]
    rows, by_cat = [], defaultdict(lambda: [0, 0, 0.0])
    for c in cases:
        per_ctx = {}
        e1, e2 = lm._encode(c["c1"]), lm._encode(c["c2"])
        div = next((i for i, (x, y) in enumerate(zip(e1, e2)) if x != y), min(len(e1), len(e2)) - 1)
        for side, ctx, gold in (("A", c["ctxA"], c["c1"]), ("B", c["ctxB"], c["c2"])):
            ctx_ids, gold_ids = lm._encode(ctx), lm._encode(gold)
            gen_ids = greedy(model, ctx_ids, max(len(gold_ids), a.show_tokens), device, real_vocab, eot)
            g = gen_ids[: len(gold_ids)]
            lp_gold, _ = lm._score_continuation(ctx_ids, gold_ids)
            lp_gen, _ = lm._score_continuation(ctx_ids, g) if g else (lp_gold, None)
            k = min(div, len(gold_ids) - 1) + 1  # tokens that must match: shared prefix + first distinguishing token
            per_ctx[side] = {"hit": gen_ids[:k] == gold_ids[:k],
                             "gap": round(float(lp_gold - lp_gen), 3),
                             "gold": gold, "generation": tok.decode(gen_ids[: a.show_tokens])}
        flip = per_ctx["A"]["hit"] and per_ctx["B"]["hit"]
        cat = c.get("category", "uncategorized")
        by_cat[cat][0] += flip
        by_cat[cat][1] += 1
        by_cat[cat][2] += (per_ctx["A"]["gap"] + per_ctx["B"]["gap"]) / 2
        rows.append({"name": c["name"], "category": cat, "flip_hit": flip, "A": per_ctx["A"], "B": per_ctx["B"],
                     "ctxA": c["ctxA"], "ctxB": c["ctxB"]})

    n = len(rows)
    summary = {
        "n_cases": n,
        "flip_hit": sum(r["flip_hit"] for r in rows),
        "hit_rate": round(sum(r[s]["hit"] for r in rows for s in "AB") / (2 * n), 3),
        "mean_gap_nats": round(sum(r[s]["gap"] for r in rows for s in "AB") / (2 * n), 3),
        "by_category": {k: {"flip_hit": f"{v[0]}/{v[1]}", "mean_gap": round(v[2] / v[1], 3)} for k, v in by_cat.items()},
    }
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps({"checkpoint": a.checkpoint, "step": step, "cases_file": a.cases_file,
                                          "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
