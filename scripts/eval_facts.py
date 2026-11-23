"""Closed-book simple-facts probe: greedy few-shot QA, alias match.

Prompt: a fixed 3-shot "Q: ... / A: ..." block, then the question; generation stops at a newline.
An answer is correct if any accepted alias appears in it as a whole word (case-insensitive,
whitespace-normalized).
Deterministic (greedy, src/eval/determinism.py).

    python scripts/eval_facts.py --checkpoint step_22888.pt \
        --facts benchmark/data/facts-closedbook/facts-simple-v1-20261123.json --output facts_step22888.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_benchmarks import load_weights  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval.determinism import make_deterministic  # noqa: E402
from src.eval.lm_eval_adapter import BubblSpaceLM  # noqa: E402
from src.model.moe import probe_grouped_mm  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402


class _Req:
    def __init__(self, context, gen_kwargs):
        self.args = (context, gen_kwargs)


def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()


def main():
    make_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.facts).read_text())
    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    model = Transformer(cfg.model).to(device)
    step = load_weights(model, args.checkpoint)
    model.eval()
    lm = BubblSpaceLM(model, tok, cfg, device)

    shots = "".join(f"Q: {x['q']}\nA: {x['a']}\n\n" for x in data["fewshot"])
    reqs = [_Req(f"{shots}Q: {it['q']}\nA:", {"until": ["\n"], "max_gen_toks": 24}) for it in data["items"]]
    outs = lm.generate_until(reqs)

    rows, by_cat = [], defaultdict(lambda: [0, 0])
    for it, out in zip(data["items"], outs):
        ans = out.strip()
        # whole-word match, so short aliases ("O", "Fe", "6") don't match inside other words
        ok = any(re.search(rf"(?<!\w){re.escape(norm(a))}(?!\w)", norm(ans)) for a in it["a"])
        rows.append({"cat": it["cat"], "q": it["q"], "gold": it["a"], "answer": ans, "correct": ok})
        by_cat[it["cat"]][0] += ok
        by_cat[it["cat"]][1] += 1
    total = sum(r["correct"] for r in rows)
    summary = {"correct": total, "n": len(rows), "acc": round(total / len(rows), 4),
               "by_category": {c: f"{a}/{n}" for c, (a, n) in by_cat.items()}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"checkpoint": args.checkpoint, "step": step, "facts": args.facts,
                                             "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        if not r["correct"]:
            print(f"  [x] {r['q']}  ->  {r['answer']!r}  (gold {r['gold'][0]})")


if __name__ == "__main__":
    main()
