"""WinoGrande-style entity-attribute binding probe via log-likelihood scoring
(not free-form generation) — reuses the exact BubblSpaceLM._score_continuation
method the real lm-eval-harness benchmarks use, so results are directly
comparable to the WinoGrande number in benchmark/README.md.

Each case is a minimal pair: two prefixes differing only in which entity is
named in the "is ___" clause, and two candidate completions. A model with
correct entity-attribute binding should prefer completion 1 under prefix A
and completion 2 under prefix B (the preference should *flip* when the named
entity flips) — exactly the WinoGrande mechanism, but hand-written fresh
cases rather than WinoGrande's own dataset.

Usage:
    python scripts/probe_entity_binding.py --checkpoint /path/to/step_19073.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[0]).rsplit("/scripts", 1)[0] or ".")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.eval.lm_eval_adapter import BubblSpaceLM
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

CASES = [
    {
        "name": "car / alley (original probe)",
        "ctxA": "The car does not fit into the alley because the alley is",
        "ctxB": "The car does not fit into the alley because the car is",
        "c1": " too narrow.",
        "c2": " too wide.",
    },
    {
        "name": "suitcase / car trunk",
        "ctxA": "The suitcase would not fit in the car trunk because the trunk was",
        "ctxB": "The suitcase would not fit in the car trunk because the suitcase was",
        "c1": " too small.",
        "c2": " too big.",
    },
    {
        "name": "truck / tunnel",
        "ctxA": "The truck could not pass through the tunnel because the tunnel was",
        "ctxB": "The truck could not pass through the tunnel because the truck was",
        "c1": " too low.",
        "c2": " too tall.",
    },
    {
        "name": "wardrobe / doorway",
        "ctxA": "The wardrobe would not go through the doorway because the doorway was",
        "ctxB": "The wardrobe would not go through the doorway because the wardrobe was",
        "c1": " too narrow.",
        "c2": " too wide.",
    },
    {
        "name": "sofa / staircase",
        "ctxA": "The sofa got stuck on the staircase because the staircase was",
        "ctxB": "The sofa got stuck on the staircase because the sofa was",
        "c1": " too narrow.",
        "c2": " too wide.",
    },
    {
        "name": "boat / bridge",
        "ctxA": "The boat could not pass under the bridge because the bridge was",
        "ctxB": "The boat could not pass under the bridge because the boat was",
        "c1": " too low.",
        "c2": " too tall.",
    },
    {
        "name": "piano / elevator",
        "ctxA": "The piano would not fit in the elevator because the elevator was",
        "ctxB": "The piano would not fit in the elevator because the piano was",
        "c1": " too small.",
        "c2": " too big.",
    },
    {
        "name": "ladder / garage",
        "ctxA": "The ladder would not stand up in the garage because the garage was",
        "ctxB": "The ladder would not stand up in the garage because the ladder was",
        "c1": " too low.",
        "c2": " too tall.",
    },
    {
        "name": "mattress / van",
        "ctxA": "The mattress would not fit in the van because the van was",
        "ctxB": "The mattress would not fit in the van because the mattress was",
        "c1": " too small.",
        "c2": " too big.",
    },
    {
        "name": "fridge / kitchen doorway",
        "ctxA": "The fridge would not fit through the kitchen doorway because the doorway was",
        "ctxB": "The fridge would not fit through the kitchen doorway because the fridge was",
        "c1": " too narrow.",
        "c2": " too wide.",
    },
    {
        "name": "bus / garage",
        "ctxA": "The bus could not fit in the garage because the garage was",
        "ctxB": "The bus could not fit in the garage because the bus was",
        "c1": " too low.",
        "c2": " too tall.",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--cases_file", default=None, help="JSON file with a top-level 'cases' list, overrides the built-in CASES")
    args = parser.parse_args()

    cases = CASES
    if args.cases_file:
        with open(args.cases_file) as f:
            data = json.load(f)
        cases = data["cases"]

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer or cfg.tokenizer.path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    model = Transformer(cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step")
    print(f"[ckpt] loaded {args.checkpoint} (step {step})")

    lm = BubblSpaceLM(model, tok, cfg, device)

    results = []
    n_correct_binding = 0
    category_totals: dict[str, list[int]] = {}
    for case in cases:
        ctxA, ctxB, c1, c2 = case["ctxA"], case["ctxB"], case["c1"], case["c2"]

        enc_ctxA = lm._encode(ctxA)
        enc_ctxB = lm._encode(ctxB)
        enc_c1 = lm._encode(c1)
        enc_c2 = lm._encode(c2)

        score_A_c1, _ = lm._score_continuation(enc_ctxA, enc_c1)
        score_A_c2, _ = lm._score_continuation(enc_ctxA, enc_c2)
        score_B_c1, _ = lm._score_continuation(enc_ctxB, enc_c1)
        score_B_c2, _ = lm._score_continuation(enc_ctxB, enc_c2)

        pref_A = c1 if score_A_c1 > score_A_c2 else c2
        pref_B = c1 if score_B_c1 > score_B_c2 else c2

        # correct binding: prefix A should prefer c1, prefix B should prefer c2
        correct = (pref_A.strip() == c1.strip()) and (pref_B.strip() == c2.strip())
        if correct:
            n_correct_binding += 1

        category = case.get("category", "uncategorized")
        category_totals.setdefault(category, [0, 0])
        category_totals[category][1] += 1
        if correct:
            category_totals[category][0] += 1

        row = {
            "name": case["name"],
            "category": category,
            "ctxA": ctxA,
            "ctxB": ctxB,
            "c1": c1,
            "c2": c2,
            "score_A_c1": score_A_c1,
            "score_A_c2": score_A_c2,
            "score_B_c1": score_B_c1,
            "score_B_c2": score_B_c2,
            "pref_A": pref_A,
            "pref_B": pref_B,
            "correct_binding": correct,
        }
        results.append(row)

        print(f"\n=== {case['name']} ===")
        print(f"  ctxA: ...{ctxA[-50:]}")
        print(f"    {c1.strip():<14} logprob={score_A_c1:8.3f}")
        print(f"    {c2.strip():<14} logprob={score_A_c2:8.3f}   -> prefers: {pref_A.strip()}")
        print(f"  ctxB: ...{ctxB[-50:]}")
        print(f"    {c1.strip():<14} logprob={score_B_c1:8.3f}")
        print(f"    {c2.strip():<14} logprob={score_B_c2:8.3f}   -> prefers: {pref_B.strip()}")
        print(f"  correct binding (flips A->c1, B->c2): {correct}")

    print(f"\n{'=' * 60}")
    print(f"Correct entity-attribute binding: {n_correct_binding}/{len(cases)}")
    print("\nBy category:")
    category_summary = {}
    for cat, (correct, total) in sorted(category_totals.items()):
        pct = 100 * correct / total
        category_summary[cat] = {"correct": correct, "total": total, "pct": pct}
        print(f"  {cat:<24} {correct}/{total}  ({pct:.0f}%)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "step": step,
                    "n_correct": n_correct_binding,
                    "n_total": len(cases),
                    "by_category": category_summary,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"[probe] saved results to {out_path}")


if __name__ == "__main__":
    main()
