"""Controls for the WinoGrande generation level (L3) of scripts/binding_levels_hf.py.

The main study found that models write the SAME name for both WinoGrande twins in 85-98% of pairs, mostly the
first-mentioned name early in training and the second-mentioned one later. Two explanations have to be ruled out:

  prompt shuffle  The 5-shot prompt itself could cause it (Zhao et al. 2021, "Calibrate Before Use": majority-label,
                  recency and common-token biases). Rerun generation with 3 different example sets/orders (seeds
                  0, 1, 2). If the habit is the model's, it survives every prompt.
  order swap      Swap the two names inside each sentence ("Sarah ... than Maria" -> "Maria ... than Sarah"); the
                  correct answer swaps with them. Then compare what the model writes:
                    same NAME as before        -> a preference for that name (frequency phase)
                    same POSITION as before    -> a preference for first / second mention (position phase)
                  Only rows where each name appears exactly once (and neither contains the other) are swapped.

Only greedy generation is rerun; L1/L2 (log-prob ranking) and the probes don't use a prompt.

    python scripts/binding_controls_hf.py --model EleutherAI/pythia-1b --revision step20000 --out controls/pythia-1b_step20000.json
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import binding_levels_hf as bl  # noqa: E402


def opt(r, k):
    return r["option1"] if k == "1" else r["option2"]


def question(r):
    return f"Sentence: {r['sentence']}\nOptions: {r['option1']} or {r['option2']}\nThe blank _ refers to:"


def make_prompt(train, seed, n_shot=5):
    rng = random.Random(seed)
    shots = [train[j] for j in rng.sample(range(len(train)), n_shot)]
    return "".join(question(s) + f" {opt(s, s['answer'])}\n\n" for s in shots)


def mention_order(r):
    p1, p2 = r["sentence"].find(r["option1"]), r["sentence"].find(r["option2"])
    return (r["option1"], r["option2"]) if (p1 != -1 and (p2 == -1 or p1 <= p2)) else (r["option2"], r["option1"])


def swap_row(r):
    """Swap the two names in the sentence; the answer (a role) now points to the other name."""
    a, b, s = r["option1"], r["option2"], r["sentence"]
    if a in b or b in a or s.count(a) != 1 or s.count(b) != 1:
        return None
    s2 = s.replace(a, "\x00").replace(b, a).replace("\x00", b)
    return {"sentence": s2, "option1": a, "option2": b, "answer": "2" if r["answer"] == "1" else "1"}


def generate_all(lm, rows, prompt):
    out = []
    for r in rows:
        gen = lm.tok.decode(lm.greedy(lm.ctx_ids(prompt + question(r)), 8)).strip().split("\n")[0]
        g = gen.lower()
        first, second = mention_order(r)
        wrote = first if g.startswith(first.lower()) else (second if g.startswith(second.lower()) else None)
        out.append({"wrote": wrote, "correct": wrote == opt(r, r["answer"]),
                    "pos": None if wrote is None else ("first" if wrote == first else "second"), "gen": gen[:40]})
    return out


def stats(res, pairs):
    n = len(res)
    ans = [x for x in res if x["wrote"]]
    return {"answered": round(len(ans) / n, 4),
            "row_acc": round(sum(x["correct"] for x in res) / n, 4),
            "L3_pairs": round(sum(res[a]["correct"] and res[b]["correct"] for a, b in pairs) / len(pairs), 4),
            "same_name_both_twins": round(sum(1 for a, b in pairs if res[a]["wrote"] and res[a]["wrote"] == res[b]["wrote"]) / len(pairs), 4),
            "first_mentioned_share": round(sum(x["pos"] == "first" for x in ans) / max(1, len(ans)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/hf_controls_cache"))
    a = ap.parse_args()
    from datasets import load_dataset

    t0 = time.time()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    lm = bl.LM(a.model, a.revision, a.cache_dir, dtype)
    val = [dict(r) for r in load_dataset("allenai/winogrande", "winogrande_xl", split="validation")]
    train = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    pairs = bl.winogrande_pairs(val)
    res = {"model": a.model, "revision": a.revision, "weights_format": lm.weights_format,
           "weight_sha256": bl.WEIGHT_SHA, "n_rows": len(val), "n_pairs": len(pairs), "prompt_shuffle": {}}

    # 1. prompt shuffle
    gens = {}
    for sd in [int(x) for x in a.seeds.split(",")]:
        gens[sd] = generate_all(lm, val, make_prompt(train, sd))
        res["prompt_shuffle"][f"seed{sd}"] = stats(gens[sd], pairs)
        print(f"seed {sd}: {res['prompt_shuffle'][f'seed{sd}']}", flush=True)
    seeds = sorted(gens)
    agree = [sum(gens[s][i]["wrote"] == gens[seeds[0]][i]["wrote"] for i in range(len(val))) / len(val) for s in seeds[1:]]
    res["prompt_shuffle"]["same_written_name_as_seed0"] = [round(x, 4) for x in agree]

    # 2. order swap (seed-0 prompt), on rows where the swap is clean; twin pairs kept only if both rows swap
    idx = [i for i, r in enumerate(val) if swap_row(r)]
    keep = set(idx)
    sw_rows = [swap_row(val[i]) for i in idx]
    pos_of = {i: k for k, i in enumerate(idx)}
    sw_pairs = [(pos_of[x], pos_of[y]) for x, y in pairs if x in keep and y in keep]
    sw = generate_all(lm, sw_rows, make_prompt(train, 0))
    orig = [gens[0][i] for i in idx]
    both = [(o, s) for o, s in zip(orig, sw) if o["wrote"] and s["wrote"]]
    res["order_swap"] = {
        "n_rows_swapped": len(idx), "n_pairs_swapped": len(sw_pairs),
        "original": stats(orig, [(pos_of[x], pos_of[y]) for x, y in pairs if x in keep and y in keep]),
        "swapped": stats(sw, sw_pairs),
        "n_rows_answered_both": len(both),
        "kept_same_name": round(sum(o["wrote"] == s["wrote"] for o, s in both) / max(1, len(both)), 4),
        "kept_same_position": round(sum(o["pos"] == s["pos"] for o, s in both) / max(1, len(both)), 4),
    }
    print(f"order swap: {res['order_swap']}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    del lm
    torch.cuda.empty_cache()
    shutil.rmtree(a.cache_dir, ignore_errors=True)
    print(f"wrote {a.out} ({res['elapsed_s']} s)")


if __name__ == "__main__":
    main()
