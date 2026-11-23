"""Contamination check: HumanEval / MBPP vs the Phase 2 code training pools (token n-gram overlap).

The training pools are stored as tokenized uint16 shards (no raw text), so overlap is measured on
token sequences with the same tokenizer: every N-token window (default 20, roughly a 13-word
overlap) of each benchmark problem's prompt, reference solution and tests is hashed, then every
N-token window of every training shard is hashed (vectorized rolling polynomial hash, uint64)
and matched.

Per problem it reports how many of its windows appear in training, split by field, and flags:
  * solution_hit: any window of the reference solution appears in training
  * prompt_frac: fraction of the prompt/description windows found
Combined with codeexec_*.json it compares contamination rates of passed vs failed problems.

Usage (on the training host):
    python scripts/contamination_check_code.py \
        --pools /data/phase2/code_python_edu /data/phase2/code_starcoder_python \
        --codeexec benchmark/phase2/codeexec_phase2_step19000.json --out contamination_code.json
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

BASE = np.uint64(1000003)


def window_hashes(tokens: np.ndarray, n: int) -> np.ndarray:
    """Polynomial hash of every length-n window (uint64 arithmetic wraps mod 2^64)."""
    t = tokens.astype(np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * BASE + t[j:j + m]
    return h


def benchmark_items(tok):
    items = []
    he = load_dataset("openai/openai_humaneval", split="test")
    for ex in he:
        items.append({"task_id": ex["task_id"], "fields": {
            "prompt": ex["prompt"], "solution": ex["canonical_solution"], "tests": ex["test"]}})
    mb = load_dataset("google-research-datasets/mbpp", "full", split="test")
    for ex in mb:
        items.append({"task_id": f"mbpp/{ex['task_id']}", "fields": {
            "prompt": ex["text"], "solution": ex["code"], "tests": "\n".join(ex["test_list"])}})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="bubblspace-tokenizer")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--codeexec", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    items = benchmark_items(tok)
    # hash -> list of (item index, field)
    index = {}
    for i, it in enumerate(items):
        it["windows"] = {}
        for f, text in it["fields"].items():
            ids = np.array(tok.encode(text, add_special_tokens=False), dtype=np.uint32)
            hs = np.unique(window_hashes(ids, args.n))
            it["windows"][f] = hs
            for h in hs.tolist():
                index.setdefault(h, []).append((i, f))
    bench = np.array(sorted(index), dtype=np.uint64)
    print(f"[bench] {len(items)} problems, {len(bench):,} distinct {args.n}-token windows")

    found = {}  # pool -> set of matched hashes
    for pool in args.pools:
        t0, total, hit = time.time(), 0, set()
        for shard in sorted(glob.glob(f"{pool}/shard_*.bin")):
            arr = np.fromfile(shard, dtype=np.uint16)
            step = 50_000_000
            for s in range(0, len(arr), step):
                chunk = arr[s:s + step + args.n - 1]
                hs = window_hashes(chunk, args.n)
                m = np.isin(hs, bench, assume_unique=False)
                if m.any():
                    hit.update(np.unique(hs[m]).tolist())
            total += len(arr)
        found[pool] = hit
        print(f"[scan] {pool}: {total:,} tokens, {len(hit):,} benchmark windows found, {time.time() - t0:.0f}s")

    passed = {}
    for f in args.codeexec:
        e = json.loads(Path(f).read_text())
        for task, res in e["results"].items():
            for x in res:
                passed.setdefault(x["task_id"], False)
                passed[x["task_id"]] |= bool(x["passed"])

    all_hit = set().union(*found.values())
    rows = []
    for it in items:
        r = {"task_id": it["task_id"]}
        for f, hs in it["windows"].items():
            n_tot = len(hs)
            n_hit = int(sum(1 for h in hs.tolist() if h in all_hit))
            r[f"{f}_windows"], r[f"{f}_hits"] = n_tot, n_hit
            r[f"{f}_by_pool"] = {Path(p).name: int(sum(1 for h in hs.tolist() if h in found[p])) for p in args.pools}
        r["solution_hit"] = r["solution_hits"] > 0
        r["prompt_frac"] = r["prompt_hits"] / max(1, r["prompt_windows"])
        r["passed"] = passed.get(it["task_id"])
        rows.append(r)

    def summarize(prefix):
        sub = [r for r in rows if r["task_id"].startswith(prefix)]
        s = {"n": len(sub),
             "solution_hit": sum(r["solution_hit"] for r in sub),
             "prompt_frac_ge_0.5": sum(r["prompt_frac"] >= 0.5 for r in sub),
             "any_hit": sum((r["solution_hits"] + r["prompt_hits"] + r["tests_hits"]) > 0 for r in sub)}
        if any(r["passed"] is not None for r in sub):
            for flag in ("solution_hit",):
                p = [r for r in sub if r["passed"]]
                f_ = [r for r in sub if r["passed"] is False]
                s[f"passed_with_{flag}"] = f"{sum(r[flag] for r in p)}/{len(p)}"
                s[f"failed_with_{flag}"] = f"{sum(r[flag] for r in f_)}/{len(f_)}"
        return s

    summary = {"humaneval": summarize("HumanEval/"), "mbpp": summarize("mbpp/"), "n": args.n,
               "pools": args.pools, "codeexec": args.codeexec}
    Path(args.out).write_text(json.dumps({"summary": summary, "problems": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    top = sorted(rows, key=lambda r: -(r["solution_hits"] / max(1, r["solution_windows"])))[:10]
    print("[top solution overlap]")
    for r in top:
        print(f"  {r['task_id']:<14} solution {r['solution_hits']}/{r['solution_windows']}  prompt {r['prompt_hits']}/{r['prompt_windows']}  "
              f"tests {r['tests_hits']}/{r['tests_windows']}  passed={r['passed']}  {r['solution_by_pool']}")


if __name__ == "__main__":
    main()
