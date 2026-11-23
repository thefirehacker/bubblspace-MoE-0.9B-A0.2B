"""Decontaminate tokenized training pools against evaluation sets (Phase 2.5a).

Every evaluation item is tokenized with the training tokenizer and hashed into n-token windows:
  n = 8  for short items: TriviaQA / NQ-open questions, the facts probe, binding probes v2-v5,
         ARC questions, PIQA goals, WinoGrande sentences (blank filled with each option)
  n = 13 for long items: HellaSwag contexts+endings, LAMBADA passages, HumanEval prompts+solutions,
         MBPP descriptions+code, CRUXEval code
A training DOCUMENT (span between <|endoftext|> tokens) is removed if any of its windows matches.
Cleaned pools are written to --out_root/<pool>/ (same shard format) with a decon_report.json.

Eval sets are loaded from parquet-backed HF datasets only (no dataset scripts are executed).

    python scripts/decontaminate.py index --out eval_index.npz
    python scripts/decontaminate.py clean --index eval_index.npz --pool /data/phase25/fact_seeking --out_root /data/phase25_decon
"""

import argparse
import glob
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

BASE = np.uint64(1000003)
NS = (8, 13)


def window_hashes(t, n):
    t = t.astype(np.uint64)
    m = len(t) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * BASE + t[j:j + m]
    return h


def eval_texts(probe_dir, facts_path):
    """(eval_name, n, text) for every item."""
    from datasets import load_dataset
    out = []
    ld = lambda *a, **k: load_dataset(*a, **k)  # noqa: E731
    for ex in ld("mandarjoshi/trivia_qa", "rc.nocontext", split="validation"):
        out.append(("triviaqa", 8, ex["question"]))
    for ex in ld("google-research-datasets/nq_open", split="validation"):
        out.append(("nq_open", 8, ex["question"]))
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        for ex in ld("allenai/ai2_arc", cfg, split="test"):
            out.append(("arc", 8, ex["question"]))
            for c in ex["choices"]["text"]:
                out.append(("arc", 8, ex["question"] + " " + c))
    for ex in ld("Rowan/hellaswag", split="validation"):
        for e in ex["endings"]:
            out.append(("hellaswag", 13, ex["ctx"] + " " + e))
    for ex in ld("baber/piqa", split="validation"):
        out.append(("piqa", 8, ex["goal"]))
        out.append(("piqa", 8, ex["goal"] + " " + ex["sol1"]))
        out.append(("piqa", 8, ex["goal"] + " " + ex["sol2"]))
    for ex in ld("allenai/winogrande", "winogrande_xl", split="validation"):
        for o in (ex["option1"], ex["option2"]):
            out.append(("winogrande", 8, ex["sentence"].replace("_", o)))
    for ex in ld("EleutherAI/lambada_openai", "default", split="test"):
        out.append(("lambada", 13, ex["text"]))
    for ex in ld("openai/openai_humaneval", split="test"):
        out.append(("humaneval", 13, ex["prompt"] + ex["canonical_solution"]))
    for ex in ld("google-research-datasets/mbpp", "full", split="test"):
        out.append(("mbpp", 13, ex["text"]))
        out.append(("mbpp", 13, ex["code"]))
    for ex in ld("cruxeval-org/cruxeval", split="test"):
        out.append(("cruxeval", 13, ex["code"]))
    for f in sorted(glob.glob(f"{probe_dir}/probefinding-winogrande-*.json")):
        if "185450" in f:  # superseded confounded first version
            continue
        tag = "probe_" + (Path(f).stem.split("-")[2] if "-v" in f else "v2")
        for c in json.loads(Path(f).read_text())["cases"]:
            out += [(tag, 8, c["ctxA"] + c["c1"]), (tag, 8, c["ctxB"] + c["c2"])]
    for it in json.loads(Path(facts_path).read_text())["items"]:
        out.append(("facts_probe", 8, it["q"]))
    return out


def cmd_index(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    items = eval_texts(args.probe_dir, args.facts)
    per_n = {n: {} for n in NS}
    counts = Counter()
    for name, n, text in items:
        ids = np.array(tok.encode(text, add_special_tokens=False), dtype=np.uint32)
        for h in np.unique(window_hashes(ids, n)).tolist():
            per_n[n].setdefault(h, name)
        counts[name] += 1
    arrays = {}
    for n in NS:
        keys = np.array(sorted(per_n[n]), dtype=np.uint64)
        arrays[f"h{n}"] = keys
        arrays[f"names{n}"] = np.array([per_n[n][k] for k in keys.tolist()])
    np.savez(args.out, **arrays)
    print(json.dumps({"items": dict(counts), "windows": {n: len(per_n[n]) for n in NS}}, indent=1))


def cmd_clean(args):
    idx = np.load(args.index)
    tables = {n: (idx[f"h{n}"], idx[f"names{n}"]) for n in NS}
    pool = Path(args.pool)
    shards = sorted(glob.glob(f"{pool}/shard_*.bin"))
    t0 = time.time()
    tokens = np.concatenate([np.fromfile(s, dtype=np.uint16) for s in shards])
    eot = np.flatnonzero(tokens == args.eot_id)
    n_docs = len(eot) + (1 if len(eot) == 0 or eot[-1] != len(tokens) - 1 else 0)
    bad_docs, by_eval = set(), Counter()
    chunk = 100_000_000
    for n, (keys, names) in tables.items():
        for s in range(0, len(tokens), chunk):
            h = window_hashes(tokens[s:s + chunk + n - 1], n)
            pos = np.flatnonzero(np.isin(h, keys))
            if not len(pos):
                continue
            starts = pos + s
            docs = np.searchsorted(eot, starts)  # document index of each hit window start
            names_hit = names[np.searchsorted(keys, h[pos])]
            for d, nm in zip(docs.tolist(), names_hit.tolist()):
                if d not in bad_docs:
                    by_eval[nm] += 1
                bad_docs.add(d)
    out = Path(args.out_root) / pool.name
    out.mkdir(parents=True, exist_ok=True)
    keep = np.ones(len(tokens), dtype=bool)
    starts = np.concatenate([[0], eot[:-1] + 1]) if len(eot) else np.array([0])
    ends = eot + 1 if len(eot) else np.array([len(tokens)])
    removed_tokens = 0
    for d in bad_docs:
        if d < len(starts):
            keep[starts[d]:ends[d]] = False
            removed_tokens += int(ends[d] - starts[d])
    clean = tokens[keep]
    shard_tokens = 100_000_000
    for i in range(0, len(clean), shard_tokens):
        clean[i:i + shard_tokens].tofile(out / f"shard_{i // shard_tokens:05d}.bin")
    meta_src = pool / "meta.json"
    meta = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    report = {"pool": str(pool), "docs": int(n_docs), "removed_docs": len(bad_docs), "removed_tokens": removed_tokens,
              "tokens_before": int(len(tokens)), "tokens_after": int(len(clean)),
              "removed_docs_by_eval": dict(by_eval.most_common()), "seconds": round(time.time() - t0)}
    meta.update({"total_tokens": int(len(clean)), "decontaminated": report, "n_shards": (len(clean) + shard_tokens - 1) // shard_tokens})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "decon_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--out", required=True)
    a.add_argument("--tokenizer", default="bubblspace-tokenizer")
    a.add_argument("--probe_dir", default="benchmark/data/probefinding-winograndestyle")
    a.add_argument("--facts", default="benchmark/data/facts-closedbook/facts-simple-v1-20261123.json")
    b = sub.add_parser("clean")
    b.add_argument("--index", required=True)
    b.add_argument("--pool", required=True)
    b.add_argument("--out_root", required=True)
    b.add_argument("--eot_id", type=int, default=0)
    args = ap.parse_args()
    cmd_index(args) if args.cmd == "index" else cmd_clean(args)


if __name__ == "__main__":
    main()
