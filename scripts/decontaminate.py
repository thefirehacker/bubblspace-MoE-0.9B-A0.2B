"""Decontaminate tokenized training pools against evaluation sets (Phase 2.5a).

Every evaluation item is tokenized with the training tokenizer and hashed into 13-token windows
(GPT-3-style). Items of 8-12 tokens (many TriviaQA / NQ questions) are matched as a whole; items under
8 tokens are skipped (too generic to match safely).

Frequency filter: an eval window that matches in more than --max_docs distinct training documents of a
pool is treated as a generic phrase (boilerplate imports, "which of the following ...") and ignored.
Real leakage of a specific question or solution shows up in only a few documents. The report lists
the most frequent ignored windows so the filter can be audited.

A training DOCUMENT (span between <|endoftext|> tokens) is removed if any non-generic window matches.
(v1 of this script used 8-token windows without a frequency filter: it removed 22% of Wikipedia docs,
mostly false positives such as Python files "matching" TriviaQA.)
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
LENGTHS = tuple(range(8, 14))  # 13-token windows; whole-item windows for 8-12-token items


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
    per_n = {n: {} for n in LENGTHS}
    counts, skipped = Counter(), Counter()
    for name, _, text in items:
        ids = np.array(tok.encode(text, add_special_tokens=False), dtype=np.uint32)
        if len(ids) < 8:
            skipped[name] += 1
            continue
        n = min(13, len(ids))
        for h in np.unique(window_hashes(ids, n)).tolist():
            per_n[n].setdefault(h, name)
        counts[name] += 1
    arrays = {}
    for n in LENGTHS:
        keys = np.array(sorted(per_n[n]), dtype=np.uint64)
        arrays[f"h{n}"] = keys
        arrays[f"names{n}"] = np.array([per_n[n][k] for k in keys.tolist()] or [""])[:len(keys)]
    np.savez(args.out, **arrays)
    print(json.dumps({"items": dict(counts), "skipped_under_8_tokens": dict(skipped),
                      "windows": {n: len(per_n[n]) for n in LENGTHS}}, indent=1))


def cmd_clean(args):
    idx = np.load(args.index)
    tables = {n: (idx[f"h{n}"], idx[f"names{n}"]) for n in LENGTHS if len(idx[f"h{n}"])}
    pool = Path(args.pool)
    shards = sorted(glob.glob(f"{pool}/shard_*.bin"))
    t0 = time.time()
    tokens = np.concatenate([np.fromfile(s, dtype=np.uint16) for s in shards])
    eot = np.flatnonzero(tokens == args.eot_id)
    n_docs = len(eot) + (1 if len(eot) == 0 or eot[-1] != len(tokens) - 1 else 0)
    # pass 1: collect every hit as (window length, hash, doc, position)
    hits = []
    chunk = 100_000_000
    for n, (keys, names) in tables.items():
        for s in range(0, len(tokens), chunk):
            h = window_hashes(tokens[s:s + chunk + n - 1], n)
            pos = np.flatnonzero(np.isin(h, keys))
            if len(pos):
                hits.append((n, h[pos], np.searchsorted(eot, pos + s), pos + s))
    # pass 2: frequency filter, keyed by (length, hash)
    doc_sets = {}
    first_pos = {}
    for n, hs, docs, poss in hits:
        for hv, d, p in zip(hs.tolist(), docs.tolist(), poss.tolist()):
            k = (n, hv)
            doc_sets.setdefault(k, set()).add(d)
            first_pos.setdefault(k, p)
    generic = {k for k, ds in doc_sets.items() if len(ds) > args.max_docs}
    bad_docs, by_eval = set(), Counter()
    for k, ds in doc_sets.items():
        if k in generic:
            continue
        n, hv = k
        keys, names = tables[n]
        nm = names[np.searchsorted(keys, np.uint64(hv))]
        new = ds - bad_docs
        by_eval[nm] += len(new)
        bad_docs |= ds
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    top_generic = sorted(generic, key=lambda k: -len(doc_sets[k]))[:12]
    generic_examples = [{"docs": len(doc_sets[k]), "eval": str(tables[k[0]][1][np.searchsorted(tables[k[0]][0], np.uint64(k[1]))]),
                         "text": tok.decode(tokens[first_pos[k]:first_pos[k] + k[0]].tolist())} for k in top_generic]
    real_examples = [{"docs": len(doc_sets[k]), "eval": str(tables[k[0]][1][np.searchsorted(tables[k[0]][0], np.uint64(k[1]))]),
                      "text": tok.decode(tokens[first_pos[k]:first_pos[k] + k[0]].tolist())}
                     for k in sorted((k for k in doc_sets if k not in generic), key=lambda k: -len(doc_sets[k]))[:8]]
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
              "removed_docs_by_eval": dict(by_eval.most_common()), "max_docs": args.max_docs,
              "generic_windows_ignored": len(generic), "top_generic_windows": generic_examples,
              "top_removed_windows": real_examples, "seconds": round(time.time() - t0)}
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
    b.add_argument("--max_docs", type=int, default=10, help="windows matching more docs than this are treated as generic")
    b.add_argument("--tokenizer", default="bubblspace-tokenizer")
    args = ap.parse_args()
    cmd_index(args) if args.cmd == "index" else cmd_clean(args)


if __name__ == "__main__":
    main()
