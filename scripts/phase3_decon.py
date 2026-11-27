"""Phase 3 SFT data: decontaminate the packed SFT pools (shard_*.bin + mask_*.bin) against evaluation sets.

Same matching as scripts/decontaminate.py (13-token windows, whole-item windows for 8-12-token items, frequency filter
for generic phrases), with the eval list extended for SFT: BFCL (simple/multiple/parallel/parallel_multiple questions),
IFEval prompts, the GSM8K test split, and our hand-test prompts (doc/handtest-01).

SFT pools are block-packed with loss masks, so a contaminated EPISODE (span ending in <|endoftext|>) is blanked in
place: its tokens become <|fim_pad|> and its mask 0. Block alignment and masks stay valid; nothing is repacked.

    python scripts/phase3_decon.py index --out ~/phase3_logs/sft_eval_index.npz --handtests /tmp/rel18.txt /tmp/calc16.txt
    python scripts/phase3_decon.py clean --index ~/phase3_logs/sft_eval_index.npz --pool ~/phase3_build/tulu3 --out_root ~/phase3_decon
"""
import argparse
import glob
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decontaminate import LENGTHS, eval_texts, window_hashes  # noqa: E402


def extra_texts(handtests):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    out = []
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    for cat in ("simple", "multiple", "parallel", "parallel_multiple"):
        for line in open(hf_hub_download(repo, f"BFCL_v3_{cat}.json", repo_type="dataset")):
            q = json.loads(line)["question"]
            turns = q[0] if q and isinstance(q[0], list) else q
            for m in turns:
                if isinstance(m, dict) and m.get("content"):
                    out.append(("bfcl", 8, m["content"]))
    for ex in load_dataset("google/IFEval", split="train"):
        out.append(("ifeval", 13, ex["prompt"]))
    for ex in load_dataset("openai/gsm8k", "main", split="test"):
        out.append(("gsm8k_test", 13, ex["question"]))
    for f in handtests:
        for line in open(f):
            if line.strip():
                out.append(("handtest", 8, line.strip()))
    return out


def cmd_index(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    items = eval_texts(a.probe_dir, a.facts) + extra_texts(a.handtests)
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
    np.savez(a.out, **arrays)
    print(json.dumps({"items": dict(counts), "skipped_under_8_tokens": dict(skipped)}, indent=1))


def cmd_clean(a):
    idx = np.load(a.index)
    tables = {n: (idx[f"h{n}"], idx[f"names{n}"]) for n in LENGTHS if len(idx[f"h{n}"])}
    pool = Path(a.pool)
    meta = json.loads((pool / "meta.json").read_text())
    pad, eot = meta["pad_token_id"], meta["eot_token"]
    shards = sorted(glob.glob(f"{pool}/shard_*.bin"))
    t0 = time.time()
    tokens = np.concatenate([np.fromfile(s, dtype=np.uint16) for s in shards])
    masks = np.concatenate([np.fromfile(s.replace("shard_", "mask_"), dtype=np.uint8) for s in shards])
    eot_pos = np.flatnonzero(tokens == eot)
    hits = []
    chunk = 100_000_000
    for n, (keys, names) in tables.items():
        for s in range(0, len(tokens), chunk):
            h = window_hashes(tokens[s:s + chunk + n - 1], n)
            pos = np.flatnonzero(np.isin(h, keys))
            if len(pos):
                hits.append((n, h[pos], np.searchsorted(eot_pos, pos + s)))
    doc_sets = {}
    for n, hs, docs in hits:
        for hv, d in zip(hs.tolist(), docs.tolist()):
            doc_sets.setdefault((n, hv), set()).add(d)
    generic = {k for k, ds in doc_sets.items() if len(ds) > a.max_docs}
    bad, by_eval = set(), Counter()
    for k, ds in doc_sets.items():
        if k in generic:
            continue
        keys, names = tables[k[0]]
        by_eval[str(names[np.searchsorted(keys, np.uint64(k[1]))])] += len(ds - bad)
        bad |= ds
    # episode d spans (eot_pos[d-1], eot_pos[d]]; pads between episodes are skipped (they have mask 0 anyway)
    removed_tokens = 0
    for d in bad:
        if d >= len(eot_pos):
            continue
        end = eot_pos[d] + 1
        start = eot_pos[d - 1] + 1 if d > 0 else 0
        seg = tokens[start:end]
        first_real = start + int(np.argmax(seg != pad)) if (seg != pad).any() else end
        tokens[first_real:end] = pad
        masks[first_real:end] = 0
        removed_tokens += end - first_real
    out = Path(a.out_root) / pool.name
    out.mkdir(parents=True, exist_ok=True)
    off = 0
    for s in shards:
        n = len(np.memmap(s, dtype=np.uint16, mode="r"))
        tokens[off:off + n].tofile(out / Path(s).name)
        masks[off:off + n].tofile(out / Path(s).name.replace("shard_", "mask_"))
        off += n
    report = {"pool": str(pool), "episodes": int(len(eot_pos)), "removed_episodes": len(bad), "removed_tokens": int(removed_tokens),
              "removed_by_eval": dict(by_eval.most_common()), "generic_windows_ignored": len(generic), "max_docs": a.max_docs,
              "seconds": round(time.time() - t0)}
    meta["decontaminated"] = report
    meta["target_tokens_after_decon"] = int(masks.sum())
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "decon_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--out", required=True)
    i.add_argument("--handtests", nargs="*", default=[])
    i.add_argument("--tokenizer", default="bubblspace-tokenizer")
    i.add_argument("--probe_dir", default="benchmark/data/probefinding-winograndestyle")
    i.add_argument("--facts", default="benchmark/data/facts-closedbook/facts-simple-v1-20261123.json")
    c = sub.add_parser("clean")
    c.add_argument("--index", required=True)
    c.add_argument("--pool", required=True)
    c.add_argument("--out_root", required=True)
    c.add_argument("--max_docs", type=int, default=10)
    a = ap.parse_args()
    cmd_index(a) if a.cmd == "index" else cmd_clean(a)


if __name__ == "__main__":
    main()
