"""Document-length distribution of tokenized pools (uint16 shards, documents separated by <|endoftext|>).

For each pool: number of documents, and how many documents / what share of tokens sit in documents
longer than 2k, 4k, 8k and 16k tokens. Used to size the long-document share of the Phase 2.5
context-extension mix.

    python scripts/measure_doc_lengths.py --pools /data/phase2/code_python_edu /data/phase2/en_wiki \
        --max_tokens 300000000 --out doc_lengths.json
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

THRESHOLDS = (2048, 4096, 8192, 16384)


def pool_lengths(pool, eot_id, max_tokens):
    """Lengths (incl. the closing EOT) of complete documents, reading shards in order up to max_tokens."""
    lengths, carry, read = [], 0, 0
    for shard in sorted(glob.glob(f"{pool}/shard_*.bin")):
        arr = np.memmap(shard, dtype=np.uint16, mode="r")
        if max_tokens and read + len(arr) > max_tokens:
            arr = arr[: max_tokens - read]
        pos = np.flatnonzero(arr == eot_id)
        if len(pos):
            first = carry + pos[0] + 1
            lengths.append(np.concatenate([[first], np.diff(pos)]))
            carry = len(arr) - pos[-1] - 1
        else:
            carry += len(arr)
        read += len(arr)
        if max_tokens and read >= max_tokens:
            break
    return (np.concatenate(lengths) if lengths else np.empty(0, dtype=np.int64)), read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", required=True)
    ap.add_argument("--eot_id", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=300_000_000, help="per pool; 0 = all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {}
    for pool in args.pools:
        L, read = pool_lengths(pool, args.eot_id, args.max_tokens)
        if len(L) == 0:
            report[Path(pool).name] = {"scanned_tokens": int(read), "docs": 0}
            continue
        total = int(L.sum())
        r = {"scanned_tokens": int(read), "docs": int(len(L)), "median": int(np.median(L)),
             "p90": int(np.percentile(L, 90)), "p99": int(np.percentile(L, 99)), "max": int(L.max())}
        for t in THRESHOLDS:
            m = L > t
            r[f"docs_gt_{t}"] = int(m.sum())
            r[f"token_share_gt_{t}"] = round(float(L[m].sum()) / total, 4)
        report[Path(pool).name] = r
        print(f"{Path(pool).name:<28} docs {r['docs']:>9,}  median {r['median']:>6}  p99 {r['p99']:>7}  "
              + "  ".join(f">{t // 1024}k {r[f'docs_gt_{t}']:>7,} ({r[f'token_share_gt_{t}']:.1%} tok)" for t in THRESHOLDS),
              flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
