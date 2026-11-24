"""Extract long documents from a tokenized pool into a new pool (Phase 2.5b context extension).

Reads uint16 shards (documents separated by <|endoftext|>), keeps every document with at least
--min_tokens tokens (whole, including its EOT), and writes them to --out as standard shards with a
meta.json. Used to build the long-document share of the 16k mix from pools already on disk.

    python scripts/extract_long_docs.py --pool /data/phase25_decon/en_wiki_full --out /data/phase25b/long_wiki --min_tokens 4096
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_tokens", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=0, help="stop after this many long-doc tokens (0 = all)")
    ap.add_argument("--eot_id", type=int, default=0)
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    buf, nbuf, shard, kept_tokens, kept_docs, seen_docs = [], 0, 0, 0, 0, 0
    carry = np.empty(0, dtype=np.uint16)
    hist = {"4k-8k": 0, "8k-16k": 0, "16k+": 0}

    def flush(final=False):
        nonlocal buf, nbuf, shard
        if nbuf >= args.shard_tokens or (final and nbuf):
            np.concatenate(buf).tofile(out / f"shard_{shard:05d}.bin")
            shard += 1
            buf, nbuf = [], 0

    for f in sorted(glob.glob(f"{args.pool}/shard_*.bin")):
        a = np.concatenate([carry, np.fromfile(f, dtype=np.uint16)])
        eot = np.flatnonzero(a == args.eot_id)
        start = 0
        for e in eot:
            seen_docs += 1
            n = e + 1 - start
            if n >= args.min_tokens:
                n = int(n); buf.append(a[start:e + 1].copy()); nbuf += n; kept_tokens += n; kept_docs += 1
                hist["16k+" if n >= 16384 else "8k-16k" if n >= 8192 else "4k-8k"] += int(n)
                flush()
            start = e + 1
            if args.max_tokens and kept_tokens >= args.max_tokens:
                break
        carry = a[start:]
        if args.max_tokens and kept_tokens >= args.max_tokens:
            break
    flush(final=True)
    meta = {"source_pool": args.pool, "min_tokens": args.min_tokens, "total_tokens": int(kept_tokens), "docs": kept_docs,
            "docs_scanned": seen_docs, "tokens_by_length": hist, "n_shards": shard, "dtype": "uint16", "eot_token": args.eot_id}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
