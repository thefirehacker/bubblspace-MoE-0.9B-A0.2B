"""Phase 3 SFT data: inspect the downloaded datasets (schema, row counts, sample rows, token lengths with our tokenizer).

Step 1-2 of doc/plan-04-phase-3-sft.md §3.2. Read-only. Prints a report and writes inspect_report.json.

    python scripts/phase3_inspect.py --raw ~/phase3_raw --out ~/phase3_logs/inspect_report.json
"""
import argparse
import collections
import glob
import json
import os
import random

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def text_of(row):
    """Rough rendered text of a row, whatever its schema: concatenate message contents or string fields."""
    for key in ("messages", "conversations", "conversation", "chat", "prompt_messages"):
        if key in row and isinstance(row[key], list):
            parts = []
            for m in row[key]:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("value") or ""))
                else:
                    parts.append(str(m))
            return "\n".join(parts)
    return "\n".join(str(v) for v in row.values() if isinstance(v, str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--only", nargs="*", default=None, help="dataset folder names to inspect (default: all)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("bubblspace-tokenizer")
    rng = random.Random(0)
    report = {}
    for d in sorted(glob.glob(os.path.join(a.raw, "*"))):
        files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        if not files:
            continue
        name = os.path.basename(d)
        if a.only and name not in a.only:
            continue
        n_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        pf = pq.ParquetFile(files[0])
        schema = [f"{fld.name}: {fld.type}" for fld in pf.schema_arrow]
        # sample rows spread over the files
        rows = []
        per = max(1, a.sample // len(files))
        for f in files:
            # read in small batches: some agent-trace files have nested rows too large for one chunk
            got = []
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64):
                got += batch.to_pylist()
                if len(got) >= 4 * per:
                    break
            rows += rng.sample(got, min(per, len(got)))
        lens = sorted(len(tok.encode(text_of(r), add_special_tokens=False)) for r in rows)
        q = lambda p: lens[min(len(lens) - 1, int(p * len(lens)))]
        info = {
            "files": len(files), "rows": n_rows, "schema": schema,
            "tokens_p50_p90_p99_max": [q(0.5), q(0.9), q(0.99), lens[-1]],
            "frac_le_16k": round(sum(x <= 16000 for x in lens) / len(lens), 3),
            "frac_le_8k": round(sum(x <= 8000 for x in lens) / len(lens), 3),
            "est_total_tokens_B": round(n_rows * (sum(lens) / len(lens)) / 1e9, 3),
        }
        for key in ("source", "dataset", "subset", "task_type", "category"):
            if any(key in r for r in rows):
                info["by_" + key] = dict(collections.Counter(str(r.get(key)) for r in rows).most_common(25))
        ex = rows[0]
        info["example"] = {k: (str(v)[:600]) for k, v in ex.items()}
        report[name] = info
        print(f"\n=== {name}: {n_rows:,} rows in {len(files)} files; est {info['est_total_tokens_B']}B tokens")
        print("schema:", schema)
        print("tokens p50/p90/p99/max:", info["tokens_p50_p90_p99_max"], "| <=16k:", info["frac_le_16k"], "| <=8k:", info["frac_le_8k"])
        for k in info:
            if k.startswith("by_"):
                print(k, info[k])
        print("example:", json.dumps(info["example"], ensure_ascii=False)[:1200])
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
