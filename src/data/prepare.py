"""Download + tokenize a HF dataset into flat uint16 token shards using the
BubblSpace tokenizer (SmolLM2 base + chat/FIM/tool/reasoning special tokens).

Usage:
    python -m src.data.prepare --out_dir /lambda/nfs/h100-f1/data/debug --max_tokens 150_000_000
    python -m src.data.prepare --out_dir /lambda/nfs/h100-f1/data/full --max_tokens 10_000_000_000

Shards are flat uint16 binary files (vocab 49280 < 65536, so 2 bytes/token
suffices). The debug run's data dir should be a short prefix of the same
dataset/tokenization as the full run — just stop this script early with a
smaller --max_tokens rather than tokenizing twice.

Pretraining text is tokenized as-is — chat/tool/think special tokens are
never inserted into raw web/code text. FIM tokens are inserted only when
--is_code and --fim_rate > 0 are both set (see src/data/fim.py).
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from src.data.fim import apply_fim

load_dotenv()

SHARD_TOKENS = 100_000_000  # ~200MB per shard at uint16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT", help="pass '' (empty string) for datasets with no named config, e.g. bigcode/starcoderdata")
    parser.add_argument("--data_dir", default=None, help="HF load_dataset data_dir (e.g. bigcode/starcoderdata language subfolder)")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--shard_tokens", type=int, default=SHARD_TOKENS)
    parser.add_argument("--is_code", action="store_true", help="enable FIM transform for this dataset")
    parser.add_argument("--fim_rate", type=float, default=0.0, help="probability of applying FIM per document (code only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_files", nargs="*", default=None,
                        help="only these repo files (e.g. the files that passed the HF security scan)")
    parser.add_argument("--min_int_score", type=int, default=None,
                        help="keep only examples with int_score >= this (FineWeb-Edu quality score)")
    parser.add_argument("--batch_docs", type=int, default=1000, help="documents per batched tokenizer call")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eot_id = tok.eos_token_id
    real_vocab_size = len(tok)
    rng = random.Random(args.seed)

    fim_ids = None
    if args.is_code and args.fim_rate > 0:
        fim_ids = (
            tok.convert_tokens_to_ids("<|fim_prefix|>"),
            tok.convert_tokens_to_ids("<|fim_middle|>"),
            tok.convert_tokens_to_ids("<|fim_suffix|>"),
        )

    ds = load_dataset(
        args.dataset,
        name=args.dataset_config or None,  # "" means "no named config" (e.g. bigcode/starcoderdata)
        data_dir=args.data_dir,
        data_files=args.data_files or None,
        split="train",
        streaming=True,
        token=os.environ.get("HF_TOKEN"),
    )

    buffer = np.empty(args.shard_tokens, dtype=np.uint16)
    buf_pos = 0
    shard_idx = 0
    total_written = 0
    pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

    def flush_shard():
        nonlocal shard_idx, buf_pos
        if buf_pos == 0:
            return
        path = out_dir / f"shard_{shard_idx:05d}.bin"
        buffer[:buf_pos].tofile(path)
        shard_idx += 1
        buf_pos = 0

    def write(ids):
        nonlocal buf_pos, total_written
        i = 0
        while i < len(ids) and total_written < args.max_tokens:
            space = args.shard_tokens - buf_pos
            take = min(space, len(ids) - i)
            buffer[buf_pos : buf_pos + take] = ids[i : i + take]
            buf_pos += take
            i += take
            total_written += take
            pbar.update(take)
            if buf_pos == args.shard_tokens:
                flush_shard()

    def encode_batch(texts):
        # one batched call: the Rust tokenizer parallelizes across documents
        for ids in tok(texts, add_special_tokens=False)["input_ids"]:
            if total_written >= args.max_tokens:
                return
            if fim_ids is not None and rng.random() < args.fim_rate:
                ids = apply_fim(ids, *fim_ids, rng)
            write(ids + [eot_id])

    batch, kept, seen = [], 0, 0
    for example in ds:
        if total_written >= args.max_tokens:
            break
        seen += 1
        if args.min_int_score is not None and example.get("int_score", 0) < args.min_int_score:
            continue
        kept += 1
        batch.append(example[args.text_field])
        if len(batch) >= args.batch_docs:
            encode_batch(batch)
            batch = []
    if batch and total_written < args.max_tokens:
        encode_batch(batch)

    flush_shard()
    pbar.close()

    meta = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "data_dir": args.data_dir,
        "tokenizer_path": args.tokenizer_path,
        "total_tokens": total_written,
        "n_shards": shard_idx,
        "shard_tokens": args.shard_tokens,
        "dtype": "uint16",
        "eot_token": eot_id,
        "real_vocab_size": real_vocab_size,
        "is_code": args.is_code,
        "fim_rate": args.fim_rate,
        "data_files": args.data_files,
        "min_int_score": args.min_int_score,
        "docs_seen": seen,
        "docs_kept": kept,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {total_written:,} tokens across {shard_idx} shard(s) to {out_dir}")


if __name__ == "__main__":
    main()
