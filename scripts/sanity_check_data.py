"""Pre-training sanity checks for the BubblSpace tokenizer / tokenized shards.

Usage:
    python scripts/sanity_check_data.py --data_dir /lambda/nfs/h100-f1/data/debug \\
        [--web_file path.txt] [--python_file path.py] [--math_file path.txt]

Without --*_file args, falls back to small built-in samples of each type —
good enough to catch gross bugs (splitting, wrong ids) but not a substitute
for running this against real ~10MB held-out samples before a full run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
from transformers import AutoTokenizer

_BUILTIN_SAMPLES = {
    "web": (
        "The history of the internet began with the development of electronic computers "
        "in the 1950s. Early research into packet switching started in the early 1960s, "
        "and the first workable prototype of the Internet came in the late 1960s."
    ),
    "python": (
        "def fibonacci(n: int) -> int:\n"
        "    if n < 2:\n"
        "        return n\n"
        "    a, b = 0, 1\n"
        "    for _ in range(n - 1):\n"
        "        a, b = b, a + b\n"
        "    return b\n"
    ),
    "math": (
        "Let f(x) = x^2 + 3x - 5. The derivative is f'(x) = 2x + 3. "
        "Setting f'(x) = 0 gives x = -3/2, the vertex of the parabola. "
        "\\int_0^1 x^2 dx = 1/3."
    ),
}


def round_trip_and_bpt(tok, gpt2, label: str, text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    decoded = tok.decode(ids)
    ok = decoded == text
    gpt2_ids = gpt2.encode_ordinary(text)

    n_bytes = len(text.encode("utf-8"))
    bpt = n_bytes / max(1, len(ids))
    gpt2_bpt = n_bytes / max(1, len(gpt2_ids))

    print(
        f"[{label}] round_trip={'OK' if ok else 'MISMATCH'} "
        f"bubblspace_tokens={len(ids)} bytes/token={bpt:.3f} "
        f"gpt2_tokens={len(gpt2_ids)} gpt2_bytes/token={gpt2_bpt:.3f} "
        f"(ratio {bpt / gpt2_bpt:.3f}x)"
    )
    if not ok:
        print(f"  in:  {text!r}")
        print(f"  out: {decoded!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    parser.add_argument("--data_dir", default=None, help="if given, scans shards for max token id")
    parser.add_argument("--web_file", default=None)
    parser.add_argument("--python_file", default=None)
    parser.add_argument("--math_file", default=None)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    gpt2 = tiktoken.get_encoding("gpt2")

    print("== Round-trip + bytes-per-token vs GPT-2 ==")
    files = {"web": args.web_file, "python": args.python_file, "math": args.math_file}
    for label, path in files.items():
        text = Path(path).read_text() if path else _BUILTIN_SAMPLES[label]
        round_trip_and_bpt(tok, gpt2, label, text)

    print("\n== Special tokens never split ==")
    special = [
        "<|im_start|>", "<|im_end|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
        "<|fim_pad|>", "<tools>", "</tools>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>", "<think>", "</think>", "<|endoftext|>",
    ]
    for token in special:
        ids = tok.encode(token, add_special_tokens=False)
        status = "OK" if len(ids) == 1 else f"SPLIT INTO {ids}"
        print(f"  {token}: {status}")

    if args.data_dir:
        data_dir = Path(args.data_dir)
        with open(data_dir / "meta.json") as f:
            meta = json.load(f)
        vocab_size = meta.get("real_vocab_size")
        print(f"\n== Scanning shards in {data_dir} (real_vocab_size={vocab_size}) ==")
        max_id = -1
        for shard_path in sorted(data_dir.glob("shard_*.bin")):
            shard = np.memmap(shard_path, dtype=np.uint16, mode="r")
            max_id = max(max_id, int(shard.max()))
        print(f"max token id across all shards: {max_id}")
        assert max_id < vocab_size, f"found token id {max_id} >= real_vocab_size {vocab_size}"
        print("OK: all token ids within vocab range.")


if __name__ == "__main__":
    main()
