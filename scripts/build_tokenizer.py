"""Build the BubblSpace tokenizer: SmolLM2-360M's tokenizer (vocab 49,152)
plus chat/FIM/tool/reasoning special tokens. Does NOT train a new tokenizer
— only extends the existing vocabulary with reserved special tokens.

Usage:
    python scripts/build_tokenizer.py --out_dir bubblspace-tokenizer
"""

import argparse

from transformers import AutoTokenizer

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"

SPECIAL_TOKENS = [
    # Chat
    "<|im_start|>", "<|im_end|>",
    # FIM
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    # Tools
    "<tools>", "</tools>", "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
    # Reasoning
    "<think>", "</think>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="bubblspace-tokenizer")
    parser.add_argument("--pad_multiple", type=int, default=128)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_len = len(tok)
    print(f"Base tokenizer ({BASE_MODEL}): {base_len} tokens")

    existing_vocab = tok.get_vocab()
    to_add = [t for t in SPECIAL_TOKENS if t not in existing_vocab]
    skipped = [t for t in SPECIAL_TOKENS if t in existing_vocab]
    if skipped:
        print(f"Already present, skipping: {skipped}")

    if to_add:
        n_added = tok.add_special_tokens({"additional_special_tokens": to_add})
        print(f"Added {n_added} new special tokens: {to_add}")

    new_len = len(tok)
    padded_vocab_size = ((new_len + args.pad_multiple - 1) // args.pad_multiple) * args.pad_multiple
    print(f"Tokenizer length after additions: {new_len}")
    print(f"Padded model vocab_size (multiple of {args.pad_multiple}): {padded_vocab_size}")

    # Sanity checks before saving.
    for token in SPECIAL_TOKENS:
        ids = tok.encode(token, add_special_tokens=False)
        assert len(ids) == 1, f"special token {token!r} was split into multiple ids: {ids}"

    sample = "The quick brown fox jumps over the lazy dog. <tool_call>{\"name\": \"x\"}</tool_call>"
    decoded = tok.decode(tok.encode(sample, add_special_tokens=False))
    assert decoded == sample, f"round-trip mismatch:\n  in:  {sample!r}\n  out: {decoded!r}"

    tok.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}/ (real_vocab_size={new_len}, model.vocab_size should be set to {padded_vocab_size})")


if __name__ == "__main__":
    main()
