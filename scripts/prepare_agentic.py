"""Format + pack Phase 2 Pool C (agentic tool-use + reasoning) data into the
BubblSpace wire format, hard-filtered to <= --seq_len tokens per episode and
packed into seq_len-aligned blocks (pad tail filled with <|fim_pad|>) so every
training window a sampler draws block-aligned (start = k * seq_len) contains a
complete episode from token 0 — never a random mid-conversation fragment.

Sources (all independently verified clean via HF's authenticated security API
before being wired in here — see doc/plan-02-phase-2-continual-training.md):
  - Nanbeige/ToolMind          (OpenAI-style role/content + tool_calls, raw .jsonl
                                 streamed directly to avoid datasets/PyArrow's
                                 schema-inference crash on this repo's heterogeneous
                                 nested `tools` JSON)
  - NousResearch/hermes-function-calling-v1 (ShareGPT-style from/value; system
                                 messages already carry <tools>...</tools> XML)
  - open-thoughts/OpenThoughts-114k (system + ShareGPT-style conversations; long
                                 reasoning traces average ~8k tokens, so the
                                 <= seq_len filter drops most rows here by design
                                 rather than truncating mid-<think>)

Wire format (matches the tokenizer's chat/tool/think special tokens):
  <|im_start|>system\n<tools>\n[...]\n</tools>\n<|im_end|>
  <|im_start|>user\n...\n<|im_end|>
  <|im_start|>assistant\n<think>\n...\n</think>\n<tool_call>\n{...}\n</tool_call><|im_end|>
  <|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>

Usage:
    python scripts/prepare_agentic.py --source toolmind --out_dir /lambda/nfs/.../phase2/agentic_toolmind
    python scripts/prepare_agentic.py --source hermes --out_dir /lambda/nfs/.../phase2/agentic_hermes
    python scripts/prepare_agentic.py --source openthoughts --out_dir /lambda/nfs/.../phase2/agentic_openthoughts
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

TOOLMIND_FILES = [
    "graph_syn_datasets/graphsyn.jsonl",
    "open_datasets/APIGen-MT-5k-query.jsonl",
    "open_datasets/BUTTONInstruct-query.jsonl",
    "open_datasets/ToolACE-query.jsonl",
    "open_datasets/When2Call-query.jsonl",
    "open_datasets/glaive-function-calling-v2-query.jsonl",
    "open_datasets/tau-train-query.jsonl",
    "open_datasets/xlam-function-calling-60k-query.jsonl",
]

HERMES_FILES = [
    "func-calling-singleturn.json",
    "func-calling.json",
    "glaive-function-calling-5k.json",
    "json-mode-agentic.json",
    "json-mode-singleturn.json",
]


def wrap(role: str, text: str) -> str:
    return f"<|im_start|>{role}\n{text}\n<|im_end|>"


def render_tool_call(call: dict) -> str:
    fn = call.get("function", call)
    payload = {"name": fn.get("name"), "arguments": fn.get("arguments")}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def iter_toolmind_episodes():
    for rel_path in TOOLMIND_FILES:
        local_path = hf_hub_download(
            repo_id="Nanbeige/ToolMind", filename=rel_path, repo_type="dataset"
        )
        with open(local_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                turns = []
                tools = row.get("tools")
                if tools:
                    turns.append(wrap("system", f"<tools>\n{json.dumps(tools, ensure_ascii=False)}\n</tools>"))

                for msg in row.get("conversations", []):
                    role = msg.get("role")
                    content = (msg.get("content") or "").strip()
                    tool_calls = msg.get("tool_calls")
                    if role == "user":
                        turns.append(wrap("user", content))
                    elif role == "tool":
                        turns.append(wrap("user", f"<tool_response>\n{content}\n</tool_response>"))
                    elif role == "assistant":
                        parts = [content] if content else []
                        for call in tool_calls or []:
                            parts.append(render_tool_call(call))
                        if parts:
                            turns.append(wrap("assistant", "\n".join(parts)))
                if turns:
                    yield "".join(turns)


def iter_hermes_episodes():
    for filename in HERMES_FILES:
        local_path = hf_hub_download(
            repo_id="NousResearch/hermes-function-calling-v1", filename=filename, repo_type="dataset"
        )
        with open(local_path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            turns = []
            for msg in row.get("conversations", []):
                role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}
                role = role_map.get(msg.get("from"), None)
                value = (msg.get("value") or "").strip()
                if role and value:
                    turns.append(wrap(role, value))
            if turns:
                yield "".join(turns)


def iter_openthoughts_episodes():
    local_path = hf_hub_download(
        repo_id="open-thoughts/OpenThoughts-114k",
        filename="data/train-00000-of-00006.parquet",
        repo_type="dataset",
    )
    import pyarrow.parquet as pq

    for part in range(6):
        path = hf_hub_download(
            repo_id="open-thoughts/OpenThoughts-114k",
            filename=f"data/train-0000{part}-of-00006.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(path)
        systems = table.column("system").to_pylist()
        conv_lists = table.column("conversations").to_pylist()
        for system, conversations in zip(systems, conv_lists):
            turns = []
            if system:
                turns.append(wrap("system", system.strip()))
            for msg in conversations:
                role = "user" if msg.get("from") == "user" else "assistant"
                value = (msg.get("value") or "")
                value = value.replace("<|begin_of_thought|>", "<think>").replace("<|end_of_thought|>", "</think>")
                value = value.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "")
                value = value.strip()
                if value:
                    turns.append(wrap(role, value))
            if turns:
                yield "".join(turns)


SOURCES = {
    "toolmind": iter_toolmind_episodes,
    "hermes": iter_hermes_episodes,
    "openthoughts": iter_openthoughts_episodes,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=list(SOURCES))
    parser.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--shard_tokens", type=int, default=102_400_000, help="50,000 blocks of 2048 tokens (~200MB/shard, matching prepare.py's convention); must be a multiple of --seq_len")
    args = parser.parse_args()
    assert args.shard_tokens % args.seq_len == 0, "--shard_tokens must be a multiple of --seq_len so block alignment (start = k * seq_len) holds across shard boundaries too"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eot_id = tok.eos_token_id
    pad_id = tok.convert_tokens_to_ids("<|fim_pad|>")
    real_vocab_size = len(tok)

    # One in-progress 2048-token block, plus a much larger shard buffer that
    # accumulates many completed blocks before a single shard file is written
    # (writing one tiny file per block does not scale — a full ToolMind pass
    # would be tens of thousands of 4KB files).
    block = np.full(args.seq_len, pad_id, dtype=np.uint16)
    block_pos = 0
    shard_buf = np.empty(args.shard_tokens, dtype=np.uint16)
    shard_buf_pos = 0
    shard_idx = 0
    total_blocks = 0
    kept_episodes = 0
    dropped_episodes = 0
    total_kept_tokens = 0

    def flush_shard():
        nonlocal shard_idx, shard_buf_pos
        if shard_buf_pos == 0:
            return
        path = out_dir / f"shard_{shard_idx:05d}.bin"
        shard_buf[:shard_buf_pos].tofile(path)
        shard_idx += 1
        shard_buf_pos = 0

    def flush_block():
        nonlocal block_pos, total_blocks, shard_buf_pos
        nonlocal_shard_space = args.shard_tokens - shard_buf_pos
        if nonlocal_shard_space < args.seq_len:
            flush_shard()
        shard_buf[shard_buf_pos : shard_buf_pos + args.seq_len] = block
        shard_buf_pos += args.seq_len
        total_blocks += 1
        block_pos = 0
        block[:] = pad_id

    pbar = tqdm(desc=f"[{args.source}] episodes", unit="ep")
    for text in SOURCES[args.source]():
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eot_id)
        pbar.update(1)

        if len(ids) > args.seq_len:
            dropped_episodes += 1
            continue
        kept_episodes += 1
        total_kept_tokens += len(ids)

        if block_pos + len(ids) > args.seq_len:
            flush_block()

        block[block_pos : block_pos + len(ids)] = ids
        block_pos += len(ids)

        if kept_episodes % 1000 == 0:
            pbar.set_postfix(kept=kept_episodes, dropped=dropped_episodes, blocks=total_blocks)

    if block_pos > 0:
        flush_block()
    flush_shard()
    pbar.close()

    meta = {
        "source": args.source,
        "tokenizer_path": args.tokenizer_path,
        "seq_len": args.seq_len,
        "block_aligned": True,
        "pad_token_id": pad_id,
        "eot_token": eot_id,
        "real_vocab_size": real_vocab_size,
        "n_shards": shard_idx,
        "kept_episodes": kept_episodes,
        "dropped_episodes_over_seq_len": dropped_episodes,
        "total_kept_tokens_excl_padding": total_kept_tokens,
        "total_blocks": total_blocks,
        "total_tokens_incl_padding": total_blocks * args.seq_len,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[{args.source}] kept {kept_episodes:,} episodes ({total_kept_tokens:,} real tokens), "
        f"dropped {dropped_episodes:,} over {args.seq_len} tokens, "
        f"wrote {total_blocks:,} blocks across {shard_idx} shard(s) to {out_dir}"
    )


if __name__ == "__main__":
    main()
