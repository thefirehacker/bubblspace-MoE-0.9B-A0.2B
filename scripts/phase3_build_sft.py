"""Phase 3 SFT data: render conversations into our wire format, mask the loss to assistant turns, pack into 16k blocks.

doc/plan-04-phase-3-sft.md §3.1-3.2. Runs on the A10 (multiprocessing over its cores). One output pool per source:

    <out>/<pool>/shard_00000.bin   uint16 tokens, whole episodes packed into --seq_len blocks, tail padded with <|fim_pad|>
    <out>/<pool>/mask_00000.bin    uint8, same length: 1 = this token is a training target (assistant turns, incl. <|im_end|>)
    <out>/<pool>/meta.json         seq_len, block_aligned, loss_mask, pad/eot ids, counts

Wire format = the one the model saw in pretraining (scripts/prepare_agentic.py):
    <|im_start|>{role}\n{text}\n<|im_end|>   turns concatenated, episode followed by <|endoftext|>
Tool calls / responses stay inside the text (<tool_call>…</tool_call>, <tool_response>…</tool_response>).

    python scripts/phase3_build_sft.py --source tulu3 --raw ~/phase3_raw --out /lambda/nfs/us-west-1-eval-inference/phase3
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROLE_OK = {"system", "user", "assistant", "tool"}
_TOK = None


# ---------------------------------------------------------------- sources: each yields a list of (role, text) turns
def _parquet_rows(pattern, columns=None, batch=256):
    for f in sorted(glob.glob(pattern, recursive=True)):
        for b in pq.ParquetFile(f).iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def src_tulu3(raw):
    for r in _parquet_rows(f"{raw}/allenai__tulu-3-sft-olmo-2-mixture-0225/data/*.parquet"):
        if "aya" in (r.get("source") or ""):  # multilingual subset: off target for this model
            continue
        yield [(m["role"], m["content"]) for m in r["messages"]]


# Run 02 (27 Nov): run 01 over-refused harmless commands ("give me the capital of India" -> "I can't provide real-time
# data"). 5.1% of Tülu replies open with a refusal, nearly all from three safety sets. Drop two of them, keep a 10% sample
# of WildGuardMix (refusals of clearly harmful requests + answers to harmless-but-edgy ones), drop refusal-opening
# replies elsewhere, and drop OLMo's hard-coded identity rows ("I am OLMo, made by Ai2").
_REFUSAL_OPEN = re.compile(r"^\s*(I'?m sorry|I am sorry|I cannot|I can'?t|I apologi[sz]e|As an AI|Sorry,|Unfortunately, I)", re.I)
RUN02_DROP_SOURCES = ("wildjailbreak", "coconot", "olmo_hardcoded")


def src_tulu3_run02(raw):
    for r in _parquet_rows(f"{raw}/allenai__tulu-3-sft-olmo-2-mixture-0225/data/*.parquet"):
        s = r.get("source") or ""
        if "aya" in s or any(d in s for d in RUN02_DROP_SOURCES):
            continue
        turns = [(m["role"], m["content"]) for m in r["messages"]]
        if "wildguardmix" in s:
            if int(hashlib.md5(json.dumps(turns).encode()).hexdigest(), 16) % 10:  # deterministic 10% sample
                continue
        else:
            first = next((t for ro, t in turns if ro == "assistant"), "") or ""
            if _REFUSAL_OPEN.match(first):
                continue
        yield turns


# Run 02b (27 Nov): the run 02 smoke test fixed over-refusal (harmless refused 12/48 -> 1/48) but also stopped refusing
# clearly harmful requests (8/8 -> 0/8). Run 02b keeps ALL of WildGuardMix (refusals of harmful prompts + answers to
# harmless prompts that sound harmful: the contrast teaches the difference), adds 10% of WildJailbreak, and 10% of
# CoCoNot after removing its "underspecified" / "real-time" replies (the ones behind "capital of India -> can't provide
# real-time data"). Refusal-opening replies in the other sources and OLMo's identity rows stay out, as in run 02.
_COCONOT_OVER = re.compile(r"real-time|real time|current|up-to-date|up to date|underspecified|context|incomplete|unclear|"
                           r"ambiguous|vague|lacks|more (information|details)|clarify|specify|which .* (are|do) you "
                           r"(referring|mean)|as of my (last|knowledge)|knowledge cutoff|browse|internet", re.I)


def _hash10(turns):
    return int(hashlib.md5(json.dumps(turns).encode()).hexdigest(), 16) % 10 == 0


def src_tulu3_run02b(raw):
    for r in _parquet_rows(f"{raw}/allenai__tulu-3-sft-olmo-2-mixture-0225/data/*.parquet"):
        s = r.get("source") or ""
        if "aya" in s or "olmo_hardcoded" in s:
            continue
        turns = [(m["role"], m["content"]) for m in r["messages"]]
        first = next((t for ro, t in turns if ro == "assistant"), "") or ""
        if "wildguardmix" in s:
            pass                                    # all of it
        elif "wildjailbreak" in s:
            if not _hash10(turns):                  # 10% sample
                continue
        elif "coconot" in s:
            if _COCONOT_OVER.search(first[:400]) or not _hash10(turns):  # filtered, then 10% sample
                continue
        elif _REFUSAL_OPEN.match(first):
            continue
        yield turns


def src_gsm8k(raw):
    for r in _parquet_rows(f"{raw}/openai__gsm8k/main/train-*.parquet"):  # train only; test stays an eval
        ans = re.sub(r"<<[^>]*>>", "", r["answer"])
        ans = re.sub(r"\n?####\s*(.+)\s*$", r"\nThe answer is \1.", ans.strip())
        yield [("user", r["question"]), ("assistant", ans)]


_BOXED_INSTR = re.compile(r"\n*Return your final response as .*$", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def src_synthetic1(raw):
    for r in _parquet_rows(f"{raw}/PrimeIntellect__SYNTHETIC-1/data/*.parquet"):
        prompt = _BOXED_INSTR.sub("", r["prompt"] or "").strip()
        tt = r.get("task_type")
        if tt == "verifiable_code":  # verified model answers only, reasoning trace removed (short-answer style)
            if (r.get("score") or 0) < 1:
                continue
            ans = _THINK.sub("", r.get("llm_response") or "").strip()
        else:  # maths, code-output prediction, StackExchange: the gold solution (short worked steps)
            ans = (r.get("gold_standard_solution") or "").strip()
        if prompt and len(ans) > 20:
            yield [("user", prompt), ("assistant", ans)]


def src_terminal(raw, cap=20000):
    n = 0
    for r in _parquet_rows(f"{raw}/nvidia__Nemotron-Terminal-Corpus/**/*.parquet", columns=["conversations"], batch=16):
        yield [(m["role"], m["content"]) for m in r["conversations"]]
        n += 1
        if n >= cap * 3:  # length filter keeps ≈47%; stop reading early
            return


def src_agentic(raw, which):
    import prepare_agentic as pa  # reuse the Phase 2 readers; their rows are already rendered in the wire format
    # APIGen-MT-5k is CC-BY-NC-4.0 (non-commercial): keep it out of the SFT data (27 Nov)
    pa.TOOLMIND_FILES = [f for f in pa.TOOLMIND_FILES if "APIGen-MT" not in f]
    it = {"toolmind": pa.iter_toolmind_episodes, "hermes": pa.iter_hermes_episodes,
          "glaive": pa.iter_glaive_fc_v2_episodes}[which]()
    seg = re.compile(r"<\|im_start\|>(\w+)\n(.*?)\n<\|im_end\|>", re.S)
    for text in it:
        turns = seg.findall(text)
        if turns:
            yield turns


_GLAIVE_TURN = re.compile(r"(?:^|\n+)(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
_GLAIVE_CALL = re.compile(r"""^\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*'(.*)'\s*\}\s*$""", re.S)


def _glaive_call(raw_call):
    """Glaive writes `{"name": "f", "arguments": '{...}'}` (arguments as a single-quoted string, not JSON).
    Normalise to the JSON our other tool sources use; keep the raw text if it can't be parsed."""
    m = _GLAIVE_CALL.match(raw_call)
    if m:
        try:
            return json.dumps({"name": m.group(1), "arguments": json.loads(m.group(2))}, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return raw_call.strip()


def src_glaive(raw):
    """glaiveai/glaive-function-calling-v2, parsed robustly: turns are separated by two OR three newlines
    (the Phase 2 reader split only on three and lost the assistant turn of ≈30% of rows)."""
    from huggingface_hub import hf_hub_download
    rows = json.load(open(hf_hub_download("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json",
                                          repo_type="dataset"), encoding="utf-8"))
    for row in rows:
        system = (row.get("system") or "").strip()
        system = system[len("SYSTEM: "):] if system.startswith("SYSTEM: ") else system
        turns = []
        if system:
            turns.append(("system", f"<tools>\n{system}\n</tools>" if "{" in system else system))
        parts = _GLAIVE_TURN.split((row.get("chat") or "").strip())[1:]
        for marker, content in zip(parts[0::2], parts[1::2]):
            content = content.replace("<|endoftext|>", "").strip()
            if not content:
                continue
            if marker == "USER":
                turns.append(("user", content))
            elif marker == "FUNCTION RESPONSE":
                turns.append(("user", f"<tool_response>\n{content}\n</tool_response>"))
            elif "<functioncall>" in content:
                turns.append(("assistant", f"<tool_call>\n{_glaive_call(content.split('<functioncall>', 1)[1])}\n</tool_call>"))
            else:
                turns.append(("assistant", content))
        if turns:
            yield turns


def src_own(raw, name):
    """Our generated sets (scripts/phase3_gen_own_sets.py): {"turns": [[role, text], ...]} per line."""
    with open(f"{raw}/own/{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            yield [tuple(t) for t in json.loads(line)["turns"]]


SOURCES = {
    "own_facts": lambda raw: src_own(raw, "own_facts"),
    "own_math": lambda raw: src_own(raw, "own_math"),
    "own_binding": lambda raw: src_own(raw, "own_binding"),
    "own_identity": lambda raw: src_own(raw, "own_identity"),
    "own_facts_cmd": lambda raw: src_own(raw, "own_facts_cmd"),
    "tulu3_run02": src_tulu3_run02,
    "tulu3_run02b": src_tulu3_run02b,
    "own_realtime": lambda raw: src_own(raw, "own_realtime"),
    "tulu3": src_tulu3, "gsm8k": src_gsm8k, "synthetic1": src_synthetic1, "terminal": src_terminal,
    "agentic_toolmind": lambda raw: src_agentic(raw, "toolmind"),
    "agentic_hermes": lambda raw: src_agentic(raw, "hermes"),
    "agentic_glaive": src_glaive,
}


# ---------------------------------------------------------------- render + mask (worker side)
def _init():
    global _TOK
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained("bubblspace-tokenizer")


def render(turns):
    """turns -> (ids uint16, mask uint8) or None. Mask is 1 on assistant text + its <|im_end|>, 0 elsewhere."""
    turns = [(("user" if r == "tool" else r), (t or "").strip()) for r, t in turns if r in ROLE_OK]
    if not any(r == "assistant" and t for r, t in turns):
        return None
    ids, mask = [], []
    for role, text in turns:
        if not text:
            return None  # empty turn: broken row
        head = _TOK.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        body = _TOK.encode(f"{text}\n<|im_end|>", add_special_tokens=False)
        ids += head + body
        mask += [0] * len(head) + ([1] * len(body) if role == "assistant" else [0] * len(body))
    ids.append(_TOK.eos_token_id)
    mask.append(0)
    return np.asarray(ids, dtype=np.uint16), np.asarray(mask, dtype=np.uint8)


# ---------------------------------------------------------------- packing (main process)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=16384)
    ap.add_argument("--blocks_per_shard", type=int, default=6000)  # ≈98M tokens per shard
    ap.add_argument("--max_episodes", type=int, default=0, help="stop after this many kept episodes (0 = all)")
    ap.add_argument("--short_keep", type=float, default=1.0,
                    help="agentic sources: keep this fraction of episodes <= 2048 tokens (those were in Phase 2 pretraining)")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bubblspace-tokenizer")
    pad, eot, L = tok.convert_tokens_to_ids("<|fim_pad|>"), tok.eos_token_id, a.seq_len
    out = Path(a.out) / a.source
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    blk_ids, blk_mask, pos = np.full(L, pad, np.uint16), np.zeros(L, np.uint8), 0
    sh_ids, sh_mask, n_blk, shard_idx = [], [], 0, 0
    stats = dict(rows=0, kept=0, dropped_long=0, dropped_broken=0, dropped_dup=0, dropped_short_sampled=0,
                 kept_tokens=0, target_tokens=0, blocks=0)
    seen = set()

    def flush_shard():
        nonlocal sh_ids, sh_mask, shard_idx
        if sh_ids:
            np.concatenate(sh_ids).tofile(out / f"shard_{shard_idx:05d}.bin")
            np.concatenate(sh_mask).tofile(out / f"mask_{shard_idx:05d}.bin")
            shard_idx += 1
        sh_ids, sh_mask = [], []

    def flush_block():
        nonlocal blk_ids, blk_mask, pos, n_blk
        sh_ids.append(blk_ids)
        sh_mask.append(blk_mask)
        stats["blocks"] += 1
        n_blk += 1
        blk_ids, blk_mask, pos = np.full(L, pad, np.uint16), np.zeros(L, np.uint8), 0
        if n_blk % a.blocks_per_shard == 0:
            flush_shard()

    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        for res in pool.imap(render, SOURCES[a.source](a.raw), chunksize=64):
            stats["rows"] += 1
            if res is None:
                stats["dropped_broken"] += 1
                continue
            ids, mask = res
            if len(ids) > L:
                stats["dropped_long"] += 1
                continue
            if a.source.startswith("agentic") and len(ids) <= 2048 and rng.random() > a.short_keep:
                stats["dropped_short_sampled"] += 1
                continue
            h = hashlib.md5(ids.tobytes()).digest()
            if h in seen:
                stats["dropped_dup"] += 1
                continue
            seen.add(h)
            if pos + len(ids) > L:
                flush_block()
            blk_ids[pos:pos + len(ids)] = ids
            blk_mask[pos:pos + len(ids)] = mask
            pos += len(ids)
            stats["kept"] += 1
            stats["kept_tokens"] += len(ids)
            stats["target_tokens"] += int(mask.sum())
            if a.max_episodes and stats["kept"] >= a.max_episodes:
                break
            if stats["rows"] % 50000 == 0:
                print(f"[{a.source}] rows {stats['rows']:,} kept {stats['kept']:,} blocks {stats['blocks']:,} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        pool.terminate()
    if pos:
        flush_block()
    flush_shard()
    meta = dict(source=a.source, seq_len=L, block_aligned=True, loss_mask=True, pad_token_id=int(pad), eot_token=int(eot),
                real_vocab_size=len(tok), dtype="uint16", mask_dtype="uint8", n_shards=shard_idx,
                total_tokens_incl_padding=stats["blocks"] * L, seconds=round(time.time() - t0), **stats)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
