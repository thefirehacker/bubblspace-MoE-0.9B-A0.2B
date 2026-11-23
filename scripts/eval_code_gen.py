"""HumanEval + MBPP, step 1 of 2: GENERATE completions only. Nothing is executed here.

Base-model style (no chat template):
  * HumanEval (164): prompt = signature + docstring; the model writes the body.
    Stops: "\\nclass", "\\ndef", "\\n#", "\\nif", "\\nprint" (Codex paper convention).
  * MBPP (500, full test split): prompt = '\"\"\"\\n{task}\\n{first test}\\n\"\"\"\\n' (bigcode-evaluation-harness
    style); the model writes the whole function.

Greedy decoding with a KV cache, deterministic (src/eval/determinism.py), so pass@1 is
reproducible. Output: one JSON with, per task, the completion and the assembled program
(prompt + completion + tests). Step 2, scripts/sandbox_exec.py, runs each program in an
isolated container; model-written code never runs on the host.

Usage:
    python scripts/eval_code_gen.py --checkpoint ckpt.pt --out benchmark/phase2/codegen_<tag>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset
from transformers import AutoTokenizer

from src.config import load_config
from src.eval.determinism import make_deterministic
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

HE_STOPS = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
MBPP_STOPS = ["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n```", "\n# Test", "\n# test"]


@torch.no_grad()
def complete(model, tok, prompt, max_new, stops, real_vocab, stop_ids, max_seq_len):
    ids = tok.encode(prompt, add_special_tokens=False)[-(max_seq_len - max_new):]
    x = torch.tensor([ids], device="cuda")
    cache = model.new_kv_cache()
    out = []
    for _ in range(max_new):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _, _ = model(x, kv_cache=cache)
        nxt = int(logits[0, -1, :real_vocab].argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        x = torch.tensor([[nxt]], device="cuda")
        if len(out) % 8 == 0 or nxt in NEWLINE_IDS:
            text = tok.decode(out)
            cut = min((text.find(s) for s in stops if s in text), default=-1)
            if cut >= 0:
                return text[:cut]
    text = tok.decode(out)
    cut = min((text.find(s) for s in stops if s in text), default=-1)
    return text[:cut] if cut >= 0 else text


NEWLINE_IDS = set()


def main():
    make_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="humaneval,mbpp")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    real_vocab = len(tok)
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|fim_pad|>")}
    NEWLINE_IDS.update(i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t]))

    model = Transformer(cfg.model).cuda().eval()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    step = ck.get("step") if isinstance(ck, dict) else None
    print(f"[ckpt] {args.checkpoint} (step {step})")

    results = {"checkpoint": args.checkpoint, "step": step, "tasks": {}}
    tasks = args.tasks.split(",")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    def save():  # after every benchmark, so an interrupted run keeps finished work
        Path(args.out).write_text(json.dumps(results, indent=1))

    if "humaneval" in tasks:
        ds = load_dataset("openai/openai_humaneval", split="test")
        rows, t0 = [], time.time()
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            # Token-boundary fix: HumanEval prompts end with a bare "\n", but this tokenizer merges
            # "\n" with the next line's indentation ("\n   " is one token). A lone trailing "\n" is
            # rare in training and the model answers it with <|endoftext|> (empty completions for
            # every task). Strip it and let the model produce "\n    ..." itself.
            prompt = ex["prompt"].rstrip("\n")
            comp = complete(model, tok, prompt, 512, HE_STOPS, real_vocab, stop_ids, cfg.model.max_seq_len)
            program = prompt + comp + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
            rows.append({"task_id": ex["task_id"], "completion": comp, "program": program})
        results["tasks"]["humaneval"] = rows
        save()
        print(f"[humaneval] {len(rows)} completions in {time.time() - t0:.0f}s", flush=True)

    if "mbpp" in tasks:
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
        rows, t0 = [], time.time()
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            prompt = f'"""\n{ex["text"]}\n{ex["test_list"][0]}\n"""\n'
            comp = complete(model, tok, prompt, 384, MBPP_STOPS, real_vocab, stop_ids, cfg.model.max_seq_len)
            setup = ex.get("test_setup_code") or ""
            program = (setup + "\n" if setup else "") + comp + "\n\n" + "\n".join(ex["test_list"]) + "\n"
            rows.append({"task_id": f"mbpp/{ex['task_id']}", "completion": comp, "program": program})
        results["tasks"]["mbpp"] = rows
        save()
        print(f"[mbpp] {len(rows)} completions in {time.time() - t0:.0f}s", flush=True)

    save()
    print(f"[codegen] saved {args.out} (NOT executed; run scripts/sandbox_exec.py)")


if __name__ == "__main__":
    main()
