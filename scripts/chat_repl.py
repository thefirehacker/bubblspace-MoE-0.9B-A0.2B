"""Interactive command-line REPL for continuously testing prompts against a
checkpoint -- no need to re-launch a script per prompt.

This is a BASE model (no SFT yet): it continues text rather than following
instructions or answering questions directly. Short declarative prompts
("The capital of France is", "def fibonacci(n):") work much better than
chat-style questions.

Usage (point --checkpoint at any .pt or .safetensors file, including the
WSP-edited ones under wsp_checkpoints/):
    python scripts/chat_repl.py --checkpoint /path/to/step_19073.pt

REPL commands:
    /temp <float>      set sampling temperature (default 0.8)
    /topp <float>      set nucleus top-p (default 0.95)
    /tokens <int>      set max new tokens per response (default 100)
    /greedy            switch to greedy decoding (argmax, deterministic)
    /sample            switch back to temperature+top-p sampling
    exit / quit         stop
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer


def load_weights(model: Transformer, checkpoint_path: str):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        model.load_state_dict(load_file(checkpoint_path))
        return None
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return ckpt.get("step")


@torch.no_grad()
def generate(model, tok, device, real_vocab_size, eot_id, max_seq_len, prompt, max_new_tokens, temperature, top_p, greedy):
    # KV cache: one prefill pass over the prompt, then one token per step.
    # RoPE is precomputed for max_seq_len positions, so the prompt is trimmed
    # to leave room for the generated tokens.
    ids = tok.encode(prompt, add_special_tokens=False)[-(max_seq_len - max_new_tokens):]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache = model.new_kv_cache()

    generated = []
    for _ in range(max_new_tokens):
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=(device == "cuda")):
            logits, _, _, _ = model(input_ids, kv_cache=cache)
        next_logits = logits[0, -1, :real_vocab_size].float()

        if greedy:
            next_id = next_logits.argmax().item()
        else:
            next_logits = next_logits / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumulative > top_p).float().argmax().item()
            keep = sorted_idx[: cutoff + 1]
            mask = torch.zeros_like(probs)
            mask[keep] = probs[keep]
            mask = mask / mask.sum()
            next_id = torch.multinomial(mask, 1).item()

        if next_id == eot_id:
            break
        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], device=device)  # only the new token; history lives in the cache

    return tok.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer or cfg.tokenizer.path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    real_vocab_size = len(tok)
    eot_id = tok.eos_token_id

    model = Transformer(cfg.model).to(device)
    step = load_weights(model, args.checkpoint)
    model.eval()
    print(f"[ckpt] loaded {args.checkpoint}" + (f" (step {step})" if step is not None else ""))
    print(f"[device] {device}")
    print()
    print("This is a BASE model (no SFT) -- it continues text, it does not answer")
    print("questions or follow instructions. Short declarative prompts work best,")
    print('e.g. "The capital of France is" or "def fibonacci(n):".')
    print()
    print("Commands: /temp <f>  /topp <f>  /tokens <n>  /greedy  /sample  exit")
    print(f"Current: temperature={args.temperature} top_p={args.top_p} max_new_tokens={args.max_new_tokens} mode=sample")
    print()

    temperature, top_p, max_new_tokens, greedy = args.temperature, args.top_p, args.max_new_tokens, False

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = prompt.strip()
        if stripped.lower() in ("exit", "quit"):
            break
        if not stripped:
            continue
        if stripped.startswith("/temp "):
            temperature = float(stripped.split()[1])
            print(f"[set] temperature={temperature}")
            continue
        if stripped.startswith("/topp "):
            top_p = float(stripped.split()[1])
            print(f"[set] top_p={top_p}")
            continue
        if stripped.startswith("/tokens "):
            max_new_tokens = int(stripped.split()[1])
            print(f"[set] max_new_tokens={max_new_tokens}")
            continue
        if stripped == "/greedy":
            greedy = True
            print("[set] mode=greedy")
            continue
        if stripped == "/sample":
            greedy = False
            print("[set] mode=sample")
            continue

        completion = generate(
            model, tok, device, real_vocab_size, eot_id, cfg.model.max_seq_len,
            prompt, max_new_tokens, temperature, top_p, greedy,
        )
        print(f"=> {completion}\n")


if __name__ == "__main__":
    main()
