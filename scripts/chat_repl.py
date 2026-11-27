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
    /rep <float>       repetition penalty on already-seen tokens (1.0 = off, default 1.15)
    /ngram <int>       never repeat an n-gram of this size (0 = off, default 4)
    /minp <float>      min-p filter: drop tokens below minp * top token prob (0 = off)
    /chat              toggle chat mode: wrap input in <|im_start|>user ... <|im_start|>assistant, stop at <|im_end|>
    /file <path>       send the contents of a file as the prompt (long prompts, multi-line text)
    /system <text>     (chat) set a system message; "/system off" clears it
    /tools <path>      (chat) load a JSON list of tool definitions into the system turn (<tools>...</tools>); "/tools off"
    /tool <json>       (chat) send a tool result back (a user turn wrapped in <tool_response>), continuing the conversation
    /multi             (chat) toggle multi-turn: keep the conversation history (on automatically after /tools)
    /reset             (chat) forget the conversation history
    /settings          show current settings
    exit / quit         stop

Decoding controls only change generation, never benchmark scores (those are
log-likelihood based and use no sampling).
"""

import argparse
import json
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


def _banned_by_ngram(seq, n):
    """Tokens that would complete an n-gram already present in seq."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    return banned


@torch.no_grad()
def generate(model, tok, device, real_vocab_size, stop_ids, max_seq_len, prompt, max_new_tokens, temperature, top_p, greedy,
             rep_penalty=1.0, no_repeat_ngram=0, min_p=0.0):
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

        context = ids + generated
        if rep_penalty != 1.0 and context:
            seen = torch.tensor(sorted(set(context)), device=device)
            vals = next_logits[seen]
            next_logits[seen] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
        banned = _banned_by_ngram(context, no_repeat_ngram)
        if banned:
            next_logits[torch.tensor(sorted(banned), device=device)] = float("-inf")

        if greedy:
            next_id = next_logits.argmax().item()
        else:
            next_logits = next_logits / max(temperature, 1e-5)
            probs = F.softmax(next_logits, dim=-1)
            if min_p > 0:
                probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
                probs = probs / probs.sum()
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumulative > top_p).float().argmax().item()
            keep = sorted_idx[: cutoff + 1]
            mask = torch.zeros_like(probs)
            mask[keep] = probs[keep]
            mask = mask / mask.sum()
            next_id = torch.multinomial(mask, 1).item()

        if next_id in stop_ids:
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
    parser.add_argument("--rep_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram", type=int, default=4)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--chat", action="store_true", help="start in chat mode")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer or cfg.tokenizer.path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    real_vocab_size = len(tok)
    eot_id = tok.eos_token_id
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")

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
    print("Commands: /temp <f>  /topp <f>  /tokens <n>  /greedy  /sample  /rep <f>  /ngram <n>  /minp <f>  /chat  /settings  exit")
    print("Chat:     /file <path>  /system <text|off>  /tools <path|off>  /tool <json>  /multi  /reset")

    temperature, top_p, max_new_tokens, greedy = args.temperature, args.top_p, args.max_new_tokens, False
    rep_penalty, no_repeat_ngram, min_p, chat = args.rep_penalty, args.no_repeat_ngram, args.min_p, args.chat
    system, tools, history, multi = None, None, [], False  # chat: history = [(role, text), ...] kept when multi is on

    def show():
        print(f"Current: mode={'greedy' if greedy else 'sample'} temperature={temperature} top_p={top_p} min_p={min_p} "
              f"rep_penalty={rep_penalty} no_repeat_ngram={no_repeat_ngram} max_new_tokens={max_new_tokens} chat={chat}")
    show()
    print()

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
        if stripped.startswith("/rep "):
            rep_penalty = float(stripped.split()[1])
            print(f"[set] rep_penalty={rep_penalty}")
            continue
        if stripped.startswith("/ngram "):
            no_repeat_ngram = int(stripped.split()[1])
            print(f"[set] no_repeat_ngram={no_repeat_ngram}")
            continue
        if stripped.startswith("/minp "):
            min_p = float(stripped.split()[1])
            print(f"[set] min_p={min_p}")
            continue
        if stripped == "/chat":
            chat = not chat
            print(f"[set] chat={chat}")
            continue
        if stripped == "/settings":
            show()
            print(f"system={'set' if system else 'off'} tools={len(tools) if tools else 0} multi-turn={multi} history_turns={len(history)}")
            continue
        if stripped == "/multi":
            multi, history = not multi, []
            print(f"[set] multi-turn={multi} (conversation cleared)")
            continue
        if stripped == "/reset":
            history = []
            print("[set] conversation cleared")
            continue
        if stripped.startswith("/system "):
            arg = stripped[len("/system "):].strip()
            system = None if arg == "off" else arg
            history = []
            print(f"[set] system={'off' if system is None else repr(system[:60])} (conversation cleared)")
            continue
        if stripped.startswith("/tools "):
            arg = stripped[len("/tools "):].strip()
            try:
                tools = None if arg == "off" else json.loads(Path(arg).read_text())
            except Exception as e:
                print(f"[error] {e}")
                continue
            history, multi = [], multi or tools is not None
            print(f"[set] tools={'off' if tools is None else [t.get('function', t).get('name') for t in tools]} multi-turn={multi} (conversation cleared)")
            continue
        role = "user"
        if stripped.startswith("/file "):
            try:
                prompt = Path(stripped[len("/file "):].strip()).read_text().strip()
            except Exception as e:
                print(f"[error] {e}")
                continue
            print(f"[file] {len(prompt):,} chars, {len(tok.encode(prompt, add_special_tokens=False)):,} tokens")
        elif stripped.startswith("/tool "):
            prompt = f"<tool_response>\n{stripped[len('/tool '):].strip()}\n</tool_response>"

        # same wire format as pretraining (prepare_agentic.py) and SFT (phase3_build_sft.py): "\n" before <|im_end|>, no gap between turns
        if chat:
            sys_text = "\n".join(x for x in (system, f"<tools>\n{json.dumps(tools)}\n</tools>" if tools else None) if x)
            turns = ([("system", sys_text)] if sys_text else []) + history + [(role, prompt)]
            text = "".join(f"<|im_start|>{r}\n{c}\n<|im_end|>" for r, c in turns) + "<|im_start|>assistant\n"
        else:
            text = prompt
        completion = generate(
            model, tok, device, real_vocab_size, {eot_id, im_end_id}, cfg.model.max_seq_len,
            text, max_new_tokens, temperature, top_p, greedy, rep_penalty, no_repeat_ngram, min_p,
        )
        print(f"=> {completion}\n")
        if chat and multi:
            history += [(role, prompt), ("assistant", completion.strip())]


if __name__ == "__main__":
    main()
