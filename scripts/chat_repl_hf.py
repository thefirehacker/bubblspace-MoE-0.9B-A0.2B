"""Interactive REPL for any Hugging Face causal LM checkpoint (e.g. OLMo-2 1B at a given training step), with the
same commands and the same decoding as scripts/chat_repl.py, so side-by-side tests against our checkpoints are fair.

Each prompt is independent (no chat history). Weights come through scripts/binding_levels_hf.fetch: safetensors
only when the repo has them, pickle .bin only after a ClamAV scan (project rule).

    python scripts/chat_repl_hf.py --model allenai/OLMo-2-0425-1B --revision stage1-step10000-tokens21B \
        --rep_penalty 1.0 --no_repeat_ngram 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from binding_levels_hf import fetch  # noqa: E402
from chat_repl import _banned_by_ngram  # noqa: E402


@torch.no_grad()
def generate(model, tok, device, prompt, max_new_tokens, temperature, top_p, greedy, rep_penalty, no_repeat_ngram, min_p):
    ids = tok(prompt, add_special_tokens=True)["input_ids"]  # BOS where the model expects one (as binding_levels_hf)
    input_ids = torch.tensor([ids], device=device)
    past, generated, vocab = None, [], len(tok)
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :vocab].float()
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
            probs = F.softmax(next_logits / max(temperature, 1e-5), dim=-1)
            if min_p > 0:
                probs = torch.where(probs >= min_p * probs.max(), probs, torch.zeros_like(probs))
                probs = probs / probs.sum()
            sp, si = torch.sort(probs, descending=True)
            keep = si[: (torch.cumsum(sp, -1) > top_p).float().argmax().item() + 1]
            mask = torch.zeros_like(probs)
            mask[keep] = probs[keep]
            next_id = torch.multinomial(mask / mask.sum(), 1).item()
        if next_id == tok.eos_token_id:
            break
        generated.append(next_id)
        input_ids = torch.tensor([[next_id]], device=device)
    return tok.decode(generated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--cache_dir", default="/lambda/nfs/us-west-1-eval-inference/hf_cache")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram", type=int, default=4)
    ap.add_argument("--min_p", type=float, default=0.0)
    a = ap.parse_args()

    path, fmt = fetch(a.model, a.revision, os.path.expanduser(a.cache_dir))
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except (ValueError, OSError):
        tok = AutoTokenizer.from_pretrained(a.model)  # some training branches ship no tokenizer files
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    print(f"[ckpt] loaded {a.model}@{a.revision or 'main'} ({fmt}, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")
    print(f"[device] {device}\n\nBASE model: it continues text. Each prompt is independent (no history).")
    print("Commands: /temp <f>  /topp <f>  /tokens <n>  /greedy  /sample  /rep <f>  /ngram <n>  /minp <f>  /settings  exit")

    s = dict(temperature=a.temperature, top_p=a.top_p, max_new_tokens=a.max_new_tokens, greedy=False,
             rep_penalty=a.rep_penalty, no_repeat_ngram=a.no_repeat_ngram, min_p=a.min_p)
    show = lambda: print("Current: mode={} temperature={temperature} top_p={top_p} min_p={min_p} rep_penalty={rep_penalty} "
                         "no_repeat_ngram={no_repeat_ngram} max_new_tokens={max_new_tokens}".format(
                             "greedy" if s["greedy"] else "sample", **s))
    show()
    setters = {"/temp": ("temperature", float), "/topp": ("top_p", float), "/tokens": ("max_new_tokens", int),
               "/rep": ("rep_penalty", float), "/ngram": ("no_repeat_ngram", int), "/minp": ("min_p", float)}
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        st = prompt.strip()
        if st.lower() in ("exit", "quit"):
            break
        if not st:
            continue
        cmd = st.split()[0]
        if cmd in setters and len(st.split()) == 2:
            k, cast = setters[cmd]
            s[k] = cast(st.split()[1])
            print(f"[set] {k}={s[k]}")
            continue
        if st in ("/greedy", "/sample"):
            s["greedy"] = st == "/greedy"
            print(f"[set] mode={'greedy' if s['greedy'] else 'sample'}")
            continue
        if st == "/settings":
            show()
            continue
        print(f"=> {generate(model, tok, device, prompt, s['max_new_tokens'], s['temperature'], s['top_p'], s['greedy'], s['rep_penalty'], s['no_repeat_ngram'], s['min_p'])}\n")


if __name__ == "__main__":
    main()
