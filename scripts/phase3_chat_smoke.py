"""Chat smoke test for an SFT checkpoint: a fixed set of conversations, rendered in the SFT wire format, greedy decoding.

Wire format (same as scripts/phase3_build_sft.py): <|im_start|>{role}\n{text}\n<|im_end|>, turns concatenated, then
<|im_start|>assistant\n. Tools go in the system turn inside <tools>…</tools>; tool results come back as a user turn wrapped
in <tool_response>…</tool_response>. Raw greedy (no repetition penalty), so habits show as they are.

    python scripts/phase3_chat_smoke.py --config configs/phase25b.yaml --checkpoint /path/to/phase3/step_1500.pt \
        --out benchmark/phase3/chat_smoke_phase3_step1500.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer

from chat_repl import generate, load_weights
from src.config import load_config
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

WEATHER = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "dict", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "description": "celsius or fahrenheit"}}, "required": ["city"]}}}]
TOOLS2 = WEATHER + [{"type": "function", "function": {
    "name": "convert_currency", "description": "Convert an amount from one currency to another.",
    "parameters": {"type": "dict", "properties": {
        "amount": {"type": "float", "description": "Amount to convert"},
        "from_currency": {"type": "string", "description": "3-letter code, e.g. USD"},
        "to_currency": {"type": "string", "description": "3-letter code, e.g. INR"}},
        "required": ["amount", "from_currency", "to_currency"]}}}]


def tools_sys(tools):
    return ("system", f"<tools>\n{json.dumps(tools)}\n</tools>")


# (category, name, turns)
CASES = [
    ("identity", "who are you", [("user", "Who are you?")]),
    ("facts", "capital of India", [("user", "What is the capital of India?")]),
    ("facts", "reverse capital", [("user", "New Delhi is the capital of which country?")]),
    ("facts", "photosynthesis", [("user", "Explain photosynthesis in two sentences.")]),
    ("facts", "largest planet", [("user", "Which is the largest planet in the solar system?")]),
    ("instruction", "three fruits", [("user", "List three fruits, one per line, nothing else.")]),
    ("instruction", "one word", [("user", "Answer with one word only: what colour is the sky on a clear day?")]),
    ("instruction", "rewrite formal", [("user", "Rewrite this more politely: 'Send me the report now.'")]),
    ("binding", "cap/room", [("user", "Ravi gave Anita a red cap, and Anita gave Ravi a blue book. Who has the red cap now?")]),
    ("binding", "passive", [("user", "The letter was written by Maya and read by Tom. Who wrote the letter?")]),
    ("binding", "relational", [("user", "Sara is taller than Ben. Ben is taller than Leo. Who is the shortest?")]),
    ("math", "add", [("user", "What is 47 + 38?")]),
    ("math", "word problem", [("user", "A shop sells pens at 12 rupees each. How much do 7 pens cost? Show your steps.")]),
    ("math", "gsm-style", [("user", "Tom has 3 boxes with 8 apples in each box. He eats 5 apples. How many apples are left?")]),
    ("code", "reverse string", [("user", "Write a Python function that reverses a string.")]),
    ("code", "pytorch linear", [("user", "Write a minimal PyTorch nn.Module with one linear layer and a forward method.")]),
    ("code", "fizzbuzz", [("user", "Write FizzBuzz in Python for numbers 1 to 15.")]),
    ("tool", "weather call", [tools_sys(WEATHER), ("user", "What's the weather in Paris right now?")]),
    ("tool", "pick right tool", [tools_sys(TOOLS2), ("user", "How much is 100 US dollars in Indian rupees?")]),
    ("tool", "no tool needed", [tools_sys(TOOLS2), ("user", "Hi! How are you today?")]),
    ("tool", "parallel", [tools_sys(WEATHER), ("user", "What's the weather in Tokyo and in London?")]),
    ("tool", "use tool result", [tools_sys(WEATHER), ("user", "What's the weather in Paris right now?"),
                                 ("assistant", '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'),
                                 ("user", '<tool_response>\n{"city": "Paris", "temp_c": 14, "condition": "light rain"}\n</tool_response>')]),
    ("multiturn", "follow-up", [("user", "My name is Arjun and I like cricket."),
                                ("assistant", "Nice to meet you, Arjun! Cricket is a great sport."),
                                ("user", "What is my name and what do I like?")]),
]


def render(turns):
    return "".join(f"<|im_start|>{r}\n{t}\n<|im_end|>" for r, t in turns) + "<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase25b.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    model = Transformer(cfg.model).to(device)
    step = load_weights(model, a.checkpoint)
    model.eval()

    lines = [f"# Chat smoke test: `{a.checkpoint}` (step {step})", "",
             "Greedy, no repetition penalty, max 200 new tokens, stop at `<|im_end|>`.", ""]
    for cat, name, turns in CASES:
        t0 = time.time()
        out = generate(model, tok, device, len(tok), stop, cfg.model.max_seq_len, render(turns),
                       a.max_new_tokens, 1.0, 1.0, True)
        last_user = [t for r, t in turns if r == "user"][-1]
        block = f"## [{cat}] {name}\n\n**user:** {last_user}\n\n**assistant:**\n```\n{out.strip()}\n```\n"
        print(block, flush=True)
        print(f"({time.time() - t0:.1f}s)\n", flush=True)
        lines.append(block)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
