"""Generate text with BubblSpace MoE-0.9B-A0.2B.

Chat (SFT model):  python generate.py --chat --prompt "What is the capital of India?"
Base-style text:   python generate.py --prompt "The capital of France is"

--chat wraps the prompt in the training chat format, <|im_start|>user\\n{prompt}\\n<|im_end|><|im_start|>assistant\\n
(the same string tokenizer.apply_chat_template(..., add_generation_prompt=True) produces), and stops at <|im_end|>.
--system adds a system turn; for tool calling pass the tools as JSON with --tools tools.json (rendered inside
<tools>…</tools> in the system turn, as in training). Greedy decoding by default.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.config import load_config
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

REPO_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_DIR / "training_config.yaml"))
    parser.add_argument("--weights", default=str(REPO_DIR / "model.safetensors"))
    parser.add_argument("--tokenizer", default=str(REPO_DIR / "tokenizer"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chat", action="store_true", help="wrap the prompt in the chat format (SFT model)")
    parser.add_argument("--system", default=None, help="optional system message (with --chat)")
    parser.add_argument("--tools", default=None, help="JSON file with a list of tool definitions (with --chat)")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    model = Transformer(cfg.model).to(device)
    model.load_state_dict(load_file(args.weights, device=device))
    model.eval()

    if args.chat:
        messages = []
        system = args.system or ""
        if args.tools:
            tools = json.loads(Path(args.tools).read_text())
            system = (system + "\n" if system else "") + f"<tools>\n{json.dumps(tools)}\n</tools>"
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": args.prompt})
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = args.prompt
    stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    ids = tok.encode(text, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    cache = model.new_kv_cache()  # one prefill over the prompt, then one token per step
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                                enabled=(device == "cuda")):
                logits, _, _, _ = model(input_ids, kv_cache=cache)
            next_logits = logits[0, -1, :len(tok)].float()
            if args.temperature <= 0:
                next_id = int(next_logits.argmax())
            else:
                vals, idx = (next_logits / args.temperature).topk(args.top_k)
                next_id = int(idx[torch.multinomial(F.softmax(vals, dim=-1), 1)])
            if next_id in stop_ids:
                break
            generated.append(next_id)
            input_ids = torch.tensor([[next_id]], device=device)
    print(tok.decode(generated) if args.chat else text + tok.decode(generated))


if __name__ == "__main__":
    main()
