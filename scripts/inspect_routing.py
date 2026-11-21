"""Inspect which MoE experts fire, per layer, per token -- and (in rollout
mode) whether the routing pattern itself locks into a repeating loop when
the generated text does.

This is NOT sparse-autoencoder/feature-level interpretability (Goodfire-style
circuit tracing) -- it's much simpler: our router already computes an exact,
known probability over all 64 experts per token per layer, so "which expert
fired" is directly readable via a forward hook, no extra training required.

Modes:
  --mode routing   one forward pass over --prompt, print top-k experts +
                   weights at the LAST token position, for every layer.
  --mode rollout   greedy-generate --max_new_tokens tokens one at a time,
                   printing each token alongside a compact per-layer
                   top-1-expert signature, so you can see directly whether
                   the routing pattern repeats in lockstep with repeated text.

Usage:
    python scripts/inspect_routing.py --checkpoint /path/to/ckpt.pt \
        --mode rollout --prompt "The marathon took longer to finish than the sprint because the sprint was" \
        --max_new_tokens 40
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["routing", "rollout"], default="routing")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--top_experts_shown", type=int, default=8)
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
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[ckpt] loaded {args.checkpoint} (step {ckpt.get('step')})")

    n_layers = cfg.model.n_layers
    top_k = cfg.model.moe.top_k

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, output):
            captured[layer_idx] = output.detach()  # (n_tokens, n_experts) router logits
        return hook

    for i, block in enumerate(model.blocks):
        block.moe.router.register_forward_hook(make_hook(i))

    @torch.no_grad()
    def route_last_position(ids: list[int]):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        model(inp)
        per_layer = []
        for i in range(n_layers):
            logits = captured[i][-1]  # last token position, (n_experts,)
            probs = F.softmax(logits.float(), dim=-1)
            topk_probs, topk_idx = probs.topk(top_k)
            per_layer.append((topk_idx.tolist(), topk_probs.tolist()))
        return per_layer

    ids = tok.encode(args.prompt, add_special_tokens=False)

    if args.mode == "routing":
        per_layer = route_last_position(ids)
        print(f"\nPrompt: {args.prompt!r}")
        print(f"Routing at the last token position (top-{top_k} of {cfg.model.moe.n_experts} experts per layer):\n")
        for i, (idx, probs) in enumerate(per_layer):
            experts_str = ", ".join(f"e{e}({p:.2f})" for e, p in zip(idx, probs))
            print(f"  layer {i:>2}: {experts_str}")
        return

    # rollout mode
    print(f"\nPrompt: {args.prompt!r}")
    print("Greedy rollout -- token, then each layer's TOP-1 expert id (compact signature):\n")
    cur_ids = list(ids)
    seen_signatures = {}
    for step in range(args.max_new_tokens):
        per_layer = route_last_position(cur_ids)
        signature = tuple(idx[0] for idx, _ in per_layer)  # top-1 expert per layer

        with torch.no_grad():
            inp = torch.tensor([cur_ids], dtype=torch.long, device=device)
            logits, _, _, _ = model(inp)
        next_id = logits[0, -1, :real_vocab_size].argmax().item()
        if next_id == eot_id:
            print("  <eot>")
            break
        token_text = tok.decode([next_id])
        cur_ids.append(next_id)

        repeat_note = ""
        if signature in seen_signatures:
            repeat_note = f"  <-- IDENTICAL routing signature to step {seen_signatures[signature]}"
        seen_signatures[signature] = step

        sig_str = ",".join(str(e) for e in signature)
        print(f"  step {step:>2}  token={token_text!r:<20} top1-experts=[{sig_str}]{repeat_note}")


if __name__ == "__main__":
    main()
