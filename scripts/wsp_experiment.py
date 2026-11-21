"""Weight-space programming experiment: amplify the attention heads whose
output already carries the most entity-conditional signal, directly in the
checkpoint's weights, with zero gradient-based training.

Motivation (see benchmark/data/probefinding-winograndestyle/results/):
step_19073 shows a real but weak entity-attribute binding signal in some
categories (spatial_height 70%) and a flat, confound-free zero in others
(comparative_value, emotional_attribution, possession_obligation). The
working hypothesis from the probe analysis is that a conditional (per-input)
binding signal exists somewhere in the network but is too weak, relative to
unconditional lexical/frequency shortcuts, to win at the output. Unlike a
constant steering vector (which shifts every input the same way and so
structurally cannot help a model tell two inputs apart), amplifying a head
whose own output already *differs* between the two framings of a minimal
pair directly increases the magnitude of a signal that is already
conditional -- a much more targeted edit.

Method:
1. For every case in the three 0%-scoring categories (comparative_value,
   emotional_attribution, possession_obligation -- 30 cases total), run
   ctxA and ctxB through the model and capture, via a forward hook on each
   layer's o_proj, the *pre-o_proj* per-head output vector at the final
   token position (i.e. each head's own contribution before it gets mixed
   into the residual stream).
2. For each (layer, head), score how much that head's output differs
   between ctxA and ctxB, normalized by the head's typical output norm
   (so a head that already varies a lot for unrelated reasons doesn't look
   artificially "selective"). This is a purely observational diagnostic --
   no weights are touched yet.
3. Take the top-K most entity-sensitive heads and scale up their
   contribution to the residual stream by directly multiplying the
   corresponding input-column slice of o_proj.weight (and o_proj.bias if
   present) by a gain factor. This is the actual weight-space edit: nothing
   else in the checkpoint changes.
4. Save the edited weights as a new checkpoint (never overwrites the
   original) and leave evaluation to scripts/probe_entity_binding.py
   against both checkpoints for a direct before/after comparison.

Usage:
    python scripts/wsp_experiment.py \
        --checkpoint /path/to/step_19073.pt \
        --cases_file benchmark/data/probefinding-winograndestyle/probefinding-winogrande-<ts>.json \
        --target_categories comparative_value,emotional_attribution,possession_obligation \
        --top_k_heads 16 --gain 2.0 \
        --out_checkpoint /path/to/wsp_step19073_<ts>.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--cases_file", required=True)
    parser.add_argument("--target_categories", default="comparative_value,emotional_attribution,possession_obligation")
    parser.add_argument("--top_k_heads", type=int, default=16)
    parser.add_argument("--gain", type=float, default=2.0)
    parser.add_argument("--out_checkpoint", required=True)
    parser.add_argument("--diagnostic_output", default=None, help="optional: save the per-head sensitivity scores")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer or cfg.tokenizer.path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    model = Transformer(cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step")
    print(f"[ckpt] loaded {args.checkpoint} (step {step})")

    n_layers = cfg.model.n_layers
    n_heads = cfg.model.attention.n_heads
    head_dim = cfg.model.attention.head_dim

    with open(args.cases_file) as f:
        data = json.load(f)
    target_cats = set(c.strip() for c in args.target_categories.split(","))
    target_cases = [c for c in data["cases"] if c.get("category") in target_cats]
    print(f"[probe] {len(target_cases)} cases from categories: {sorted(target_cats)}")

    # --- Step 1-2: capture pre-o_proj per-head activations at the last token, diagnose sensitivity ---
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp):
            # inp[0]: (bsz=1, seq_len, n_heads*head_dim) -- the o_proj input
            captured[layer_idx] = inp[0][0, -1, :].detach().view(n_heads, head_dim).clone()
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.o_proj.register_forward_pre_hook(make_hook(i)))

    @torch.no_grad()
    def run_capture(text: str) -> torch.Tensor:
        ids = encode(tok, text)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        model(inp)
        # stack into (n_layers, n_heads, head_dim)
        return torch.stack([captured[i] for i in range(n_layers)], dim=0)

    diff_sum = torch.zeros(n_layers, n_heads)
    norm_sum = torch.zeros(n_layers, n_heads)
    n_cases = 0
    for case in target_cases:
        actA = run_capture(case["ctxA"]).float().cpu()
        actB = run_capture(case["ctxB"]).float().cpu()
        diff = (actA - actB).norm(dim=-1)  # (n_layers, n_heads)
        typical = 0.5 * (actA.norm(dim=-1) + actB.norm(dim=-1)) + 1e-6
        diff_sum += diff / typical
        norm_sum += 1
        n_cases += 1

    for h in hooks:
        h.remove()

    sensitivity = (diff_sum / norm_sum).numpy()  # (n_layers, n_heads), relative sensitivity
    print(f"[diag] computed sensitivity over {n_cases} cases, shape {sensitivity.shape}")

    flat = [(sensitivity[l, h], l, h) for l in range(n_layers) for h in range(n_heads)]
    flat.sort(key=lambda t: -t[0])
    top = flat[: args.top_k_heads]
    print(f"\nTop {args.top_k_heads} most entity-sensitive heads (relative diff norm):")
    for score, l, h in top:
        print(f"  layer {l:>2} head {h:>2}  sensitivity={score:.3f}")

    if args.diagnostic_output:
        out = Path(args.diagnostic_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "n_cases": n_cases,
                    "target_categories": sorted(target_cats),
                    "sensitivity": sensitivity.tolist(),
                    "top_heads": [{"layer": l, "head": h, "sensitivity": float(s)} for s, l, h in top],
                },
                f,
                indent=2,
            )
        print(f"[diag] saved to {out}")

    # --- Step 3: edit o_proj weights for the selected heads ---
    edited_layers = set()
    with torch.no_grad():
        for score, l, h in top:
            o_proj = model.blocks[l].attn.o_proj
            col_start, col_end = h * head_dim, (h + 1) * head_dim
            o_proj.weight[:, col_start:col_end] *= args.gain
            edited_layers.add(l)
    print(f"\n[edit] scaled o_proj input-columns for {len(top)} heads across {len(edited_layers)} layers by gain={args.gain}")

    # --- Step 4: save new checkpoint ---
    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "wsp_edit": {
                "base_checkpoint": args.checkpoint,
                "target_categories": sorted(target_cats),
                "n_cases": n_cases,
                "top_k_heads": args.top_k_heads,
                "gain": args.gain,
                "edited_heads": [{"layer": l, "head": h, "sensitivity": float(s)} for s, l, h in top],
            },
        },
        out_path,
    )
    print(f"[wsp] saved edited checkpoint to {out_path}")


if __name__ == "__main__":
    main()
