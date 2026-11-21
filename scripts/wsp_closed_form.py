"""Closed-form, activation-derived weight-space edit (ROME/MEMIT-style), as a
principled replacement for the naive fixed-gain scaling in wsp_experiment.py.

Unlike the first attempt (pick heads from a diagnostic, multiply by an
arbitrary pre-chosen gain), this script derives BOTH the direction and
magnitude of the weight edit from the model's own activations and gradients,
and explicitly protects unrelated behavior via a regularized least-squares
solve -- the same mathematical structure used by areu01or00/qwen0.8B-wsp
(there: mlp.down_proj as a key-value associative memory; here: one
attention layer's o_proj, since our failure mode looks attention-circuit-
shaped, not MLP-fact-shaped).

Method, per edited layer L:
1. POSITIVE keys/values (what should change): for each of the 30
   target-category cases, under both ctxA and ctxB, capture o_proj's output
   z at the last context token position (hook + retain_grad), compute the
   margin loss (favor the correct completion over the incorrect one via a
   fully differentiable log-likelihood scorer), backprop, and read off the
   gradient at z. The target value v* = z - eta * grad (one local gradient
   step, eta chosen as a fraction of ||z|| so the step size is derived from
   the activation's own scale, not an arbitrary constant). R's columns are
   (v* - z); Kp's columns are the corresponding o_proj INPUT at that
   position (forward_pre_hook).
2. NEGATIVE/protected keys (what must NOT change): the same capture (input
   only, no gradient) at the same layer/position for the other 70 probe
   cases (both contexts) -- these get target delta 0.
3. Solve ΔW = R Kp^T (lam * Kn Kn^T + Kp Kp^T + mu*I)^-1, mu set adaptively
   from the Gram matrix's own scale (not a hand-picked constant).
4. Add ΔW to o_proj.weight at layer L, save as a new checkpoint, leave
   evaluation to probe_entity_binding.py.

Usage:
    python scripts/wsp_closed_form.py \
        --checkpoint /path/to/step_19073.pt \
        --cases_file benchmark/data/probefinding-winograndestyle/probefinding-winogrande-<ts>.json \
        --target_categories comparative_value,emotional_attribution,possession_obligation \
        --edit_layer 6 --lam 1.0 \
        --out_checkpoint /path/to/wsp_closedform_step19073_<ts>.pt
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


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def diff_score_continuation(model, real_vocab_size, device, context_enc, continuation_enc, capture_layer, capture_store):
    """Differentiable version of BubblSpaceLM._score_continuation, plus
    captures the input/output of capture_layer's o_proj at the LAST CONTEXT
    token position (the position whose hidden state predicts the first
    continuation token -- our edit target)."""
    full_enc = context_enc + continuation_enc
    inp = torch.tensor([full_enc[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(full_enc[1:], dtype=torch.long, device=device)
    pos = len(context_enc) - 1  # index into `inp`/outputs of the last context token

    captured = {}

    def pre_hook(module, args):
        captured["input"] = args[0]  # (1, seq_len, d_model) -- do NOT detach, need graph

    def fwd_hook(module, args, output):
        output.retain_grad()
        captured["output"] = output

    h1 = model.blocks[capture_layer].attn.o_proj.register_forward_pre_hook(pre_hook)
    h2 = model.blocks[capture_layer].attn.o_proj.register_forward_hook(fwd_hook)
    logits, _, _, _ = model(inp)
    h1.remove()
    h2.remove()

    logits = logits[0, :, :real_vocab_size].float()
    cont_len = len(continuation_enc)
    cont_logits = logits[-cont_len:]
    cont_targets = targets[-cont_len:]
    log_probs = F.log_softmax(cont_logits, dim=-1)
    token_logprobs = log_probs.gather(1, cont_targets.unsqueeze(-1)).squeeze(-1)
    total_logprob = token_logprobs.sum()

    capture_store["input_at_pos"] = captured["input"][0, pos, :]  # keep graph-free copy taken by caller
    capture_store["output_tensor"] = captured["output"]
    capture_store["pos"] = pos
    return total_logprob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--cases_file", required=True)
    parser.add_argument("--target_categories", default="comparative_value,emotional_attribution,possession_obligation")
    parser.add_argument("--edit_layer", type=int, default=6)
    parser.add_argument("--lam", type=float, default=1.0, help="weight on protecting negative/unrelated keys")
    parser.add_argument("--eta_frac", type=float, default=0.3, help="gradient step size as a fraction of ||z||")
    parser.add_argument("--out_checkpoint", required=True)
    parser.add_argument("--diagnostic_output", default=None)
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
    model.eval()  # eval mode (no dropout etc) but we still need gradients -- do NOT wrap in no_grad
    step = ckpt.get("step")
    print(f"[ckpt] loaded {args.checkpoint} (step {step})")

    with open(args.cases_file) as f:
        data = json.load(f)
    target_cats = set(c.strip() for c in args.target_categories.split(","))
    target_cases = [c for c in data["cases"] if c.get("category") in target_cats]
    other_cases = [c for c in data["cases"] if c.get("category") not in target_cats]
    print(f"[probe] {len(target_cases)} target cases, {len(other_cases)} protected (other) cases")

    L = args.edit_layer

    def get_target_kv(ctx_text, correct_c, wrong_c):
        ctx_enc = encode(tok, ctx_text)
        if len(ctx_enc) == 0:
            ctx_enc = [eot_id]
        correct_enc = encode(tok, correct_c)
        wrong_enc = encode(tok, wrong_c)

        model.zero_grad(set_to_none=True)
        store_c = {}
        logp_correct = diff_score_continuation(model, real_vocab_size, device, ctx_enc, correct_enc, L, store_c)
        z_input = store_c["input_at_pos"].detach().clone()  # key: o_proj input, same regardless of which completion follows
        loss = -logp_correct
        loss.backward()
        grad_correct = store_c["output_tensor"].grad[0, store_c["pos"], :].detach().clone()

        model.zero_grad(set_to_none=True)
        store_w = {}
        logp_wrong = diff_score_continuation(model, real_vocab_size, device, ctx_enc, wrong_enc, L, store_w)
        loss2 = logp_wrong
        loss2.backward()
        grad_wrong = store_w["output_tensor"].grad[0, store_w["pos"], :].detach().clone()

        total_grad = grad_correct + grad_wrong  # d(-logp_correct + logp_wrong)/dz
        z_output = store_c["output_tensor"][0, store_c["pos"], :].detach().clone()

        step_dir = -total_grad
        step_dir = step_dir / (step_dir.norm() + 1e-8)
        eta = args.eta_frac * z_output.norm().item()
        v_star = z_output + eta * step_dir
        return z_input, (v_star - z_output)

    def get_protected_key(ctx_text):
        ctx_enc = encode(tok, ctx_text)
        if len(ctx_enc) == 0:
            ctx_enc = [eot_id]
        inp = torch.tensor([ctx_enc], dtype=torch.long, device=device)
        captured = {}

        def pre_hook(module, inp_args):
            captured["input"] = inp_args[0][0, -1, :].detach().clone()

        h = model.blocks[L].attn.o_proj.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            model(inp)
        h.remove()
        return captured["input"]

    print(f"[edit] building positive (key, value-delta) pairs at layer {L}...")
    Kp_list, R_list = [], []
    for case in target_cases:
        for ctx, correct, wrong in [(case["ctxA"], case["c1"], case["c2"]), (case["ctxB"], case["c2"], case["c1"])]:
            k, r = get_target_kv(ctx, correct, wrong)
            Kp_list.append(k)
            R_list.append(r)
    Kp = torch.stack(Kp_list, dim=1)  # (d_model, n_pos)
    R = torch.stack(R_list, dim=1)  # (d_model, n_pos)
    print(f"[edit] {Kp.shape[1]} positive keys collected")

    print(f"[edit] building negative/protected keys at layer {L}...")
    Kn_list = []
    for case in other_cases:
        Kn_list.append(get_protected_key(case["ctxA"]))
        Kn_list.append(get_protected_key(case["ctxB"]))
    Kn = torch.stack(Kn_list, dim=1)  # (d_model, n_neg)
    print(f"[edit] {Kn.shape[1]} protected keys collected")

    Kp64, R64, Kn64 = Kp.double(), R.double(), Kn.double()
    gram = Kp64 @ Kp64.T + args.lam * (Kn64 @ Kn64.T)
    mu = 1e-2 * (gram.diagonal().mean().item())
    d = gram.shape[0]
    gram = gram + mu * torch.eye(d, dtype=torch.float64, device=gram.device)
    delta_w = (R64 @ Kp64.T) @ torch.linalg.inv(gram)
    delta_w = delta_w.float()
    print(f"[edit] solved delta_w, shape={tuple(delta_w.shape)}, norm={delta_w.norm().item():.4f}, mu={mu:.6g}")

    orig_norm = model.blocks[L].attn.o_proj.weight.norm().item()
    with torch.no_grad():
        model.blocks[L].attn.o_proj.weight += delta_w
    new_norm = model.blocks[L].attn.o_proj.weight.norm().item()
    print(f"[edit] o_proj.weight norm {orig_norm:.4f} -> {new_norm:.4f} (delta_w norm {delta_w.norm().item():.4f})")

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "wsp_edit": {
                "method": "closed_form_ridge",
                "base_checkpoint": args.checkpoint,
                "edit_layer": L,
                "target_categories": sorted(target_cats),
                "n_positive_keys": Kp.shape[1],
                "n_protected_keys": Kn.shape[1],
                "lam": args.lam,
                "mu": mu,
                "eta_frac": args.eta_frac,
                "delta_w_norm": delta_w.norm().item(),
            },
        },
        out_path,
    )
    print(f"[wsp] saved edited checkpoint to {out_path}")

    if args.diagnostic_output:
        out = Path(args.diagnostic_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(
                {
                    "edit_layer": L,
                    "n_positive_keys": Kp.shape[1],
                    "n_protected_keys": Kn.shape[1],
                    "lam": args.lam,
                    "mu": mu,
                    "delta_w_norm": delta_w.norm().item(),
                    "orig_o_proj_norm": orig_norm,
                    "new_o_proj_norm": new_norm,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
