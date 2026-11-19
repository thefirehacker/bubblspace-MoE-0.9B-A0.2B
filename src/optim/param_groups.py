import torch
import torch.nn as nn

from .muon import Muon

# 2D (or 3D stacked-expert) hidden matrices: orthogonalized by Muon.
_MUON_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",  # attention output-gate projection
    "W_gate",
    "W_up",
    "W_down",
)

# Tied embedding/lm_head, routers, and all norm weights (including the
# zero-centered norm's "w", which IS weight-decayed here unlike the usual
# no-decay-for-norms convention — see ZeroCenteredRMSNorm docstring).
_ADAMW_SUFFIXES = (
    "embed.weight",
    "router.weight",
)


def split_param_groups(model: nn.Module):
    muon_params = []
    adamw_params = []
    unclassified = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(_MUON_SUFFIXES):
            muon_params.append(p)
        elif name.endswith(_ADAMW_SUFFIXES) or name.endswith(".w"):
            adamw_params.append(p)
        else:
            unclassified.append(name)

    if unclassified:
        raise ValueError(f"Unclassified parameters (add a rule in param_groups.py): {unclassified}")

    n_total = sum(1 for _, p in model.named_parameters() if p.requires_grad)
    assert len(muon_params) + len(adamw_params) == n_total, "parameter accounted for more than once"

    return muon_params, adamw_params


def build_optimizers(model: nn.Module, muon_cfg, adamw_cfg):
    muon_params, adamw_params = split_param_groups(model)

    muon_opt = Muon(
        muon_params,
        lr=muon_cfg.lr,
        momentum=muon_cfg.momentum,
        nesterov=muon_cfg.nesterov,
        ns_steps=muon_cfg.ns_steps,
        weight_decay=muon_cfg.weight_decay,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=adamw_cfg.lr,
        betas=tuple(adamw_cfg.betas),
        eps=adamw_cfg.eps,
        weight_decay=adamw_cfg.weight_decay,
    )
    return muon_opt, adamw_opt
