import math

import torch.nn as nn


def init_weights(model: nn.Module, std: float, n_layers: int) -> None:
    """std=0.02 everywhere by default; residual_proj_std on the two
    projections that write directly back into the residual stream
    (attn.o_proj and experts.W_down), per the depth-scaled GPT-2-style
    residual init. Router weights are left untouched here — MoEFFN
    initializes its own router with router_init_std at construction time.
    Norm weights (ZeroCenteredRMSNorm.w, 1-D) are skipped and stay at their
    zero init.
    """
    residual_std = std / math.sqrt(2 * n_layers)

    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        if name.endswith("W_down") or name.endswith("o_proj.weight"):
            nn.init.normal_(p, mean=0.0, std=residual_std)
        elif name.endswith("router.weight"):
            continue
        else:
            nn.init.normal_(p, mean=0.0, std=std)
