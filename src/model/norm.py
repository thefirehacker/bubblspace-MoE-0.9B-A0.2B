import torch
import torch.nn as nn


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with effective weight (1 + w), w initialized to zero.

    Weight-decaying w pulls it toward 0, i.e. toward the identity transform
    (effective weight 1), rather than toward zero gain as decaying a
    conventional RMSNorm weight directly would. w is therefore intentionally
    included in the weight-decayed AdamW group, unlike the usual
    no-decay-for-norms convention.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * (1.0 + self.w.float())).to(dtype)
