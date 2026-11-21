import torch
import torch.nn as nn

from .attention import Attention
from .moe import ExpertLoadTracker, MoEFFN
from .norm import ZeroCenteredRMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, attn_cfg, moe_cfg, norm_eps: float):
        super().__init__()
        self.norm1 = ZeroCenteredRMSNorm(d_model, eps=norm_eps)
        self.attn = Attention(
            d_model=d_model,
            n_heads=attn_cfg.n_heads,
            head_dim=attn_cfg.head_dim,
            n_kv_heads=attn_cfg.n_kv_heads,
            norm_eps=norm_eps,
            output_gate=attn_cfg.output_gate,
            bias=attn_cfg.bias,
        )
        self.norm2 = ZeroCenteredRMSNorm(d_model, eps=norm_eps)
        self.moe = MoEFFN(
            d_model=d_model,
            n_experts=moe_cfg.n_experts,
            top_k=moe_cfg.top_k,
            expert_hidden=moe_cfg.expert_hidden,
            backend=moe_cfg.backend,
            router_init_std=moe_cfg.router_init_std,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        load_tracker: ExpertLoadTracker | None,
        layer_idx: int,
        block_mask=None,
    ):
        x = x + self.attn(self.norm1(x), cos, sin, block_mask)
        moe_out, lb_loss, z_loss, hard_counts = self.moe(self.norm2(x), load_tracker, layer_idx)
        x = x + moe_out
        return x, lb_loss, z_loss, hard_counts
