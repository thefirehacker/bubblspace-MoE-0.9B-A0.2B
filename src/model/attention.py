import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import ZeroCenteredRMSNorm
from .rope import apply_rope


class Attention(nn.Module):
    """GQA attention with per-head QK-norm (applied before RoPE), RoPE, and an
    elementwise output gate: Y = sdpa_out * sigmoid(X @ W_g), applied before o_proj.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        norm_eps: float,
        output_gate: bool,
        bias: bool,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.output_gate = output_gate

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=bias)

        self.q_norm = ZeroCenteredRMSNorm(head_dim, eps=norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(head_dim, eps=norm_eps)

        if output_gate:
            self.gate_proj = nn.Linear(d_model, n_heads * head_dim, bias=bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q.transpose(1, 2), cos, sin)  # (bsz, n_heads, seq_len, head_dim)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(self.n_heads != self.n_kv_heads)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)

        if self.output_gate:
            gate = torch.sigmoid(self.gate_proj(x))
            attn_out = attn_out * gate

        return self.o_proj(attn_out)

    @torch.no_grad()
    def max_attn_logit(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> float:
        """Unfused recompute of pre-softmax attention logits, for periodic diagnostics only."""
        bsz, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim))
        q = apply_rope(q.transpose(1, 2), cos, sin)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        logits = (q.float() @ k.float().transpose(-1, -2)) / (self.head_dim ** 0.5)
        return logits.abs().max().item()
