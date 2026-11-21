import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import ZeroCenteredRMSNorm
from .rope import apply_rope

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # older torch: doc masking unavailable, SDPA causal path still works
    create_block_mask = None
    flex_attention = None


def build_doc_block_mask(idx: torch.Tensor, eot_id: int):
    """Causal + same-document BlockMask for flex_attention.

    A token's document id = number of <|endoftext|> tokens strictly before it,
    so the EOT token belongs to the document it closes and the next token
    starts a fresh one.
    """
    if create_block_mask is None:
        raise RuntimeError("flex_attention not available in this torch build")
    bsz, seq_len = idx.shape
    is_eot = (idx == eot_id).to(torch.int32)
    doc_ids = torch.cumsum(is_eot, dim=1) - is_eot

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask(mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=idx.device)


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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q.transpose(1, 2), cos, sin)  # (bsz, n_heads, seq_len, head_dim)
        k = apply_rope(k.transpose(1, 2), cos, sin)
        v = v.transpose(1, 2)

        if block_mask is not None:
            # Phase 2: causal + same-document mask (tokens never attend across
            # <|endoftext|> into an unrelated document / packed episode).
            attn_out = flex_attention(
                q, k, v, block_mask=block_mask, enable_gqa=(self.n_heads != self.n_kv_heads)
            )
        else:
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
