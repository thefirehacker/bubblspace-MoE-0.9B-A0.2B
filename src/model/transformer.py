import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .init import init_weights
from .moe import ExpertLoadTracker
from .norm import ZeroCenteredRMSNorm
from .rope import precompute_rope


class KVCache:
    """Inference-only key/value cache: one dict per layer plus the number of
    positions already processed (the RoPE offset for the next call). Use one
    prefill call with the prompt, then feed one token per call."""

    def __init__(self, n_layers: int):
        self.layers = [dict() for _ in range(n_layers)]
        self.seq_len = 0


class Transformer(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.n_layers = model_cfg.n_layers

        self.embed = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=model_cfg.d_model,
                    attn_cfg=model_cfg.attention,
                    moe_cfg=model_cfg.moe,
                    norm_eps=model_cfg.norm_eps,
                )
                for _ in range(model_cfg.n_layers)
            ]
        )
        self.final_norm = ZeroCenteredRMSNorm(model_cfg.d_model, eps=model_cfg.norm_eps)

        cos, sin = precompute_rope(
            model_cfg.attention.head_dim, model_cfg.max_seq_len, model_cfg.attention.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        init_weights(self, model_cfg.init.std, model_cfg.n_layers)

    def make_load_tracker(self, device) -> ExpertLoadTracker:
        return ExpertLoadTracker(self.n_layers, self.cfg.moe.n_experts, device)

    def new_kv_cache(self) -> KVCache:
        return KVCache(self.n_layers)

    def forward(self, idx: torch.Tensor, load_tracker: ExpertLoadTracker | None = None, block_mask=None, kv_cache: KVCache | None = None):
        # block_mask=None keeps the exact Phase 1 path (causal SDPA). Phase 2
        # builds a doc mask outside the compiled graph (attention.build_doc_block_mask)
        # and passes it in here.
        bsz, seq_len = idx.shape
        x = self.embed(idx)
        start = kv_cache.seq_len if kv_cache is not None else 0
        cos = self.rope_cos[start : start + seq_len].to(x.device)
        sin = self.rope_sin[start : start + seq_len].to(x.device)

        lb_loss_sum = x.new_zeros(())
        z_loss_sum = x.new_zeros(())
        hard_counts = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.layers[i] if kv_cache is not None else None
            x, lb, z, hc = block(x, cos, sin, load_tracker, i, block_mask, layer_cache)
            lb_loss_sum = lb_loss_sum + lb
            z_loss_sum = z_loss_sum + z
            hard_counts.append(hc)

        if kv_cache is not None:
            kv_cache.seq_len += seq_len
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)

        lb_loss = lb_loss_sum / self.n_layers
        z_loss = z_loss_sum / self.n_layers
        return logits, lb_loss, z_loss, hard_counts
