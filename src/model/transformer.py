import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .init import init_weights
from .moe import ExpertLoadTracker
from .norm import ZeroCenteredRMSNorm
from .rope import precompute_rope


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

    def forward(self, idx: torch.Tensor, load_tracker: ExpertLoadTracker | None = None, block_mask=None):
        # block_mask=None keeps the exact Phase 1 path (causal SDPA). Phase 2
        # builds a doc mask outside the compiled graph (attention.build_doc_block_mask)
        # and passes it in here.
        bsz, seq_len = idx.shape
        x = self.embed(idx)
        cos = self.rope_cos[:seq_len].to(x.device)
        sin = self.rope_sin[:seq_len].to(x.device)

        lb_loss_sum = x.new_zeros(())
        z_loss_sum = x.new_zeros(())
        hard_counts = []
        for i, block in enumerate(self.blocks):
            x, lb, z, hc = block(x, cos, sin, load_tracker, i, block_mask)
            lb_loss_sum = lb_loss_sum + lb
            z_loss_sum = z_loss_sum + z
            hard_counts.append(hc)

        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)

        lb_loss = lb_loss_sum / self.n_layers
        z_loss = z_loss_sum / self.n_layers
        return logits, lb_loss, z_loss, hard_counts
