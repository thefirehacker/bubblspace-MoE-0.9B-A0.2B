import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.model.transformer import Transformer


def test_param_count_matches_target():
    cfg = load_config("configs/base.yaml")
    with torch.device("meta"):
        model = Transformer(cfg.model)

    total = sum(p.numel() for p in model.parameters())

    moe = cfg.model.moe
    attn = cfg.model.attention
    d_model = cfg.model.d_model
    n_layers = cfg.model.n_layers

    active_per_layer_experts = moe.top_k * (2 * d_model * moe.expert_hidden + moe.expert_hidden * d_model)
    active_experts_total = active_per_layer_experts * n_layers

    attn_params_per_layer = (
        attn.n_heads * attn.head_dim * d_model  # q
        + attn.n_kv_heads * attn.head_dim * d_model  # k
        + attn.n_kv_heads * attn.head_dim * d_model  # v
        + attn.n_heads * attn.head_dim * d_model  # o
        + (attn.n_heads * attn.head_dim * d_model if attn.output_gate else 0)  # gate
    )
    embed_params = cfg.model.vocab_size * d_model
    router_params = d_model * moe.n_experts * n_layers

    active_estimate = attn_params_per_layer * n_layers + embed_params + router_params + active_experts_total

    # Targets from the architecture spec: ~917M total / ~213M active.
    assert 850_000_000 < total < 1_000_000_000, f"total params {total:,} out of expected range"
    assert 180_000_000 < active_estimate < 260_000_000, f"active params {active_estimate:,} out of expected range"


if __name__ == "__main__":
    test_param_count_matches_target()
    print("ok")
