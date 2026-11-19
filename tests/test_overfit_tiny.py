import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AttentionConfig, InitConfig, ModelConfig, MoEConfig
from src.model.transformer import Transformer
from src.optim.param_groups import build_optimizers


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _tiny_model_config() -> ModelConfig:
    attn = AttentionConfig(
        n_heads=2, head_dim=8, n_kv_heads=1, qk_norm=True, rope_theta=10000, output_gate=True, bias=False
    )
    moe = MoEConfig(
        n_experts=4,
        top_k=2,
        expert_hidden=8,
        activation="silu",
        router_init_std=0.02,
        backend="fallback",
        load_balance_coef=0.01,
        router_z_loss_coef=0.001,
    )
    init = InitConfig(std=0.02)
    return ModelConfig(
        n_layers=2,
        d_model=16,
        vocab_size=32,
        tie_embeddings=True,
        max_seq_len=16,
        norm_eps=1e-6,
        attention=attn,
        moe=moe,
        init=init,
    )


def test_overfit_tiny_fixed_batch():
    """Wiring smoke test: forward/backward/Muon+AdamW step, no real data or
    GPU required. A tiny model repeatedly trained on one fixed batch should
    drive its loss down substantially, confirming the full pipeline is wired
    correctly before any GPU time is spent.
    """
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    model = Transformer(cfg)

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.randint(0, cfg.vocab_size, (2, 8))

    muon_cfg = _Obj(lr=1e-2, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0)
    adamw_cfg = _Obj(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    muon_opt, adamw_opt = build_optimizers(model, muon_cfg, adamw_cfg)

    load_tracker = model.make_load_tracker(x.device)

    losses = []
    for _ in range(100):
        load_tracker.reset()
        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        logits, lb_loss, z_loss, _ = model(x, load_tracker)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)) + 0.01 * lb_loss + 0.001 * z_loss
        loss.backward()
        muon_opt.step()
        adamw_opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert all(torch.isfinite(torch.tensor(l)) for l in losses), "loss became non-finite"


if __name__ == "__main__":
    test_overfit_tiny_fixed_batch()
    print("ok")
