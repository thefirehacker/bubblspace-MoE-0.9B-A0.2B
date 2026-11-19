import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.moe import MoEFFN, probe_grouped_mm


def _dense_reference_forward(moe: MoEFFN, x: torch.Tensor) -> torch.Tensor:
    """Correctness oracle only: computes every token through every expert
    (O(n_experts) wasted FLOPs vs. real top-k routing), masks/weights by the
    router's top-k assignment. Never used in production — only to validate
    the two real backends against each other.
    """
    bsz, seq_len, d = x.shape
    x_flat = x.reshape(-1, d)
    logits = moe.router(x_flat)
    probs = F.softmax(logits.float(), dim=-1)
    topk_probs, topk_idx = probs.topk(moe.top_k, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

    weight = torch.zeros_like(probs)
    weight.scatter_(1, topk_idx, topk_probs)

    out = torch.zeros_like(x_flat)
    for e in range(moe.n_experts):
        h = F.silu(x_flat @ moe.W_gate[e]) * (x_flat @ moe.W_up[e])
        out_e = h @ moe.W_down[e]
        out = out + out_e * weight[:, e : e + 1]
    return out.view(bsz, seq_len, d)


def _make_moe(backend: str, seed: int = 0) -> MoEFFN:
    torch.manual_seed(seed)
    moe = MoEFFN(
        d_model=32,
        n_experts=8,
        top_k=2,
        expert_hidden=16,
        backend=backend,
        router_init_std=0.02,
    )
    return moe


def test_fallback_matches_dense_reference():
    moe = _make_moe("fallback")
    x = torch.randn(2, 5, 32, requires_grad=True)

    out_fallback, _, _, _ = moe(x)
    out_dense = _dense_reference_forward(moe, x)

    torch.testing.assert_close(out_fallback, out_dense, atol=1e-5, rtol=1e-5)


def test_grouped_mm_matches_fallback():
    if not probe_grouped_mm() or not torch.cuda.is_available():
        import pytest

        pytest.skip("torch._grouped_mm not usable on this machine/build")

    device = "cuda"
    torch.manual_seed(0)
    moe_fallback = _make_moe("fallback").to(device)
    moe_grouped = copy.deepcopy(moe_fallback)
    moe_grouped.backend = "grouped_mm"

    x = torch.randn(2, 5, 32, device=device, requires_grad=True)

    out_fallback, lb_f, z_f, _ = moe_fallback(x)
    out_grouped, lb_g, z_g, _ = moe_grouped(x)

    torch.testing.assert_close(out_fallback, out_grouped, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lb_f, lb_g, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z_f, z_g, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    test_fallback_matches_dense_reference()
    test_grouped_mm_matches_fallback()
    print("ok")
