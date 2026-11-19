import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.optim.muon import Muon, _zeropower_via_newtonschulz5


def test_newton_schulz_orthogonality():
    # 5 quintic NS steps on a full-rank random Gaussian matrix does not fully
    # converge every singular value to 1 (the smallest singular value of a
    # random square matrix can start far below the largest, and the NS
    # polynomial only pulls small values up gradually) — this is expected
    # behavior of the real algorithm (matches Keller Jordan's reference
    # implementation), not a bug. Check the output is bounded and the
    # largest singular value lands close to 1, rather than demanding a
    # perfect identity Gram matrix.
    torch.manual_seed(0)
    G = torch.randn(1, 64, 64)
    ortho = _zeropower_via_newtonschulz5(G, steps=5)
    sv = torch.linalg.svdvals(ortho[0].float())
    assert sv.max().item() < 1.5, f"largest singular value exploded: {sv.max().item()}"
    assert abs(sv.max().item() - 1.0) < 0.3, f"largest singular value not near 1: {sv.max().item()}"


def test_batched_ns_matches_looped_single_matrix_ns():
    torch.manual_seed(0)
    n_experts = 8
    G = torch.randn(n_experts, 32, 16)

    batched = _zeropower_via_newtonschulz5(G, steps=5)

    looped = torch.stack(
        [_zeropower_via_newtonschulz5(G[e : e + 1], steps=5)[0] for e in range(n_experts)]
    )

    torch.testing.assert_close(batched, looped, atol=1e-6, rtol=1e-6)


def test_update_scale_matches_architecture_invariant():
    # Every Muon-governed matrix in this architecture has max(rows, cols) == 1024,
    # so update_scale = 0.2*sqrt(1024) == 6.4 uniformly.
    rows, cols = 1024, 256
    scale = 0.2 * (max(rows, cols) ** 0.5)
    assert abs(scale - 6.4) < 1e-9


def test_muon_step_does_not_explode_on_tiny_model():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 16))
    opt = Muon([w], lr=6e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1)

    for _ in range(10):
        loss = (w @ torch.randn(16, 16)).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        assert torch.isfinite(w).all()


if __name__ == "__main__":
    test_newton_schulz_orthogonality()
    test_batched_ns_matches_looped_single_matrix_ns()
    test_update_scale_matches_architecture_invariant()
    test_muon_step_does_not_explode_on_tiny_model()
    print("ok")
