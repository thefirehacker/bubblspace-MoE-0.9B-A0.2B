from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def probe_grouped_mm() -> bool:
    """Checks whether torch._grouped_mm is present and actually runs on this
    build/device. This is a private, experimental PyTorch API whose signature
    has shifted across versions — never trust hasattr() alone, always probe
    with a real call. Run this once at startup (see scripts/train.py) to
    resolve moe.backend: "auto" to "grouped_mm" or "fallback".
    """
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a = torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        b = torch.randn(2, 8, 8, device=device, dtype=torch.bfloat16)
        offs = torch.tensor([2, 4], dtype=torch.int32, device=device)
        torch._grouped_mm(a, b, offs=offs)
        return True
    except Exception:
        return False


class ExpertLoadTracker:
    """Running, detached per-expert token counts accumulated across the
    grad-accumulation micro-batches of one optimizer step ("global_batch"
    load balancing). The training loop must call reset() after each
    optimizer step.
    """

    def __init__(self, n_layers: int, n_experts: int, device):
        self.counts = torch.zeros(n_layers, n_experts, device=device)

    def update(self, layer_idx: int, hard_counts: torch.Tensor) -> None:
        self.counts[layer_idx] += hard_counts.detach()

    def reset(self) -> None:
        self.counts.zero_()


class MoEFFN(nn.Module):
    """Dropless top-k MoE FFN (SwiGLU experts) with two interchangeable
    compute backends:
      - "grouped_mm": torch._grouped_mm fast path.
      - "fallback":   gather + per-expert-loop. The Python loop runs only in
                       this forward pass, never in the optimizer — it is NOT
                       the same constraint as Muon's batched Newton-Schulz.

    Experts are stored as stacked nn.Parameter tensors (not an nn.ModuleList)
    so both backends and the Muon optimizer's batched per-expert step can
    treat them as a single (n_experts, ...) tensor.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        expert_hidden: int,
        backend: str,
        router_init_std: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.backend = backend

        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=router_init_std)

        # Default std=0.02 init so MoEFFN is sane to use/test standalone;
        # Transformer-level init_weights rescales W_down to residual_proj_std
        # when this module is used inside a full model.
        self.W_gate = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden).normal_(std=0.02))
        self.W_up = nn.Parameter(torch.empty(n_experts, d_model, expert_hidden).normal_(std=0.02))
        self.W_down = nn.Parameter(torch.empty(n_experts, expert_hidden, d_model).normal_(std=0.02))

    def forward(
        self,
        x: torch.Tensor,
        load_tracker: ExpertLoadTracker | None = None,
        layer_idx: int | None = None,
    ):
        bsz, seq_len, d = x.shape
        x_flat = x.reshape(-1, d)
        n_tokens = x_flat.shape[0]

        logits = self.router(x_flat)  # (N, E), compute in the router's own dtype
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = (topk_probs / topk_probs.sum(dim=-1, keepdim=True)).to(x.dtype)

        if self.backend == "grouped_mm":
            out_flat = self._forward_grouped_mm(x_flat, topk_idx, topk_probs)
        elif self.backend == "fallback":
            out_flat = self._forward_fallback(x_flat, topk_idx, topk_probs)
        else:
            raise ValueError(f"Unknown MoE backend: {self.backend}")

        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

        flat_expert = topk_idx.reshape(-1)
        hard_counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        hard_counts.scatter_add_(
            0, flat_expert, torch.ones_like(flat_expert, dtype=torch.float32)
        )

        if load_tracker is not None and layer_idx is not None:
            load_tracker.update(layer_idx, hard_counts)
            running = load_tracker.counts[layer_idx]
            f_i = running / running.sum().clamp_min(1.0)
        else:
            f_i = hard_counts / hard_counts.sum().clamp_min(1.0)

        p_i = probs.mean(dim=0)  # differentiable, this micro-batch only
        lb_loss = self.n_experts * (f_i.detach() * p_i).sum()

        return out_flat.view(bsz, seq_len, d), lb_loss, z_loss, hard_counts.detach()

    def _forward_fallback(
        self, x_flat: torch.Tensor, topk_idx: torch.Tensor, topk_probs: torch.Tensor
    ) -> torch.Tensor:
        n_tokens, d = x_flat.shape
        flat_expert = topk_idx.reshape(-1)
        flat_token = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(self.top_k)
        flat_weight = topk_probs.reshape(-1)

        out = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            mask = flat_expert == e
            if not mask.any():
                continue
            tok_idx = flat_token[mask]
            w = flat_weight[mask].unsqueeze(-1)
            xe = x_flat[tok_idx]
            h = F.silu(xe @ self.W_gate[e]) * (xe @ self.W_up[e])
            down = h @ self.W_down[e]
            out.index_add_(0, tok_idx, down * w)
        return out

    def _forward_grouped_mm(
        self, x_flat: torch.Tensor, topk_idx: torch.Tensor, topk_probs: torch.Tensor
    ) -> torch.Tensor:
        n_tokens, d = x_flat.shape
        flat_expert = topk_idx.reshape(-1)
        flat_token = torch.arange(n_tokens, device=x_flat.device).repeat_interleave(self.top_k)
        flat_weight = topk_probs.reshape(-1)

        sort_idx = torch.argsort(flat_expert, stable=True)
        sorted_expert = flat_expert[sort_idx]
        sorted_token = flat_token[sort_idx]
        sorted_weight = flat_weight[sort_idx]

        counts = torch.bincount(sorted_expert, minlength=self.n_experts)
        offs = counts.cumsum(0).to(torch.int32)

        # torch._grouped_mm is a private op that torch.autocast does not know
        # to cast — it requires bf16/fp16 inputs explicitly, unlike ordinary
        # matmul which autocast handles automatically under the training
        # loop's autocast context.
        gathered_x = x_flat[sorted_token].to(torch.bfloat16)
        w_gate = self.W_gate.to(torch.bfloat16)
        w_up = self.W_up.to(torch.bfloat16)
        w_down = self.W_down.to(torch.bfloat16)

        gate = torch._grouped_mm(gathered_x, w_gate, offs=offs)
        up = torch._grouped_mm(gathered_x, w_up, offs=offs)
        h = F.silu(gate) * up
        down = torch._grouped_mm(h, w_down, offs=offs)
        down = down * sorted_weight.unsqueeze(-1).to(down.dtype)

        out = torch.zeros(n_tokens, d, device=x_flat.device, dtype=down.dtype)
        out.index_add_(0, sorted_token, down)
        return out.to(x_flat.dtype)
