import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """G: (batch, rows, cols). Returns an orthogonalized version via quintic
    Newton-Schulz iteration (Keller Jordan's coefficients), batched over the
    leading dim so ordinary 2D matrices (batch=1) and per-expert stacks
    (batch=n_experts) share one path — no Python loop over experts.
    """
    assert G.dim() == 3
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


class Muon(Optimizer):
    """Moonlight-style Muon: Newton-Schulz orthogonalized momentum updates,
    scaled by update_scale = 0.2*sqrt(max(rows, cols)) to match AdamW's
    update RMS, with decoupled weight decay (the Moonlight variant adds WD;
    the original Muon did not — intentional here, not an oversight).

    Accepts both ordinary 2D weight matrices and 3D stacked-expert weight
    tensors (n_experts, rows, cols) in the same param group.
    """

    def __init__(self, params, lr=6e-4, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dim() not in (2, 3):
                    raise ValueError(
                        f"Muon only supports 2D or 3D (stacked-expert) params, got shape {tuple(grad.shape)}"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                was_2d = update.dim() == 2
                update3d = update.unsqueeze(0) if was_2d else update

                ortho = _zeropower_via_newtonschulz5(update3d, ns_steps)
                rows, cols = ortho.shape[-2], ortho.shape[-1]
                scale = 0.2 * (max(rows, cols) ** 0.5)
                ortho = ortho * scale

                if was_2d:
                    ortho = ortho.squeeze(0)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(ortho, alpha=-lr)

        return loss
