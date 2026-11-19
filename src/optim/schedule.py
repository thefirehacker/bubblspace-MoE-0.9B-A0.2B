import math

from torch.optim.lr_scheduler import LambdaLR


def lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    decay: str,
    min_lr_ratio: float,
    wsd_decay_frac: float,
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)

    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))

    if decay == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    elif decay == "wsd":
        decay_start = 1.0 - wsd_decay_frac
        if progress < decay_start:
            return 1.0
        decay_progress = (progress - decay_start) / wsd_decay_frac
        return 1.0 - (1.0 - min_lr_ratio) * decay_progress
    else:
        raise ValueError(f"Unknown decay type: {decay}")


def build_scheduler(optimizer, warmup_steps: int, total_steps: int, decay: str, min_lr_ratio: float, wsd_decay_frac: float):
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, warmup_steps, total_steps, decay, min_lr_ratio, wsd_decay_frac),
    )
