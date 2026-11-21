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


def phase2_lr_multiplier(
    step: int,
    warmup_steps: int,
    decay_start_step: int,
    total_steps: int,
    start_ratio: float,
    end_ratio: float,
) -> float:
    """Phase 2 continual-training WSD, as a multiplier on the Phase 2 PEAK lr.

    0 -> warmup_steps:            linear re-warm from start_ratio (Phase 1's
                                  final lr / peak) up to 1.0
    warmup -> decay_start_step:   stable at 1.0 (Stage 2A plateau)
    decay_start -> total_steps:   linear decay 1.0 -> end_ratio (Stage 2B anneal)
    """
    if step < warmup_steps:
        return start_ratio + (1.0 - start_ratio) * (step + 1) / max(1, warmup_steps)
    if step < decay_start_step:
        return 1.0
    progress = min(1.0, (step - decay_start_step) / max(1, total_steps - decay_start_step))
    return 1.0 - (1.0 - end_ratio) * progress


def build_phase2_scheduler(optimizer, peak_lr: float, warmup_steps: int, decay_start_step: int, total_steps: int, start_lr: float, end_lr: float):
    # LambdaLR multiplies each group's `initial_lr`. A Phase 1 optimizer
    # state_dict carries initial_lr=6e-4 and lr=6e-5, so overwrite both with
    # the Phase 2 peak BEFORE constructing the scheduler.
    for group in optimizer.param_groups:
        group["lr"] = peak_lr
        group["initial_lr"] = peak_lr
    start_ratio, end_ratio = start_lr / peak_lr, end_lr / peak_lr
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step: phase2_lr_multiplier(step, warmup_steps, decay_start_step, total_steps, start_ratio, end_ratio),
    )
