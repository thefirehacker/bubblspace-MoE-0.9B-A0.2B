"""Bit-reproducible evaluation.

Without this, two identical forward passes of the MoE model differ by up to
~0.1 in the logits: non-deterministic CUDA kernels introduce tiny float
differences, and those flip near-tied top-8 expert choices in the router
(measured: 0-12 tokens per layer get a different expert set between two runs
of the same input), which visibly changes outputs. Greedy generations can
then diverge after a few tokens, and close log-likelihood probe cases can
flip, so the same checkpoint would score differently run to run.

With deterministic algorithms enabled, repeated runs are bit-identical
(verified on both MoE backends). Call make_deterministic() BEFORE any CUDA
work. Eval-only: training does not use this.
"""

import os

import torch


def make_deterministic(seed: int = 0) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
