"""Fill-in-the-middle transforms for code samples.

Per the tokenizer spec: FIM tokens are only ever inserted into code samples
when FIM is explicitly enabled -- never into raw pretraining web text.

Two entry points:
- apply_fim(): file-level PSM, used by Phase 1's prepare.py (kept unchanged
  so Phase 1 stays reproducible).
- apply_window_fim(): Phase 2, applied on the fly to a single document span
  *inside* an already-sampled training window, so <|fim_prefix|>,
  <|fim_suffix|> and <|fim_middle|> are always co-resident in the same
  window (the file-level version split them across windows for any file
  longer than seq_len). 50/50 PSM / SPM.
"""

import random

import numpy as np


def apply_fim(
    token_ids: list[int],
    fim_prefix_id: int,
    fim_middle_id: int,
    fim_suffix_id: int,
    rng: random.Random,
) -> list[int]:
    if len(token_ids) < 4:
        return token_ids

    lo, hi = sorted(rng.sample(range(1, len(token_ids)), 2))
    prefix, middle, suffix = token_ids[:lo], token_ids[lo:hi], token_ids[hi:]

    # PSM layout: <PRE> prefix <SUF> suffix <MID> middle
    return [fim_prefix_id, *prefix, fim_suffix_id, *suffix, fim_middle_id, *middle]


def apply_window_fim(
    window: np.ndarray,
    eot_id: int,
    fim_prefix_id: int,
    fim_middle_id: int,
    fim_suffix_id: int,
    rng: random.Random,
    min_span: int = 16,
) -> np.ndarray:
    """Rewrite the longest document span inside `window` as a FIM example.

    Returns an array of the SAME length as the input (the 3 inserted
    sentinels push the tail out; it is truncated), so callers can keep a
    fixed (seq_len + 1) window. If no span is long enough, returns the window
    unchanged.
    """
    n = len(window)
    eot_pos = np.flatnonzero(window == eot_id)
    bounds = [-1, *eot_pos.tolist(), n]
    best = None
    for a, b in zip(bounds[:-1], bounds[1:]):
        start, end = a + 1, b  # span excludes the EOT itself
        if end - start >= min_span and (best is None or end - start > best[1] - best[0]):
            best = (start, end)
    if best is None:
        return window

    start, end = best
    span = window[start:end].tolist()
    lo, hi = sorted(rng.sample(range(1, len(span)), 2))
    prefix, middle, suffix = span[:lo], span[lo:hi], span[hi:]

    if rng.random() < 0.5:
        # PSM: <PRE> prefix <SUF> suffix <MID> middle
        rewritten = [fim_prefix_id, *prefix, fim_suffix_id, *suffix, fim_middle_id, *middle]
    else:
        # SPM (StarCoder variant): <PRE> <SUF> suffix <MID> prefix middle
        rewritten = [fim_prefix_id, fim_suffix_id, *suffix, fim_middle_id, *prefix, *middle]

    out = np.concatenate([window[:start], np.asarray(rewritten, dtype=window.dtype), window[end:]])
    return out[:n]
