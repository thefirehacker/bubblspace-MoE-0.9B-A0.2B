"""Fill-in-the-middle (PSM: prefix/suffix/middle) transform for code samples.

Per the tokenizer spec: FIM tokens are only ever inserted into code samples
when FIM is explicitly enabled — never into raw pretraining web text.
"""

import random


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
