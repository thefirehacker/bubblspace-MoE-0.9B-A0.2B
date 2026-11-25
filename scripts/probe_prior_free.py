"""Re-score saved binding-probe results without the model's built-in preferences (doc/binding-study-02-priors-and-word-order.md).

scripts/probe_entity_binding.py saves four log-probs per minimal pair:
    score_A_c1, score_A_c2   context A (correct ending c1)
    score_B_c1, score_B_c2   context B (correct ending c2)

Pair rule (the original metric): A prefers c1 AND B prefers c2. A model that prefers one ENDING in general (a prior)
fails the pair even if the context moves its preference the right way.

Prior-free metric: the 2x2 interaction
    (score_A_c1 - score_A_c2) - (score_B_c1 - score_B_c2) > 0
asks only whether changing the entity moves the preference toward the right ending. A general preference for one ending
or for one entity cancels out. Chance is 50%. It is a weaker bar than the pair rule (direction only), so both are reported,
together with the median size of the effect in nats.

Failure types under the pair rule:
    reversed      both contexts prefer the WRONG ending (a consistent wrong rule, e.g. passive read as active)
    same ending   both contexts prefer the same ending (a prior, or the context is ignored)

    python scripts/probe_prior_free.py benchmark/phase2/probe_v5_phase2_step22888.json [more.json ...]
"""

import collections
import json
import statistics
import sys


def analyse(path):
    cats = collections.OrderedDict()
    for r in json.load(open(path))["results"]:
        c = cats.setdefault(r["category"], {"n": 0, "pair": 0, "prior_free": 0, "reversed": 0, "same_ending": 0, "effect": []})
        a, b = r["pref_A"] == r["c1"], r["pref_B"] == r["c2"]
        inter = (r["score_A_c1"] - r["score_A_c2"]) - (r["score_B_c1"] - r["score_B_c2"])
        c["n"] += 1
        c["pair"] += a and b
        c["prior_free"] += inter > 0
        c["reversed"] += (not a) and (not b)
        c["same_ending"] += a != b
        c["effect"].append(inter)
    return cats


def main():
    for path in sys.argv[1:]:
        cats = analyse(path)
        n = sum(c["n"] for c in cats.values())
        print(f"\n{path}")
        print(f"  all: pair rule {sum(c['pair'] for c in cats.values())}/{n}   prior-free {sum(c['prior_free'] for c in cats.values())}/{n}")
        for k, c in sorted(cats.items()):
            print(f"  {k:16s} pair {c['pair']:3d}/{c['n']:<3d}  prior-free {c['prior_free']:3d}/{c['n']:<3d}  "
                  f"reversed {c['reversed']:3d}  same ending {c['same_ending']:3d}  median effect {statistics.median(c['effect']):+6.1f} nats")


if __name__ == "__main__":
    main()
