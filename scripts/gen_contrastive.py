"""Generate a synthetic contrastive entity-attribute binding corpus for Phase 2.

Why: the 100-case binding probe (benchmark/data/probefinding-winograndestyle/)
found a flat zero on comparative, emotional and possession/obligation
attribution, and greedy generation confirmed it ("The marathon took longer
than the sprint because the sprint was... longer than the marathon"). Passive
web prose rarely puts BOTH directions of a minimal pair next to each other,
so next-token prediction never gets pressure to flip the attribute when the
named entity flips. This corpus does exactly that, densely.

Each document states the same situation with the attribute bound to each
entity, in varied surface forms (paired statements, Q&A, explicit negation,
short narrative), across five relation families. Entity pairs used by the
probe set are excluded so the probe remains a held-out measurement.

Output: flat uint16 shards (EOT-separated documents), same format as
src/data/prepare.py, consumed as a `flat` pool by MultiPoolTokenDataset.

Usage:
    python scripts/gen_contrastive.py --out_dir /path/phase2/contrastive \
        --n_docs 1500000 --tokenizer_path bubblspace-tokenizer
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

NAMES = (
    "Aarav Abigail Adaeze Aiko Alejandro Amara Amir Ana Anders Anika Arjun Astrid Ayesha Bao Beatriz "
    "Bilal Camila Carlos Chen Chloe Dante Darius Deepa Diego Dmitri Elena Elif Emeka Emma Esther Fatima "
    "Felix Freya Gabriel Grace Hana Hassan Helena Hiro Ibrahim Ines Isaac Ivan Jamal Jana Javier Jin Joao "
    "Jonas Kai Kamala Karim Kavya Keiko Kenji Kofi Lars Layla Leila Leo Lin Lucia Luis Maya Mehmet Mei "
    "Mira Mohammed Nadia Naledi Nikhil Nina Noor Olu Omar Oscar Pablo Paula Pedro Priya Rafael Ravi Rhea "
    "Rosa Rohan Ruth Sakura Samir Sana Santiago Sara Selin Sergio Shreya Simone Sofia Soren Sunil Tariq "
    "Tess Thabo Tomas Uma Valentina Vikram Wanjiru Wei Xavier Yara Yusuf Zainab Zara Zoe"
).split()

# ---------------------------------------------------------------- comparative
# (dimension, [ (more_entity, less_entity), ... ], outcome phrases, more_attr, less_attr)
COMPARATIVE = [
    ("duration", [("lecture", "coffee break"), ("flight to Sydney", "bus ride across town"), ("winter", "lunch hour"),
                  ("movie trilogy", "short film"), ("road trip", "walk to the corner shop"), ("semester", "weekend workshop"),
                  ("century", "decade"), ("opera", "pop song"), ("night shift", "quick errand"), ("hike up the mountain", "stroll in the park")],
     ["took longer than the {b}", "lasted longer than the {b}", "used up more of the day than the {b}"],
     ["much longer", "long and drawn out", "far more time-consuming"], ["short", "over quickly", "brief"]),
    ("size", [("whale", "minnow"), ("stadium", "garden shed"), ("oak tree", "seedling"), ("truck", "bicycle"),
              ("mansion", "cottage"), ("continent", "island"), ("watermelon", "grape"), ("airliner", "kite"),
              ("cathedral", "chapel"), ("lake", "puddle")],
     ["took up more space than the {b}", "needed more room than the {b}", "was harder to move than the {b}"],
     ["enormous", "much bigger", "huge"], ["tiny", "much smaller", "small"]),
    ("weight", [("anvil", "feather"), ("piano", "violin"), ("bag of cement", "bag of crisps"), ("bowling ball", "tennis ball"),
                ("motorbike", "skateboard"), ("brick", "sponge"), ("suit of armour", "silk scarf"), ("iron gate", "paper screen")],
     ["was harder to carry than the {b}", "weighed more than the {b}", "bent the shelf more than the {b}"],
     ["very heavy", "much heavier", "dense and heavy"], ["light", "much lighter", "almost weightless"]),
    ("speed", [("cheetah", "tortoise"), ("jet", "hot-air balloon"), ("racing bike", "tricycle"), ("bullet train", "steam engine"),
               ("speedboat", "rowing boat"), ("falcon", "pigeon"), ("motorway", "dirt track"), ("fibre connection", "dial-up modem")],
     ["got there before the {b}", "finished sooner than the {b}", "covered the distance faster than the {b}"],
     ["much faster", "very quick", "built for speed"], ["slow", "much slower", "sluggish"]),
    ("price", [("yacht", "canoe"), ("gold watch", "plastic watch"), ("first-class ticket", "economy ticket"), ("sports stadium", "community hall"),
               ("designer coat", "thrift-store jacket"), ("penthouse", "studio flat"), ("truffle", "potato"), ("original manuscript", "photocopy")],
     ["cost more than the {b}", "sold for a higher price than the {b}", "was harder to afford than the {b}"],
     ["expensive", "far more valuable", "rare and costly"], ["cheap", "inexpensive", "common and cheap"]),
    ("height", [("skyscraper", "bungalow"), ("giraffe", "dachshund"), ("redwood", "shrub"), ("lighthouse", "lamppost"),
                ("mountain", "hill"), ("radio mast", "fence post"), ("basketball player", "toddler"), ("tower crane", "stepladder")],
     ["could be seen from further away than the {b}", "cast a longer shadow than the {b}", "reached higher than the {b}"],
     ["very tall", "much taller", "towering"], ["short", "much shorter", "low"]),
    ("temperature", [("oven", "fridge"), ("desert", "glacier"), ("kettle", "ice tray"), ("summer afternoon", "winter night"),
                     ("forge", "cellar"), ("sauna", "swimming pool"), ("engine block", "window pane")],
     ["felt hotter to the touch than the {b}", "melted the butter faster than the {b}", "made people sweat more than the {b}"],
     ["very hot", "much warmer", "scorching"], ["cold", "much cooler", "freezing"]),
]

# ---------------------------------------------------------------- physical fit
FIT = [  # (object, container, verb phrase, container_attr, object_attr)
    ("sofa", "lift", "would not fit in the", "too small", "too big"),
    ("grand piano", "hallway", "would not go down the", "too narrow", "too wide"),
    ("bookcase", "car boot", "would not fit in the", "too small", "too large"),
    ("ladder", "shed", "would not stand upright in the", "too low", "too tall"),
    ("boat", "tunnel under the canal", "could not pass through the", "too low", "too tall"),
    ("fridge freezer", "back door", "would not go through the", "too narrow", "too wide"),
    ("double bed", "spare room", "would not fit in the", "too small", "too big"),
    ("lorry", "village lane", "could not drive down the", "too narrow", "too wide"),
    ("christmas tree", "living room", "would not stand up in the", "too low", "too tall"),
    ("cello case", "taxi", "would not fit in the", "too small", "too bulky"),
    ("motor home", "car park", "could not enter the", "too low", "too high"),
    ("rug", "box", "would not fit in the", "too small", "too long"),
    ("dining table", "doorframe", "would not go through the", "too narrow", "too wide"),
    ("surfboard", "elevator", "would not fit in the", "too short", "too long"),
]

# ---------------------------------------------------------------- person-pair relations
# (verb phrase, reasons_for_object_person, reasons_for_subject_person)
EMOTIONAL = [
    ("comforted", ["heartbroken", "grieving", "frightened", "lonely"], ["kind", "caring", "gentle", "patient"]),
    ("thanked", ["helpful", "generous", "supportive", "thoughtful"], ["grateful", "polite", "appreciative", "relieved"]),
    ("apologized to", ["hurt", "upset", "offended", "disappointed"], ["sorry", "ashamed", "remorseful", "guilty"]),
    ("forgave", ["truly sorry", "remorseful", "apologetic", "honest about it"], ["forgiving", "big-hearted", "easygoing", "generous"]),
    ("congratulated", ["successful", "victorious", "promoted", "the winner"], ["proud", "happy for them", "sporting", "warm"]),
    ("reassured", ["anxious", "nervous", "worried", "scared"], ["calm", "confident", "experienced", "steady"]),
    ("scolded", ["careless", "late again", "rude", "reckless"], ["strict", "angry", "frustrated", "impatient"]),
    ("avoided", ["rude", "unpleasant", "dishonest", "boastful"], ["shy", "uncomfortable", "busy", "wary"]),
]
POSSESSION = [
    ("lent money to", ["short of cash", "broke", "behind on rent", "struggling"], ["wealthy", "generous", "well off", "trusting"]),
    ("returned the book to", ["the owner", "asking for it back", "about to move away", "the one who had lent it"], ["finished with it", "honest", "done reading", "careful with borrowed things"]),
    ("paid", ["owed the money", "the one who did the work", "waiting for payment", "the seller"], ["in debt", "the buyer", "honest", "keen to settle up"]),
    ("asked for help from", ["an expert", "strong", "experienced", "good with tools"], ["stuck", "overwhelmed", "new to the job", "confused"]),
    ("gave a gift to", ["celebrating a birthday", "leaving the company", "a close friend", "recovering from illness"], ["thoughtful", "generous", "fond of them", "grateful"]),
    ("borrowed a ladder from", ["the owner of a ladder", "a helpful neighbour", "a builder", "prepared for jobs like this"], ["fixing the roof", "without a ladder", "in the middle of repairs", "short of tools"]),
]
ABILITY = [
    ("beat", ["out of practice", "inexperienced", "tired", "overconfident"], ["well trained", "more skilled", "experienced", "better prepared"]),
    ("taught", ["a beginner", "eager to learn", "new to the subject", "struggling with it"], ["an expert", "experienced", "patient", "qualified"]),
    ("outran", ["slower", "out of shape", "injured", "carrying a heavy bag"], ["faster", "fitter", "a trained sprinter", "well rested"]),
    ("hired", ["highly qualified", "talented", "experienced", "reliable"], ["short-staffed", "expanding", "growing quickly", "in need of help"]),
    ("corrected", ["mistaken", "wrong", "confused", "careless with the figures"], ["an expert", "careful", "certain", "well informed"]),
]


def comparative_doc(rng):
    dim, pairs, outcomes, more_attrs, less_attrs = rng.choice(COMPARATIVE)
    a, b = rng.choice(pairs)
    outcome = rng.choice(outcomes).format(b=b)
    more, less = rng.choice(more_attrs), rng.choice(less_attrs)
    head = f"The {a} {outcome}"
    s_a = f"{head} because the {a} was {more}."
    s_b = f"{head} because the {b} was {less}."
    return _style(rng, head, a, b, more, less, s_a, s_b)


def fit_doc(rng):
    obj, cont, vp, cont_attr, obj_attr = rng.choice(FIT)
    head = f"The {obj} {vp} {cont}"
    s_c = f"{head} because the {cont} was {cont_attr}."
    s_o = f"{head} because the {obj} was {obj_attr}."
    return _style(rng, head, cont, obj, cont_attr, obj_attr, s_c, s_o)


def person_doc(rng, frames):
    verb, obj_reasons, subj_reasons = rng.choice(frames)
    x, y = rng.sample(NAMES, 2)
    ro, rs = rng.choice(obj_reasons), rng.choice(subj_reasons)
    head = f"{x} {verb} {y}"
    s_obj = f"{head} because {y} was {ro}."
    s_subj = f"{head} because {x} was {rs}."
    return _style(rng, head, y, x, ro, rs, s_obj, s_subj, people=True)


def _style(rng, head, e1, e2, attr1, attr2, s1, s2, people=False):
    """e1 carries attr1 (sentence s1), e2 carries attr2 (sentence s2)."""
    the1 = e1 if people else f"the {e1}"
    the2 = e2 if people else f"the {e2}"
    style = rng.randrange(6)
    if style == 0:  # paired statements, random order
        return " ".join(rng.sample([s1, s2], 2))
    if style == 1:  # explicit contrast
        return f"{s1} Put the other way around: {s2}"
    if style == 2:  # Q&A
        return (f"Question: {head}. Which one was {attr1}? Answer: {the1.capitalize() if not people else the1}. "
                f"Which one was {attr2}? Answer: {the2.capitalize() if not people else the2}.")
    if style == 3:  # negation
        return (f"{head}, and it was not because {the2} was {attr1}. "
                f"{the1[0].upper() + the1[1:]} was {attr1}, while {the2} was {attr2}.")
    if style == 4:  # single direction only (keeps the corpus from being trivially paired)
        return rng.choice([s1, s2])
    # style 5: short narrative
    return (f"{head}. People who saw it agreed on the reason: {the1} was {attr1}. "
            f"Nobody thought {the2} was {attr1}; if anything, {the2} was {attr2}.")


def load_probe_contexts(probe_files):
    ctxs = []
    for pf in probe_files:
        data = json.loads(Path(pf).read_text())
        for c in data["cases"]:
            ctxs.append((c["ctxA"] + " " + c["ctxB"]).lower())
    return ctxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_docs", type=int, default=1_500_000)
    ap.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--probe_files", nargs="*", default=[])
    args = ap.parse_args()

    # contamination guard, so the probe stays a held-out measurement:
    # (1) no generator entity PAIR may co-occur in any probe sentence;
    # (2) no person name used in the probe may appear in generated docs.
    probe_ctxs = load_probe_contexts(args.probe_files)
    gen_pairs = [(a, b) for _, ps, *_ in COMPARATIVE for a, b in ps] + [(o, c) for o, c, *_ in FIT]
    overlap = [(a, b) for a, b in gen_pairs if any(f" {a.lower()} " in f" {c} " and f" {b.lower()} " in f" {c} " for c in probe_ctxs)]
    assert not overlap, f"generator entity pairs co-occur in probe cases: {overlap}"
    probe_text = " ".join(probe_ctxs)
    dropped = [n for n in NAMES if f" {n.lower()} " in f" {probe_text} "]
    NAMES[:] = [n for n in NAMES if n not in dropped]
    print(f"[guard] dropped probe names from generator: {dropped}; {len(NAMES)} names remain")

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eot = tok.eos_token_id
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    makers = [
        (0.30, comparative_doc),
        (0.15, fit_doc),
        (0.20, lambda r: person_doc(r, EMOTIONAL)),
        (0.20, lambda r: person_doc(r, POSSESSION)),
        (0.15, lambda r: person_doc(r, ABILITY)),
    ]
    weights = [w for w, _ in makers]

    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    pos, shard, total, unique = 0, 0, 0, set()
    batch = []

    def flush_batch():
        nonlocal pos, shard, total
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids = ids + [eot]
            if pos + len(ids) > args.shard_tokens:
                buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1; pos = 0
            buf[pos : pos + len(ids)] = ids
            pos += len(ids)
            total += len(ids)
        batch.clear()

    # n_docs = generation ATTEMPTS; only unique documents are written, so
    # small template families (e.g. physical fit) cannot dominate by sheer
    # duplication. Repetition is controlled later by the pool's sampling weight.
    for i in range(args.n_docs):
        doc = rng.choices(makers, weights=weights)[0][1](rng)
        if doc in unique:
            continue
        unique.add(doc)
        batch.append(doc)
        if len(batch) >= 10_000:
            flush_batch()
    if batch:
        flush_batch()
    if pos:
        buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1

    meta = {
        "dataset": "synthetic_contrastive_binding",
        "tokenizer_path": args.tokenizer_path,
        "total_tokens": total,
        "n_docs": args.n_docs,
        "unique_docs": len(unique),
        "n_shards": shard,
        "dtype": "uint16",
        "eot_token": eot,
        "real_vocab_size": len(tok),
        "is_code": False,
        "fim_rate": 0.0,
        "probe_exclusion_files": args.probe_files,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
