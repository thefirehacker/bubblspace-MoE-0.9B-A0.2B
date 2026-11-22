"""Contrastive entity-attribute binding corpus, v2 (for the Phase 2 anneal).

Why v2: the v1 corpus (scripts/gen_contrastive.py) expresses every fact with
one frame ("{event} because {entity} was {attr}") inside 6 fixed wrappers, two
of them with fixed phrases ("Put the other way around:", "People who saw it
agreed"). Its validation loss flattened at ~0.59 by Phase 2 step ~500 and the
model began reproducing those phrases in free generation, while probe v3
(different sentence structures) did not move. The model learned the template,
not the binding.

v2 keeps the target (the attribute must follow the right entity) and removes
the template:
  * 15 training sentence frames for "event + reason" (fronted, consequence,
    cleft, semicolon, ...), no single wrapper phrase.
  * Pronoun and former/latter frames with explicit resolution (Winograd-style
    reference, which v1 never trained).
  * Document shapes: single statement, both directions with varied
    connectives, corrections, varied Q&A/dialogue formats, narrative embedding
    with distractor sentences, two situations per document, and multi-step
    possession tracking.
  * Larger inventories (more dimensions, objects, relations).
  * Held out for evaluation (never in training): 4 sentence frames, ~20% of
    entity pairs per family, ~1/6 of names. --probe_out writes a probe (v4)
    built only from held-out frames + entities, in the probe_entity_binding.py
    format, to measure template-free transfer.

Contamination guards (asserted): no generator entity pair co-occurs in a
probe v2/v3 context; probe names are removed; no probe v3 frame phrase and no
held-out frame phrase appears in any training document.

Usage:
    python scripts/gen_contrastive_v2.py --out_dir /path/phase2/contrastive_v2 \
        --n_docs 2000000 --probe_files <v2.json> <v3.json> --probe_out probe_v4.json
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NAMES = (
    "Aarav Abigail Adaeze Aiko Alejandro Amara Amir Ana Anders Anika Arjun Astrid Ayesha Bao Beatriz "
    "Bilal Camila Chen Chloe Dante Darius Deepa Diego Dmitri Elena Elif Emeka Esther Fatima "
    "Felix Freya Gabriel Hana Hassan Helena Hiro Ibrahim Ines Isaac Ivan Jamal Jana Javier Jin Joao "
    "Jonas Kai Kamala Karim Kavya Keiko Kenji Kofi Lars Layla Leila Leo Lin Lucia Luis Maya Mehmet Mei "
    "Mira Mohammed Nadia Naledi Nikhil Nina Noor Olu Omar Oscar Pablo Paula Pedro Rafael Ravi Rhea "
    "Rosa Rohan Ruth Sakura Samir Sana Santiago Sara Selin Sergio Shreya Simone Sofia Soren Sunil Tariq "
    "Tess Thabo Tomas Uma Valentina Vikram Wanjiru Xavier Yara Yusuf Zainab Zara Zoe Ada Bruno Cyrus "
    "Dalia Ezra Farah Gideon Hugo Ilse Jude Kiran Lena Marco Nils Odile Pia Quinn Reza Saskia Teo "
    "Ulla Vera Wren Yosef Zeno Anouk Basil Clio Dov Emil Fern Gus Hedda Ike Juno Kasimir Lotte"
).split()

# comparative: (dimension, pairs (more, less), event templates with {a} {b}, more attrs, less attrs)
COMPARATIVE = [
    ("duration", [("lecture", "coffee break"), ("flight to Sydney", "bus ride across town"), ("winter", "lunch hour"),
                  ("movie trilogy", "short film"), ("road trip", "walk to the corner shop"), ("semester", "weekend workshop"),
                  ("century", "decade"), ("opera", "pop song"), ("night shift", "quick errand"), ("hike up the mountain", "stroll in the park"),
                  ("board meeting", "phone call"), ("ocean voyage", "ferry crossing"), ("chess tournament", "card trick"), ("novel", "postcard")],
     ["The {a} took longer than the {b}", "The {a} lasted longer than the {b}", "The {a} used up more of the day than the {b}",
      "People spent more time on the {a} than on the {b}"],
     ["much longer", "long and drawn out", "far more time-consuming", "very lengthy"], ["short", "over quickly", "brief", "quick"]),
    ("size", [("whale", "minnow"), ("stadium", "garden shed"), ("oak tree", "seedling"), ("truck", "bicycle"),
              ("mansion", "cottage"), ("continent", "island"), ("watermelon", "grape"), ("airliner", "kite"),
              ("cathedral", "chapel"), ("lake", "puddle"), ("warehouse", "kiosk"), ("elephant", "mouse"), ("galleon", "dinghy")],
     ["The {a} took up more space than the {b}", "The {a} needed more room than the {b}", "The {a} was harder to move than the {b}",
      "Fitting the {a} in the photo was harder than fitting the {b}"],
     ["enormous", "much bigger", "huge", "massive"], ["tiny", "much smaller", "small", "compact"]),
    ("weight", [("anvil", "feather"), ("piano", "violin"), ("bag of cement", "bag of crisps"), ("bowling ball", "tennis ball"),
                ("motorbike", "skateboard"), ("brick", "sponge"), ("suit of armour", "silk scarf"), ("iron gate", "paper screen"),
                ("marble statue", "cardboard cutout"), ("crate of books", "box of tissues")],
     ["The {a} was harder to carry than the {b}", "The {a} weighed more than the {b}", "The {a} bent the shelf more than the {b}",
      "Lifting the {a} took more effort than lifting the {b}"],
     ["very heavy", "much heavier", "dense and heavy", "a real burden"], ["light", "much lighter", "almost weightless", "easy to lift"]),
    ("speed", [("cheetah", "tortoise"), ("jet", "hot-air balloon"), ("racing bike", "tricycle"), ("bullet train", "steam engine"),
               ("speedboat", "rowing boat"), ("falcon", "pigeon"), ("motorway", "dirt track"), ("fibre connection", "dial-up modem"),
               ("express courier", "surface mail"), ("hare", "snail")],
     ["The {a} got there before the {b}", "The {a} finished sooner than the {b}", "The {a} covered the distance faster than the {b}",
      "The {b} arrived long after the {a}"],
     ["much faster", "very quick", "built for speed", "rapid"], ["slow", "much slower", "sluggish", "unhurried"]),
    ("price", [("yacht", "canoe"), ("gold watch", "plastic watch"), ("first-class ticket", "economy ticket"), ("sports stadium", "community hall"),
               ("designer coat", "thrift-store jacket"), ("penthouse", "studio flat"), ("truffle", "potato"), ("original manuscript", "photocopy"),
               ("diamond ring", "paper clip"), ("sports car", "scooter")],
     ["The {a} cost more than the {b}", "The {a} sold for a higher price than the {b}", "The {a} was harder to afford than the {b}",
      "Buyers saved for months for the {a} but not for the {b}"],
     ["expensive", "far more valuable", "rare and costly", "pricey"], ["cheap", "inexpensive", "common and cheap", "affordable"]),
    ("height", [("skyscraper", "bungalow"), ("giraffe", "dachshund"), ("redwood", "shrub"), ("lighthouse", "lamppost"),
                ("mountain", "hill"), ("radio mast", "fence post"), ("basketball player", "toddler"), ("tower crane", "stepladder"),
                ("church spire", "garden wall"), ("palm tree", "fern")],
     ["The {a} could be seen from further away than the {b}", "The {a} cast a longer shadow than the {b}", "The {a} reached higher than the {b}"],
     ["very tall", "much taller", "towering", "lofty"], ["short", "much shorter", "low", "squat"]),
    ("temperature", [("oven", "fridge"), ("desert", "glacier"), ("kettle", "ice tray"), ("summer afternoon", "winter night"),
                     ("forge", "cellar"), ("sauna", "swimming pool"), ("engine block", "window pane"), ("radiator", "marble floor")],
     ["The {a} felt hotter to the touch than the {b}", "The {a} melted the butter faster than the {b}", "The {a} made people sweat more than the {b}"],
     ["very hot", "much warmer", "scorching", "baking"], ["cold", "much cooler", "freezing", "chilly"]),
    ("loudness", [("jackhammer", "whisper"), ("drum kit", "flute"), ("thunderclap", "rustling leaf"), ("fire alarm", "ticking clock"),
                  ("rock concert", "library"), ("jet engine", "desk fan")],
     ["The {a} drowned out the {b}", "The {a} could be heard further away than the {b}", "The {a} woke the baby but the {b} did not"],
     ["deafening", "very loud", "noisy", "booming"], ["quiet", "faint", "soft", "barely audible"]),
    ("age", [("castle", "car park"), ("fossil", "newspaper"), ("cathedral", "shopping mall"), ("Roman road", "cycle lane"),
             ("manuscript", "text message"), ("pyramid", "petrol station")],
     ["The {a} had seen far more history than the {b}", "The {a} appeared in older records than the {b}", "The {a} predated the {b}"],
     ["ancient", "very old", "centuries old", "much older"], ["new", "recent", "modern", "much newer"]),
    ("brightness", [("searchlight", "candle"), ("noon sun", "crescent moon"), ("stadium floodlight", "night light"), ("flare", "glow-worm")],
     ["The {a} lit up more of the field than the {b}", "The {a} was easier to see from afar than the {b}", "The {a} hurt the eyes more than the {b}"],
     ["dazzling", "very bright", "blinding", "brilliant"], ["dim", "faint", "weak", "soft"]),
]

# physical fit: (object, container, event verb phrase, container attr, object attr)
FIT = [
    ("sofa", "lift", "would not fit in the", "too small", "too big"),
    ("grand piano", "hallway", "would not go down the", "too narrow", "too wide"),
    ("bookcase", "car boot", "would not fit in the", "too small", "too large"),
    ("ladder", "shed", "would not stand upright in the", "too low", "too tall"),
    ("barge", "tunnel under the canal", "could not pass through the", "too low", "too tall"),
    ("fridge freezer", "back door", "would not go through the", "too narrow", "too wide"),
    ("double bed", "spare room", "would not fit in the", "too small", "too big"),
    ("lorry", "village lane", "could not drive down the", "too narrow", "too wide"),
    ("Christmas tree", "living room", "would not stand up in the", "too low", "too tall"),
    ("cello case", "taxi", "would not fit in the", "too small", "too bulky"),
    ("motor home", "multi-storey car park", "could not enter the", "too low", "too high"),
    ("rug", "box", "would not fit in the", "too small", "too long"),
    ("dining table", "doorframe", "would not go through the", "too narrow", "too wide"),
    ("surfboard", "lift", "would not fit in the", "too short", "too long"),
    ("canoe", "garage", "would not fit in the", "too short", "too long"),
    ("giraffe", "horse trailer", "would not fit in the", "too low", "too tall"),
    ("aquarium", "alcove", "would not fit in the", "too narrow", "too wide"),
    ("wardrobe", "stairwell", "could not be carried up the", "too narrow", "too bulky"),
    ("trampoline", "back garden", "would not fit in the", "too small", "too big"),
    ("harp", "hatchback", "would not fit in the", "too small", "too tall"),
]

# person pairs: (verb, reasons for the OBJECT person, reasons for the SUBJECT person)
EMOTIONAL = [
    ("comforted", ["heartbroken", "grieving", "frightened", "lonely"], ["kind", "caring", "gentle", "patient"]),
    ("thanked", ["helpful", "generous", "supportive", "thoughtful"], ["grateful", "polite", "appreciative", "relieved"]),
    ("apologized to", ["hurt", "upset", "offended", "disappointed"], ["sorry", "ashamed", "remorseful", "guilty"]),
    ("forgave", ["truly sorry", "remorseful", "apologetic", "honest about it"], ["forgiving", "big-hearted", "easygoing", "generous"]),
    ("congratulated", ["successful", "victorious", "promoted", "the winner"], ["proud", "happy for them", "sporting", "warm"]),
    ("reassured", ["anxious", "nervous", "worried", "scared"], ["calm", "confident", "experienced", "steady"]),
    ("scolded", ["careless", "late again", "rude", "reckless"], ["strict", "angry", "frustrated", "impatient"]),
    ("avoided", ["rude", "unpleasant", "dishonest", "boastful"], ["shy", "uncomfortable", "busy", "wary"]),
    ("cheered up", ["gloomy", "downcast", "miserable", "feeling low"], ["cheerful", "funny", "upbeat", "good-humoured"]),
    ("envied", ["admired by everyone", "very successful", "lucky", "popular"], ["insecure", "jealous", "unhappy with their own life", "resentful"]),
]
POSSESSION = [
    ("lent money to", ["short of cash", "broke", "behind on rent", "struggling"], ["wealthy", "generous", "well off", "trusting"]),
    ("returned the book to", ["the owner", "asking for it back", "about to move away", "the one who had lent it"], ["finished with it", "honest", "done reading", "careful with borrowed things"]),
    ("paid", ["owed the money", "the one who did the work", "waiting for payment", "the seller"], ["in debt", "the buyer", "honest", "keen to settle up"]),
    ("asked for help from", ["an expert", "strong", "experienced", "good with tools"], ["stuck", "overwhelmed", "new to the job", "confused"]),
    ("gave a gift to", ["celebrating a birthday", "leaving the company", "a close friend", "recovering from illness"], ["thoughtful", "generous", "fond of them", "grateful"]),
    ("borrowed a ladder from", ["the owner of a ladder", "a helpful neighbour", "a builder", "prepared for jobs like this"], ["fixing the roof", "without a ladder", "in the middle of repairs", "short of tools"]),
    ("sold a bicycle to", ["looking for a cheap ride", "a keen cyclist", "in need of transport", "the buyer"], ["moving abroad", "short of space", "the seller", "upgrading to a car"]),
    ("rented a room to", ["a student", "new in town", "looking for a place to stay", "the tenant"], ["the landlord", "the owner of a big house", "living alone", "glad of the income"]),
]
ABILITY = [
    ("beat", ["out of practice", "inexperienced", "tired", "overconfident"], ["well trained", "more skilled", "experienced", "better prepared"]),
    ("taught", ["a beginner", "eager to learn", "new to the subject", "struggling with it"], ["an expert", "experienced", "patient", "qualified"]),
    ("outran", ["slower", "out of shape", "injured", "carrying a heavy bag"], ["faster", "fitter", "a trained sprinter", "well rested"]),
    ("hired", ["highly qualified", "talented", "experienced", "reliable"], ["short-staffed", "expanding", "growing quickly", "in need of help"]),
    ("corrected", ["mistaken", "wrong", "confused", "careless with the figures"], ["an expert", "careful", "certain", "well informed"]),
    ("rescued", ["stranded", "in danger", "trapped", "lost"], ["brave", "a trained lifeguard", "strong", "quick-thinking"]),
    ("tutored", ["behind in maths", "struggling at school", "preparing for an exam", "a slow reader"], ["a gifted student", "a former teacher", "patient", "top of the class"]),
]
ITEMS = ["umbrella", "camera", "notebook", "guitar", "torch", "map", "key", "scarf", "telescope", "kettle", "hammer", "chessboard"]

# sentence frames (training). Each: f(event, event_lc, E, A, people) -> sentence
def _cap(s):
    return s[0].upper() + s[1:]

TRAIN_FRAMES = [
    lambda ev, lc, E, A, p: f"{ev} because {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}, since {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev} as {E} was {A}.",
    lambda ev, lc, E, A, p: f"Because {E} was {A}, {lc}.",
    lambda ev, lc, E, A, p: f"Since {E} was {A}, {lc}.",
    lambda ev, lc, E, A, p: f"{_cap(E)} was {A}, so {lc}.",
    lambda ev, lc, E, A, p: f"{_cap(E)} was {A}, and that is why {lc}.",
    lambda ev, lc, E, A, p: f"{ev}. The reason: {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}; {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}, simply because {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}, which happened because {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}. Why? {_cap(E)} was {A}.",
    lambda ev, lc, E, A, p: f"{ev} for one reason: {E} was {A}.",
    lambda ev, lc, E, A, p: f"{ev}, mainly because {E} was {A}.",
    lambda ev, lc, E, A, p: f"It was {E} {'who' if p else 'that'} was {A}, which explains why {lc}.",
]
# held-out frames: evaluation only (probe v4), never in training. Prefix ends right before the attribute.
HELDOUT_FRAMES = [
    lambda ev, E: f"{ev}, given that {E} was",
    lambda ev, E: f"{ev}; the trouble was that {E} was",
    lambda ev, E: f"{ev}. Anyone could tell that {E} was",
    lambda ev, E: f"{ev}, owing to the fact that {E} was",
]
BANNED_IN_TRAIN = ["given that", "the trouble was", "anyone could tell", "owing to",          # held-out frames
                   "the one that was", "the person who was", "of the two", "was clearly the",  # probe v3 frames
                   "put the other way around", "people who saw it agreed"]                    # v1 fixed phrases

BOTH_CONNECTIVES = ["", "Also, ", "And ", "At the same time, ", "Equally, ", "It is also true that ", "Seen from the other side, ",
                    "Looking at it the other way, ", "Likewise, ", "On top of that, ", "From the other angle, ", "Add to that: "]
FILLERS = ["It was a {day} {time}.", "The street outside was quiet.", "Rain had been falling since early morning.", "Nobody was in a hurry that day.",
           "A {thing} lay forgotten on the table.", "Someone had left the radio on.", "The {day} market was busy as usual.",
           "It had been a long week for everyone.", "A dog barked somewhere down the road.", "The {thing} on the shelf had not moved in years.",
           "Later that {time}, everyone went home.", "The news spread quickly.", "Nothing else about the day was unusual."]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["morning", "afternoon", "evening"]
THINGS = ["vase", "newspaper", "teacup", "pencil", "clock", "coat", "basket", "lamp"]


@dataclass
class Sit:
    event: str      # main clause, no final period
    e1: str         # entity with a1 (with determiner, or a name)
    e2: str
    a1: str
    a2: str
    people: bool
    first: int      # which of (1, 2) is mentioned first in the event
    family: str


def _lc(ev, people):
    return ev if people else ev[0].lower() + ev[1:]


def split_holdout(seq, every=5):
    train = [x for i, x in enumerate(seq) if i % every != every - 1]
    held = [x for i, x in enumerate(seq) if i % every == every - 1]
    return train, held


def make_sit(rng, family, pairs_or_frames, names):
    if family == "comparative":
        dim, pairs, events, mores, lesses = pairs_or_frames
        a, b = rng.choice(pairs)
        ev = rng.choice(events).format(a=a, b=b)
        first = 1 if ev.index(a) < ev.index(b) else 2
        return Sit(ev, f"the {a}", f"the {b}", rng.choice(mores), rng.choice(lesses), False, first, "comparative")
    if family == "fit":
        obj, cont, vp, ca, oa = pairs_or_frames
        return Sit(f"The {obj} {vp} {cont}", f"the {cont}", f"the {obj}", ca, oa, False, 2, "fit")
    verb, obj_r, subj_r = pairs_or_frames
    x, y = rng.sample(names, 2)
    return Sit(f"{x} {verb} {y}", y, x, rng.choice(obj_r), rng.choice(subj_r), True, 2, family)


def statement(rng, s, which):
    E, A = (s.e1, s.a1) if which == 1 else (s.e2, s.a2)
    return rng.choice(TRAIN_FRAMES)(s.event, _lc(s.event, s.people), E, A, s.people)


def pronoun_statement(rng, s, which):
    E, A = (s.e1, s.a1) if which == 1 else (s.e2, s.a2)
    if s.people:
        ref = "the former" if which == s.first else "the latter"
        return f"{s.event} because {ref} was {A}. Here, \"{ref}\" refers to {E}."
    res = rng.choice([f"Here, \"it\" refers to {E}.", f"In that sentence, \"it\" means {E}.", f"(\"It\" is {E}.)"])
    return f"{s.event} because it was {A}. {res}"


def question(rng, s, which):
    E, A = (s.e1, s.a1) if which == 1 else (s.e2, s.a2)
    O = s.e2 if which == 1 else s.e1
    who = "Who" if s.people else "Which"
    x, y = rng.sample([E, O], 2)
    fmt = rng.randrange(9)
    if fmt == 0:
        return f"Q: {who} was {A}? A: {_cap(E)}."
    if fmt == 1:
        return f"\"{who} was {A}?\" someone asked. \"{_cap(E)},\" came the reply."
    if fmt == 2:
        return f"Quiz: {who.lower()} was {A}? Answer: {E}."
    if fmt == 3:
        return f"If you are asked {who.lower()} was {A}, the answer is {E}."
    if fmt == 4:
        return f"{who} was {A}, {x} or {y}? {_cap(E)}."
    if fmt == 5:
        return f"{_cap(x)} or {y}: {who.lower()} was {A}? {_cap(E)}, clearly."
    if fmt == 6:
        return f"Check: {who.lower()} was {A}? {_cap(E)}."
    if fmt == 7:
        return f"So {who.lower()} was {A}? That would be {E}."
    return f"{who} was {A}? {_cap(E)}."


def filler(rng):
    return rng.choice(FILLERS).format(day=rng.choice(DAYS), time=rng.choice(TIMES), thing=rng.choice(THINGS))


def correction(rng, s, which):
    E, A = (s.e1, s.a1) if which == 1 else (s.e2, s.a2)
    O = s.e2 if which == 1 else s.e1
    fmt = rng.randrange(4)
    if fmt == 0:
        return f"{s.event}. It was not that {O} was {A}; {E} was {A}."
    if fmt == 1:
        return f"{s.event}. Was {O} {A}? No. {_cap(E)} was."
    if fmt == 2:
        return f"{s.event}. Some might guess {O} was {A}, but in fact {E} was {A}."
    return f"{s.event}. It was {E}, not {O}, {'who' if s.people else 'that'} was {A}."


def possession_chain(rng, names):
    item = rng.choice(ITEMS)
    people = rng.sample(names, rng.randint(3, 4))
    verbs = ["gave it to", "lent it to", "passed it to", "handed it to", "sold it to"]
    parts = [f"{people[0]} had a {item}."]
    for i in range(1, len(people)):
        parts.append(f"{'Then ' if i > 1 else ''}{people[i - 1]} {rng.choice(verbs)} {people[i]}.")
    end = rng.randrange(3)
    if end == 0:
        parts.append(f"Now the {item} is with {people[-1]}.")
    elif end == 1:
        parts.append(f"Q: Who has the {item} now? A: {people[-1]}. Q: Who had it first? A: {people[0]}.")
    else:
        parts.append(f"So {people[-1]} has the {item}, and {people[0]} no longer does.")
    return " ".join(parts)


def document(rng, fam_pool, names):
    if rng.random() < 0.08:
        return possession_chain(rng, names)
    fam, spec = rng.choice(fam_pool)
    s = make_sit(rng, fam, spec, names)
    w = rng.choice([1, 2])
    shape = rng.randrange(9)
    if shape == 0:      # single statement
        return statement(rng, s, w)
    if shape == 1:      # both directions, varied connective
        c = rng.choice(BOTH_CONNECTIVES)
        second = statement(rng, s, 3 - w)
        if c and second.split()[0] not in NAMES:  # keep names capitalized
            second = second[0].lower() + second[1:]
        return f"{statement(rng, s, w)} {c}{second}"
    if shape == 2:      # correction
        return correction(rng, s, w)
    if shape == 3:      # statement + question(s)
        out = [statement(rng, s, w), question(rng, s, w)]
        if rng.random() < 0.5:
            out.append(question(rng, s, 3 - w))
        return " ".join(out)
    if shape == 4:      # narrative embedding with distractors
        fl = list(dict.fromkeys(filler(rng) for _ in range(3)))
        body = fl[:rng.randint(1, 2)] + [statement(rng, s, w)]
        if rng.random() < 0.5 and len(fl) > 2:
            body.append(fl[2])
        return " ".join(body)
    if shape == 5:      # pronoun / former-latter with resolution
        return pronoun_statement(rng, s, w)
    if shape == 6:      # two situations, then a question about one
        fam2, spec2 = rng.choice(fam_pool)
        s2 = make_sit(rng, fam2, spec2, names)
        w2 = rng.choice([1, 2])
        target, tw = rng.choice([(s, w), (s2, w2)])
        return f"{statement(rng, s, w)} {statement(rng, s2, w2)} {question(rng, target, tw)}"
    if shape == 7:      # both directions, both as questions
        return f"{s.event}. {question(rng, s, 1)} {question(rng, s, 2)}"
    return f"{statement(rng, s, w)} {filler(rng)} {question(rng, s, w)}"


def load_probe_contexts(probe_files):
    ctxs = []
    for pf in probe_files:
        for c in json.loads(Path(pf).read_text())["cases"]:
            ctxs.append((c["ctxA"] + " " + c["ctxB"]).lower())
    return ctxs


def build_probe_v4(rng, held_pool, held_names, n_per_family=24):
    cases = []
    by_fam = {}
    for fam, spec in held_pool:
        by_fam.setdefault(fam, []).append(spec)
    for fam, specs in sorted(by_fam.items()):
        seen = set()
        tries = 0
        while sum(1 for c in cases if c["category"] == fam) < n_per_family and tries < 5000:
            tries += 1
            s = make_sit(rng, fam, rng.choice(specs), held_names)
            frame = rng.choice(HELDOUT_FRAMES)
            ctxA, ctxB = frame(s.event, s.e1), frame(s.event, s.e2)
            if ctxA in seen:
                continue
            seen.add(ctxA)
            cases.append({"name": f"v4-{fam}-{len(cases)}", "category": fam, "ctxA": ctxA, "ctxB": ctxB,
                          "c1": f" {s.a1}.", "c2": f" {s.a2}."})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_docs", type=int, default=2_000_000)
    ap.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--probe_files", nargs="*", default=[])
    ap.add_argument("--probe_out", default=None)
    ap.add_argument("--dry_run", type=int, default=0, help="print N sample docs and exit")
    args = ap.parse_args()

    # guard 1: probe names never appear in generated docs
    probe_ctxs = load_probe_contexts(args.probe_files)
    probe_text = " " + " ".join(probe_ctxs) + " "
    names = [n for n in NAMES if f" {n.lower()} " not in probe_text and f" {n.lower()}." not in probe_text]
    print(f"[guard] dropped {len(NAMES) - len(names)} probe names; {len(names)} remain")
    train_names, held_names = split_holdout(names, every=6)

    # hold out ~20% of entity pairs / relation frames per family for probe v4
    train_pool, held_pool = [], []
    for dim, pairs, events, mores, lesses in COMPARATIVE:
        tr, he = split_holdout(pairs)
        train_pool.append(("comparative", (dim, tr, events, mores, lesses)))
        if he:
            held_pool.append(("comparative", (dim, he, events, mores, lesses)))
    tr, he = split_holdout(FIT)
    train_pool += [("fit", f) for f in tr]
    held_pool += [("fit", f) for f in he]
    for fam, frames in (("emotional", EMOTIONAL), ("possession", POSSESSION), ("ability", ABILITY)):
        tr, he = split_holdout(frames)
        train_pool += [(fam, f) for f in tr]
        held_pool += [(fam, f) for f in he]

    # guard 2: no generator entity pair co-occurs in any probe v2/v3 context
    gen_pairs = [(a, b) for _, pairs, *_ in COMPARATIVE for a, b in pairs] + [(o, c) for o, c, *_ in FIT]
    overlap = [(a, b) for a, b in gen_pairs if any(f" {a.lower()} " in f" {c} " and f" {b.lower()} " in f" {c} " for c in probe_ctxs)]
    assert not overlap, f"generator entity pairs co-occur in probe cases: {overlap}"

    rng = random.Random(args.seed)
    if args.dry_run:
        for _ in range(args.dry_run):
            print("-", document(rng, train_pool, train_names))
        v4 = build_probe_v4(random.Random(args.seed + 1), held_pool, held_names)
        print(f"[probe v4] {len(v4)} cases, e.g.:")
        for c in v4[::25]:
            print("  ", c["category"], "|", c["ctxA"], "->", c["c1"], "||", c["ctxB"], "->", c["c2"])
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eot = tok.eos_token_id
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    pos, shard, total, unique, batch = 0, 0, 0, set(), []

    def flush():
        nonlocal pos, shard, total
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids = ids + [eot]
            if pos + len(ids) > args.shard_tokens:
                buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1; pos = 0
            buf[pos:pos + len(ids)] = ids
            pos += len(ids); total += len(ids)
        batch.clear()

    for _ in range(args.n_docs):
        doc = document(rng, train_pool, train_names)
        if doc in unique:
            continue
        low = doc.lower()
        bad = [b for b in BANNED_IN_TRAIN if b in low]  # guard 3: no held-out / probe-v3 / v1 phrase
        assert not bad, f"banned phrase {bad} in: {doc}"
        unique.add(doc)
        batch.append(doc)
        if len(batch) >= 20_000:
            flush()
    if batch:
        flush()
    if pos:
        buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1

    meta = {"dataset": "synthetic_contrastive_binding_v2", "tokenizer_path": args.tokenizer_path, "total_tokens": total,
            "n_docs": args.n_docs, "unique_docs": len(unique), "n_shards": shard, "dtype": "uint16", "eot_token": eot,
            "real_vocab_size": len(tok), "is_code": False, "fim_rate": 0.0, "probe_exclusion_files": args.probe_files,
            "train_frames": len(TRAIN_FRAMES), "heldout_frames": len(HELDOUT_FRAMES)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.probe_out:
        cases = build_probe_v4(random.Random(args.seed + 1), held_pool, held_names)
        Path(args.probe_out).write_text(json.dumps({
            "description": "Probe v4: entity-attribute binding scored by log-likelihood, built ONLY from sentence frames, "
                           "entity pairs/relations and names held out of the contrastive v2 training corpus. Measures "
                           "template-free transfer of binding.",
            "categories": sorted({c["category"] for c in cases}), "cases": cases}, indent=2))
        print(f"[probe v4] wrote {len(cases)} cases to {args.probe_out}")


if __name__ == "__main__":
    main()
