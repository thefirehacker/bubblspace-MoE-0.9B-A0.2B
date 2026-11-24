"""Contrastive entity-attribute binding corpus, v3 (for Phase 2.5a).

Why v3: v2 (scripts/gen_contrastive_v2.py) taught binding that transferred to new words and names
(probe v4, held-out frames/pairs/names: 17 -> 82/112 in the Phase 2 anneal) but not to new sentence
STRUCTURES (probe v3: 4 -> 6/70). v3 keeps all of v2 (same vocabulary, same held-out split, same guards)
and adds structural variety:

  New training structures (on v2's families):
    reported speech ("Everyone knew that ..."), whereas/while contrasts, negation pairs ("X was not A;
    Y was"), "both ... but only", direct comparatives in both directions ("The A was longer than the B.
    The B was shorter than the A."), role statements/questions (who acted, who was affected),
    counterfactuals for physical fit ("Had the lift not been too small, ...").
  New relation families:
    part-whole ("The wheel is part of the bicycle, not the other way around"), cause-effect ("The storm
    caused the flood; the flood did not cause the storm"), temporal order ("Maya left before Leo").

  Held out ENTIRELY (never in training, asserted) -> probe v5, built from TRAINING names/entities so it
  measures pure structural transfer:
    * passive voice     "Leo was comforted by Samir because Leo was ..."
    * relative clauses  "Samir comforted Leo, who was ..."
    * "respectively"    "Leo and Samir were, respectively, ..."

  Still banned (as in v2): probe v3 frames, v2's held-out frames (probe v4), v1's fixed phrases; v2's
  held-out entity pairs and names stay out of training, so probe v4 remains clean.

Outputs: tokenized shards (training) and labeled parquet (text, family, structure) for analysis / HF.

    python scripts/gen_contrastive_v3.py --out_dir /data/phase25/contrastive_v3 --parquet_dir /raw/contrastive_hf/v3 \
        --n_docs 9000000 --probe_files <v2.json> <v3.json> <v4.json> --probe_out probe_v5.json
"""

import argparse
import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("v2", Path(__file__).with_name("gen_contrastive_v2.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
_cap, Sit = v2._cap, v2.Sit

# comparative adjectives per v2 dimension: (more, less)
COMPARATIVE_ADJ = {"duration": ("longer", "shorter"), "size": ("bigger", "smaller"), "weight": ("heavier", "lighter"),
                   "speed": ("faster", "slower"), "price": ("more expensive", "cheaper"), "height": ("taller", "shorter"),
                   "temperature": ("hotter", "colder"), "loudness": ("louder", "quieter"), "age": ("older", "newer"),
                   "brightness": ("brighter", "dimmer")}

PART_WHOLE = [("wheel", "bicycle"), ("page", "book"), ("petal", "flower"), ("keyboard", "laptop"), ("engine", "car"),
              ("branch", "tree"), ("chapter", "novel"), ("kitchen", "house"), ("lens", "camera"), ("string", "guitar"),
              ("sail", "sailing boat"), ("handle", "door"), ("leaf", "plant"), ("screen", "phone"), ("blade", "knife"),
              ("wing", "aeroplane"), ("button", "shirt"), ("brick", "wall"), ("key", "piano"), ("carriage", "train"),
              ("verse", "song"), ("tile", "roof"), ("drawer", "desk"), ("spoke", "wheel"), ("finger", "hand"),
              ("yolk", "egg"), ("player", "team"), ("star", "galaxy"), ("island", "archipelago"), ("cell", "tissue"),
              ("cog", "clock"), ("lid", "jar"), ("chimney", "factory"), ("sleeve", "jacket"), ("heel", "shoe"), ("bristle", "brush"),
              ("rung", "ladder"), ("pedal", "bicycle"), ("hinge", "gate"), ("fin", "fish"), ("claw", "crab"), ("feather", "bird"),
              ("stamen", "flower"), ("bulb", "lamp"), ("nib", "pen"), ("mast", "ship"), ("scene", "play"), ("note", "chord"),
              ("word", "sentence"), ("township", "district"), ("molecule", "crystal"), ("neuron", "brain"), ("seed", "fruit"),
              ("trunk", "tree"), ("cockpit", "aeroplane"), ("tread", "tyre"), ("antenna", "radio"), ("chip", "computer"),
              ("platform", "station"), ("goalpost", "football pitch")]

CAUSE_EFFECT = [("storm", "flood"), ("fire", "smoke"), ("drought", "crop failure"), ("heavy rain", "puddle"),
                ("earthquake", "landslide"), ("spark", "explosion"), ("infection", "fever"), ("heatwave", "power cut"),
                ("frost", "burst pipe"), ("landslide", "road closure"), ("cold snap", "frozen pond"), ("traffic jam", "late arrival"),
                ("loud alarm", "sudden waking"), ("strike", "cancelled trains"), ("leak", "damp wall"), ("overgrazing", "bare soil"),
                ("power surge", "blown fuse"), ("typo", "failed login"), ("flat tyre", "missed bus"), ("high tide", "flooded path"),
                ("blizzard", "school closure"), ("lightning strike", "forest fire"), ("oil spill", "dead seabirds"), ("price rise", "falling sales"),
                ("dropped glass", "broken shards"), ("burst dam", "flooded valley"), ("long drought", "dry riverbed"), ("hailstorm", "dented cars"),
                ("missed alarm", "late start"), ("software bug", "system crash"), ("heavy snowfall", "blocked road"), ("gas leak", "evacuation"),
                ("low battery", "dead phone"), ("burnt toast", "smoke alarm"), ("pollen", "sneezing"), ("deforestation", "soil erosion"),
                ("volcanic eruption", "ash cloud"), ("ice on the road", "car skid"), ("strong wind", "fallen tree"), ("overflowing drain", "flooded street"),
                ("unpaid bill", "disconnected line")]

TEMPORAL_EVENTS = ["left the party", "arrived at the station", "finished the race", "woke up", "reached the summit",
                   "handed in the essay", "boarded the bus", "signed the contract", "crossed the bridge", "joined the club"]

PASSIVE = {"comforted": "comforted", "thanked": "thanked", "congratulated": "congratulated", "reassured": "reassured",
           "scolded": "scolded", "avoided": "avoided", "envied": "envied", "forgave": "forgiven", "beat": "beaten",
           "taught": "taught", "outran": "outrun", "hired": "hired", "corrected": "corrected", "rescued": "rescued",
           "tutored": "tutored"}

REPORTERS = ["Everyone knew that", "It was obvious to the neighbours that", "The report made clear that", "Most people agreed that",
             "The teacher explained that", "According to the story,", "It turned out that", "The witnesses confirmed that",
             "As the notes recorded,", "Nobody doubted that"]

# held-out structures (probe v5) + everything v2 banned
BANNED_V3 = v2.BANNED_IN_TRAIN + ["respectively", ", who was", ", which was", ", who were", ", which were"]
PASSIVE_RE = re.compile(r"\b(was|were|is|are|been|being) (" + "|".join(sorted(set(PASSIVE.values()))) + r"|[a-z]+ed) by\b")


def lc_first(s):
    return s if s.split()[0] in v2.NAMES else s[0].lower() + s[1:]


def comparative_dim(s, train_pool):
    return None


# ---------- new structures on v2 situations ----------
def reported(rng, s, w):
    return f"{rng.choice(REPORTERS)} {lc_first(v2.statement(rng, s, w))}"


def whereas(rng, s, w):
    E1, A1, E2, A2 = (s.e1, s.a1, s.e2, s.a2) if w == 1 else (s.e2, s.a2, s.e1, s.a1)
    c = rng.choice(["whereas", "while", "but"])
    return f"{s.event}. {_cap(E1)} was {A1}, {c} {E2} was {A2}."


def negation(rng, s, w):
    E, A = (s.e1, s.a1) if w == 1 else (s.e2, s.a2)
    O = s.e2 if w == 1 else s.e1
    fmt = rng.randrange(3)
    if fmt == 0:
        return f"{s.event}. {_cap(O)} was not {A}; {E} was."
    if fmt == 1:
        return f"{s.event}. {_cap(E)} was {A}, not {O}."
    return f"{s.event}. Whatever else was true, {O} was not the {A} one here. {_cap(E)} was."


def both_only(rng, s, w):
    E, A = (s.e1, s.a1) if w == 1 else (s.e2, s.a2)
    if not s.people:
        return None
    return f"{s.event}. Both {s.e1} and {s.e2} were involved, but only {E} was {A}."


def role_statement(rng, s):
    if not s.people:
        return None
    x, y = s.e2, s.e1  # make_sit: event "x verb y", e1 = y (object), e2 = x (subject)
    verb = s.event[len(x) + 1:-(len(y) + 1)]
    fmt = rng.randrange(3)
    if fmt == 0:
        return f"{s.event}. {_cap(x)} did the action; {y} was on the receiving end."
    if fmt == 1:
        return f"In \"{s.event}\", {x} is the doer and {y} is the receiver."
    return f"{s.event}. Q: Who {verb} whom? A: {x} {verb} {y}, not the reverse."


def counterfactual(rng, s, spec, names):
    if s.family != "fit":
        return None
    obj, cont, vp, ca, oa = spec
    who = rng.choice(names)
    lead = rng.choice([f"{s.event}.", f"{who} tried, but the {obj} {vp} {cont}.", f"{s.event}, and {who} had to give up.",
                       f"{rng.choice(v2.DAYS)}: {s.event[0].lower() + s.event[1:]}."])
    if rng.random() < 0.5:
        tail = rng.choice([f"Had the {cont} not been {ca}, the {obj} would have gone in; the {obj} itself was not the problem.",
                           f"If the {cont} had not been {ca}, it would have worked. The {obj} was fine.",
                           f"A different {cont}, one that was not {ca}, would have solved it."])
    else:
        tail = rng.choice([f"Had the {obj} not been {oa}, it would have gone in; the {cont} was not the problem.",
                           f"If the {obj} had not been {oa}, it would have worked. The {cont} was fine.",
                           f"A different {obj}, one that was not {oa}, would have solved it."])
    return f"{lead} {tail}"


def _dc_sent(rng, dim, a, b):
    more, less = COMPARATIVE_ADJ[dim]
    return rng.choice([f"The {a} was {more} than the {b}.", f"The {b} was {less} than the {a}.",
                       f"Compared with the {a}, the {b} was {less}.", f"Compared with the {b}, the {a} was {more}.",
                       f"Of these, the {a} was {more} and the {b} was {less}."])


def _dc_q(rng, dim, a, b):
    more, less = COMPARATIVE_ADJ[dim]
    return rng.choice([f"Which was {more}? The {a}.", f"Which was {less}? The {b}.", f"Is the {b} {more} than the {a}? No.",
                       f"Is the {a} {less} than the {b}? No, the {a} is {more}.", f"So the {b} was {less}, not {more}."])


def direct_comparative_doc(rng, train_pool, names):
    comps = [sp for f, sp in train_pool if f == "comparative"]
    items = []
    for _ in range(rng.randint(1, 3)):
        dim, pairs, *_ = rng.choice(comps)
        a, b = rng.choice(pairs)
        items.append((dim, a, b))
    sents = [_dc_sent(rng, *it) for it in items] + [_dc_q(rng, *rng.choice(items))]
    return _frame(rng, names, sents)


# ---------- new families (combinatorial: several facts per document, framing, questions) ----------
SETTINGS = ["In the science class,", "At the museum,", "During the quiz,", "On the field trip,", "In the workshop,", "At dinner,",
            "In the library,", "On the radio programme,", "In the textbook,", "During the lesson,"]


def _frame(rng, names, sents):
    r = rng.random()
    if r < 0.3:
        x, y = rng.sample(names, 2)
        return f"{x} explained to {y}: " + " ".join(sents)
    if r < 0.5:
        return f"{rng.choice(SETTINGS)} {rng.choice(names)} noted the following. " + " ".join(sents)
    if r < 0.6:
        return v2.filler(rng) + " " + " ".join(sents)
    return " ".join(sents)


def _pw_sent(rng, p, w):
    return rng.choice([f"The {p} is part of the {w}.", f"A {w} has a {p}.", f"Every {w} contains a {p}.",
                       f"The {p} belongs to the {w}, not the other way around.", f"A {p} is one piece of a {w}.",
                       f"You can find a {p} in a {w}.", f"The {w} includes the {p}.", f"The {p} is a component of the {w}."])


def _pw_q(rng, p, w):
    return rng.choice([f"Which is the whole, the {p} or the {w}? The {w}.", f"Which is the part, the {w} or the {p}? The {p}.",
                       f"Q: What is the {p} part of? A: The {w}.", f"Is the {w} part of the {p}? No. The {p} is part of the {w}.",
                       f"So the {w} is the larger whole and the {p} is the piece.", f"Does a {p} contain a {w}? No; a {w} contains a {p}."])


def part_whole_doc(rng, pairs, names):
    ps = rng.sample(pairs, rng.randint(1, 3))
    sents = [_pw_sent(rng, p, w) for p, w in ps]
    sents.append(_pw_q(rng, *rng.choice(ps)))
    return _frame(rng, names, sents)


def _ce_sent(rng, c, e):
    return rng.choice([f"The {c} caused the {e}.", f"The {e} happened because of the {c}.", f"The {c} led to the {e}.",
                       f"First came the {c}, and then, as a result, the {e}.", f"The {e} was a result of the {c}.",
                       f"Because of the {c}, there was a {e}.", f"The {c} brought about the {e}.", f"Without the {c}, there would have been no {e}."])


def _ce_q(rng, c, e):
    return rng.choice([f"What was the cause of the {e}? The {c}.", f"What did the {c} lead to? The {e}.",
                       f"Did the {e} cause the {c}? No. The {c} caused the {e}.", f"Which came first, the {e} or the {c}? The {c}.",
                       f"Q: Cause or effect, the {e}? A: The effect.", f"Q: Cause or effect, the {c}? A: The cause."])


def cause_effect_doc(rng, pairs, names):
    ps = rng.sample(pairs, rng.randint(1, 3))
    sents = [_ce_sent(rng, c, e) for c, e in ps]
    sents.append(_ce_q(rng, *rng.choice(ps)))
    return _frame(rng, names, sents)


def temporal_doc(rng, names):
    x, y = rng.sample(names, 2)
    ev = rng.choice(TEMPORAL_EVENTS)
    fmt = rng.randrange(4)
    if fmt == 0:
        return f"{x} {ev} before {y}. Who was first? {x}. Who was later? {y}."
    if fmt == 1:
        return f"{y} {ev} after {x}. So {x} {ev} first."
    if fmt == 2:
        return f"By the time {y} {ev}, {x} had already done so."
    return f"{x} {ev} first, and {y} {ev} some time later. Q: Did {y} come before {x}? A: No, {x} came first."


def document(rng, train_pool, names, pw_pairs, ce_pairs):
    """Returns (text, family, structure)."""
    r = rng.random()
    if r < 0.35:  # all of v2's document shapes
        return v2.document(rng, train_pool, names), "v2_mix", "v2_shapes"
    if r < 0.45:
        return part_whole_doc(rng, pw_pairs, names), "part_whole", "part_whole"
    if r < 0.55:
        return cause_effect_doc(rng, ce_pairs, names), "cause_effect", "cause_effect"
    if r < 0.62:
        return temporal_doc(rng, names), "temporal", "temporal"
    fam, spec = rng.choice(train_pool)
    if fam == "comparative" and rng.random() < 0.35:
        return direct_comparative_doc(rng, train_pool, names), fam, "direct_comparative"
    s = v2.make_sit(rng, fam, spec, names)
    w = rng.choice([1, 2])
    for name, fn in rng.sample([("reported", lambda: reported(rng, s, w)), ("whereas", lambda: whereas(rng, s, w)),
                                ("negation", lambda: negation(rng, s, w)), ("both_only", lambda: both_only(rng, s, w)),
                                ("role", lambda: role_statement(rng, s)), ("counterfactual", lambda: counterfactual(rng, s, spec, names))], 6):
        text = fn()
        if text:
            return text, fam, name
    return v2.statement(rng, s, w), fam, "statement"


# ---------- probe v5 (held-out structures, training vocabulary) ----------
def build_probe_v5(rng, train_pool, names, n_per=40):
    people = [(f, sp) for f, sp in train_pool if f in ("emotional", "ability") and sp[0] in PASSIVE]
    cases, seen = [], set()

    def add(cat, ctxA, ctxB, c1, c2):
        if ctxA not in seen:
            seen.add(ctxA)
            cases.append({"name": f"v5-{cat}-{len(cases)}", "category": cat, "structure": cat,
                          "ctxA": ctxA, "ctxB": ctxB, "c1": c1, "c2": c2})

    while sum(c["category"] == "passive" for c in cases) < n_per:
        fam, (verb, obj_r, subj_r) = rng.choice(people)
        x, y = rng.sample(names, 2)
        a1, a2 = rng.choice(obj_r), rng.choice(subj_r)
        ev = f"{y} was {PASSIVE[verb]} by {x}"
        add("passive", f"{ev} because {y} was", f"{ev} because {x} was", f" {a1}.", f" {a2}.")
    while sum(c["category"] == "relative_clause" for c in cases) < n_per:
        fam, (verb, obj_r, subj_r) = rng.choice(people)
        x, y = rng.sample(names, 2)
        a1, a2 = rng.choice(obj_r), rng.choice(subj_r)
        add("relative_clause", f"{x} {verb} {y}, who was", f"{y} was {PASSIVE[verb]} by {x}, who was", f" {a1}", f" {a2}")
    while sum(c["category"] == "respectively" for c in cases) < n_per:
        fam, spec = rng.choice(train_pool)
        if fam in ("comparative", "fit") or rng.random() < 0.5:
            fam, spec = rng.choice([p for p in train_pool if p[0] in ("comparative", "fit")])
        s = v2.make_sit(rng, fam, spec, names)
        e1, e2 = (s.e1, s.e2)
        add("respectively", f"{s.event}. {_cap(e1)} and {e2} were, respectively,", f"{s.event}. {_cap(e2)} and {e1} were, respectively,",
            f" {s.a1} and {s.a2}.", f" {s.a2} and {s.a1}.")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parquet_dir", default=None)
    ap.add_argument("--n_docs", type=int, default=9_000_000)
    ap.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--probe_files", nargs="*", default=[])
    ap.add_argument("--probe_out", default=None)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    probe_ctxs = v2.load_probe_contexts(args.probe_files)
    probe_text = " " + " ".join(probe_ctxs) + " "
    names = [n for n in v2.NAMES if f" {n.lower()} " not in probe_text and f" {n.lower()}." not in probe_text]
    train_names, _held_names = v2.split_holdout(names, every=6)  # same split as v2: probe v4 names stay out
    train_pool = []
    for dim, pairs, events, mores, lesses in v2.COMPARATIVE:
        tr, _ = v2.split_holdout(pairs)
        train_pool.append(("comparative", (dim, tr, events, mores, lesses)))
    tr, _ = v2.split_holdout(v2.FIT)
    train_pool += [("fit", f) for f in tr]
    for fam, frames in (("emotional", v2.EMOTIONAL), ("possession", v2.POSSESSION), ("ability", v2.ABILITY)):
        tr, _ = v2.split_holdout(frames)
        train_pool += [(fam, f) for f in tr]
    pw = [p for p in PART_WHOLE if not any(f" {p[0]} " in f" {c} " and f" {p[1]} " in f" {c} " for c in probe_ctxs)]
    ce = [p for p in CAUSE_EFFECT if not any(f" {p[0]} " in f" {c} " and f" {p[1]} " in f" {c} " for c in probe_ctxs)]
    print(f"[guard] names {len(train_names)} train; part-whole {len(pw)}/{len(PART_WHOLE)}, cause-effect {len(ce)}/{len(CAUSE_EFFECT)} pairs kept")

    def check(doc):
        low = doc.lower()
        bad = [b for b in BANNED_V3 if b in low]
        assert not bad and not PASSIVE_RE.search(low), f"held-out/banned structure {bad or 'passive'} in: {doc}"

    rng = random.Random(args.seed)
    if args.dry_run:
        for _ in range(args.dry_run):
            t, f, st = document(rng, train_pool, train_names, pw, ce)
            check(t)
            print(f"- [{f}/{st}] {t}")
        for c in build_probe_v5(random.Random(args.seed + 1), train_pool, train_names)[::15]:
            print("  v5", c["category"], "|", c["ctxA"], "->", c["c1"], "||", c["ctxB"], "->", c["c2"])
        return

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    eot = tok.eos_token_id
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq_rows = {"text": [], "family": [], "structure": []}
    pq_part = 0
    if args.parquet_dir:
        import pyarrow as pa
        import pyarrow.parquet as pq
        Path(args.parquet_dir).mkdir(parents=True, exist_ok=True)
    buf = np.empty(args.shard_tokens, dtype=np.uint16)
    pos, shard, total, unique, batch = 0, 0, 0, set(), []
    counts = {}

    def flush():
        nonlocal pos, shard, total
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids = ids + [eot]
            if pos + len(ids) > args.shard_tokens:
                buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1; pos = 0
            buf[pos:pos + len(ids)] = ids
            pos += len(ids); total += len(ids)
        batch.clear()

    def flush_pq(final=False):
        nonlocal pq_part
        if args.parquet_dir and (len(pq_rows["text"]) >= 500_000 or (final and pq_rows["text"])):
            pq.write_table(pa.table({**pq_rows, "version": ["v3"] * len(pq_rows["text"])}),
                           f"{args.parquet_dir}/train-{pq_part:05d}.parquet", compression="zstd")
            pq_part += 1
            for k in pq_rows:
                pq_rows[k].clear()

    skipped_v2 = 0
    for _ in range(args.n_docs):
        doc, fam, st = document(rng, train_pool, train_names, pw, ce)
        if doc in unique:
            continue
        if st == "v2_shapes":
            # some v2 shapes share a surface form with v3's held-out structures (e.g. the cleft
            # "It was X, not Y, who was A" contains ", who was"): skip them to keep probe v5 clean
            try:
                check(doc)
            except AssertionError:
                skipped_v2 += 1
                continue
        else:
            check(doc)  # v3's own structures must never produce a held-out structure
        unique.add(doc)
        batch.append(doc)
        counts[st] = counts.get(st, 0) + 1
        pq_rows["text"].append(doc); pq_rows["family"].append(fam); pq_rows["structure"].append(st)
        if len(batch) >= 20_000:
            flush()
        flush_pq()
    if batch:
        flush()
    flush_pq(final=True)
    if pos:
        buf[:pos].tofile(out / f"shard_{shard:05d}.bin"); shard += 1

    meta = {"dataset": "synthetic_contrastive_binding_v3", "tokenizer_path": args.tokenizer_path, "total_tokens": total,
            "n_docs": args.n_docs, "unique_docs": len(unique), "n_shards": shard, "dtype": "uint16", "eot_token": eot,
            "real_vocab_size": len(tok), "is_code": False, "fim_rate": 0.0, "probe_exclusion_files": args.probe_files,
            "heldout_structures": ["passive", "relative_clause", "respectively"], "docs_by_structure": counts,
            "skipped_v2_docs_with_heldout_surface": skipped_v2}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    if args.probe_out:
        cases = build_probe_v5(random.Random(args.seed + 1), train_pool, train_names)
        Path(args.probe_out).write_text(json.dumps({
            "description": "Probe v5: entity-attribute binding scored by log-likelihood, built from sentence STRUCTURES held "
                           "out of the contrastive v3 corpus (passive voice, relative clauses, 'respectively') but with "
                           "training names and entities. Measures structural transfer of binding.",
            "categories": sorted({c["category"] for c in cases}), "cases": cases}, indent=2))
        print(f"[probe v5] wrote {len(cases)} cases to {args.probe_out}")


if __name__ == "__main__":
    main()
