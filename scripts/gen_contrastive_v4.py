"""Contrastive binding corpus v4 (TEST version) + probe v6: break the word-order rule.

Why (doc/binding-study-02-priors-and-word-order.md): in contrastive v2 / v3 every event is written doer-first
("X forgave Y", "The sofa would not fit in the lift"), and v3 even asserts that no passive sentence is generated,
to keep passive as a held-out structure. So "first-mentioned = doer" was true in 100% of training documents, and
the model learned it: on probe v5's passive pairs it gives the REVERSED answer in 39-40 of 40.

This script writes text documents (JSONL) for an A10 fine-tuning test with two arms from the same generator and seed:

  --arm control   exactly contrastive v3 (documents from scripts/gen_contrastive_v3.py)
  --arm fix       v3 with ORDER ALTERNATION: in half of the eligible situations the event is written receiver-first,
                    people (verbs with a passive form):  "Y was forgiven by X"
                    physical fit:                        "The lift could not take the sofa"
                  Everything else (frames, questions, corrections, vocabulary, held-out split) is unchanged, so
                  mention order no longer predicts who did what.

Still held out of BOTH arms: relative clauses, "respectively" (probe v5), v2's held-out frames, names and pairs (probe v4).
The fix arm also skips any document that contains a probe-v5 passive context word for word.

Probe v6 (structures neither arm trains on; training names):
  passive_heldout_frame  "Y was forgiven by X, given that Y was"            (passive + a frame held out of training)
  object_relative        "Y, whom X forgave, was" / "X, who forgave Y, was" (receiver mentioned first, no passive verb)
  fit_inverse            "The lift could not take the sofa; the trouble was that the lift was"

    python scripts/gen_contrastive_v4.py --arm fix --n_docs 300000 --out data/binding_test/fix.jsonl \
        --probe_files <v2.json> <v3.json> <v4.json> <v5.json> --probe_v5 <v5.json> --probe_out probe_v6.json
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

_spec = importlib.util.spec_from_file_location("v3", Path(__file__).with_name("gen_contrastive_v3.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2, PASSIVE, _cap = v3.v2, v3.PASSIVE, v3._cap

INVERSE_FIT = ["could not take", "could not accommodate", "had no room for"]
ALT_RATE = 0.5

_orig_make_sit = v2.make_sit
_orig_role = v3.role_statement


def make_sit_alt(rng, family, spec, names):
    """v2.make_sit, but half of the eligible situations are written receiver-first."""
    s = _orig_make_sit(rng, family, spec, names)
    if family == "fit" and rng.random() < ALT_RATE:
        obj, cont = spec[0], spec[1]
        s = v2.Sit(f"The {cont} {rng.choice(INVERSE_FIT)} the {obj}", s.e1, s.e2, s.a1, s.a2, False, 1, "fit")
        s.alt = "fit_inverse"
    elif s.people and spec[0] in PASSIVE and rng.random() < ALT_RATE:
        x, y = s.e2, s.e1  # make_sit: "x verb y", e1 = y (receiver), e2 = x (doer)
        s = v2.Sit(f"{y} was {PASSIVE[spec[0]]} by {x}", y, x, s.a1, s.a2, True, 1, family)
        s.alt, s.active = "passive", (x, spec[0], y)
    return s


def role_statement_alt(rng, s):
    """v3.role_statement parses "x verb y" out of the event; passive events need their own wording."""
    if getattr(s, "alt", None) != "passive":
        return _orig_role(rng, s)
    x, verb, y = s.active
    fmt = rng.randrange(3)
    if fmt == 0:
        return f"{s.event}. {_cap(x)} did the action; {y} was on the receiving end."
    if fmt == 1:
        return f"In \"{s.event}\", {x} is the doer and {y} is the receiver."
    return f"{s.event}. Q: Who {verb} whom? A: {x} {verb} {y}, not the reverse."


def build_pools(probe_files):
    """Same pools and held-out split as gen_contrastive_v3.main()."""
    probe_ctxs = v2.load_probe_contexts(probe_files)
    probe_text = " " + " ".join(probe_ctxs) + " "
    names = [n for n in v2.NAMES if f" {n.lower()} " not in probe_text and f" {n.lower()}." not in probe_text]
    train_names, _ = v2.split_holdout(names, every=6)
    pool = []
    for dim, pairs, events, mores, lesses in v2.COMPARATIVE:
        tr, _ = v2.split_holdout(pairs)
        pool.append(("comparative", (dim, tr, events, mores, lesses)))
    tr, _ = v2.split_holdout(v2.FIT)
    pool += [("fit", f) for f in tr]
    for fam, frames in (("emotional", v2.EMOTIONAL), ("possession", v2.POSSESSION), ("ability", v2.ABILITY)):
        tr, _ = v2.split_holdout(frames)
        pool += [(fam, f) for f in tr]
    pw = [p for p in v3.PART_WHOLE if not any(f" {p[0]} " in f" {c} " and f" {p[1]} " in f" {c} " for c in probe_ctxs)]
    ce = [p for p in v3.CAUSE_EFFECT if not any(f" {p[0]} " in f" {c} " and f" {p[1]} " in f" {c} " for c in probe_ctxs)]
    return pool, train_names, pw, ce


def build_probe_v6(rng, pool, names, n_per=40):
    people = [(f, sp) for f, sp in pool if f in ("emotional", "ability") and sp[0] in PASSIVE]
    fits = [sp for f, sp in pool if f == "fit"]
    cases, seen = [], set()

    def add(cat, ctxA, ctxB, c1, c2):
        if ctxA not in seen:
            seen.add(ctxA)
            cases.append({"name": f"v6-{cat}-{len(cases)}", "category": cat, "structure": cat,
                          "ctxA": ctxA, "ctxB": ctxB, "c1": c1, "c2": c2})

    def count(cat):
        return sum(c["category"] == cat for c in cases)

    while count("passive_heldout_frame") < n_per:
        _, (verb, obj_r, subj_r) = rng.choice(people)
        x, y = rng.sample(names, 2)
        ev, frame = f"{y} was {PASSIVE[verb]} by {x}", rng.choice(v2.HELDOUT_FRAMES)
        add("passive_heldout_frame", frame(ev, y), frame(ev, x), f" {rng.choice(obj_r)}.", f" {rng.choice(subj_r)}.")
    while count("object_relative") < n_per:
        _, (verb, obj_r, subj_r) = rng.choice(people)
        x, y = rng.sample(names, 2)
        add("object_relative", f"{y}, whom {x} {verb}, was", f"{x}, who {verb} {y}, was", f" {rng.choice(obj_r)}.", f" {rng.choice(subj_r)}.")
    while count("fit_inverse") < n_per:
        obj, cont, vp, ca, oa = rng.choice(fits)
        ev, frame = f"The {cont} {rng.choice(INVERSE_FIT)} the {obj}", rng.choice(v2.HELDOUT_FRAMES)
        add("fit_inverse", frame(ev, f"the {cont}"), frame(ev, f"the {obj}"), f" {ca}.", f" {oa}.")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["control", "fix"], required=True)
    ap.add_argument("--n_docs", type=int, default=300_000)
    ap.add_argument("--out", required=True, help="JSONL: text, family, structure, alt")
    ap.add_argument("--shard_dir", default=None, help="also write tokenized uint16 shards + meta.json here (training pool format)")
    ap.add_argument("--tokenizer_path", default="bubblspace-tokenizer")
    ap.add_argument("--shard_tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--probe_files", nargs="*", default=[], help="probes whose names/contexts stay out of training (v2-v5)")
    ap.add_argument("--probe_v5", default=None, help="probe v5 file: its passive contexts are excluded word for word")
    ap.add_argument("--probe_out", default=None, help="write probe v6 here")
    ap.add_argument("--dry_run", type=int, default=0)
    a = ap.parse_args()

    pool, names, pw, ce = build_pools(a.probe_files)
    v5_passive = []
    if a.probe_v5:
        v5_passive = [c[k] for c in json.loads(Path(a.probe_v5).read_text())["cases"] if c["category"] == "passive" for k in ("ctxA", "ctxB")]
    if a.arm == "fix":
        v2.make_sit = make_sit_alt          # v2.document and every v3 structure call v2.make_sit
        v3.role_statement = role_statement_alt

    def allowed(doc):
        low = doc.lower()
        if any(b in low for b in v3.BANNED_V3):       # held-out frames, relative clauses, "respectively"
            return False
        if a.arm == "control":
            return not v3.PASSIVE_RE.search(low)      # control = v3: no passive at all
        return not any(c in doc for c in v5_passive)

    rng = random.Random(a.seed)
    out = None if a.dry_run else open(a.out, "w")
    seen, counts, n, skipped = set(), {}, 0, 0
    target = a.dry_run or a.n_docs
    while n < target:
        text, fam, st = v3.document(rng, pool, names, pw, ce)
        if text in seen or not allowed(text):
            skipped += 1
            continue
        seen.add(text)
        alt = "passive" if v3.PASSIVE_RE.search(text.lower()) else ("fit_inverse" if any(f" {i} the " in text for i in INVERSE_FIT) else "none")
        counts[alt] = counts.get(alt, 0) + 1
        n += 1
        if a.dry_run:
            print(f"- [{fam}/{st}/{alt}] {text}")
        else:
            out.write(json.dumps({"text": text, "family": fam, "structure": st, "alt": alt}) + "\n")
    if out:
        out.close()
    summary = {"arm": a.arm, "docs": n, "skipped": skipped, "by_alternation": counts}
    print(json.dumps(summary))

    if a.shard_dir and not a.dry_run:
        # same shard format as gen_contrastive_v3.py / prepare.py: uint16 tokens, <|endoftext|> after every document
        import numpy as np
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_path)
        eot = tok.eos_token_id
        sd = Path(a.shard_dir)
        sd.mkdir(parents=True, exist_ok=True)
        buf = np.empty(a.shard_tokens, dtype=np.uint16)
        pos, shard, total, batch = 0, 0, 0, []

        def flush():
            nonlocal pos, shard, total
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                ids = ids + [eot]
                if pos + len(ids) > a.shard_tokens:
                    buf[:pos].tofile(sd / f"shard_{shard:05d}.bin"); shard += 1; pos = 0
                buf[pos:pos + len(ids)] = ids
                pos += len(ids); total += len(ids)
            batch.clear()

        for line in open(a.out):
            batch.append(json.loads(line)["text"])
            if len(batch) >= 20_000:
                flush()
        if batch:
            flush()
        if pos:
            buf[:pos].tofile(sd / f"shard_{shard:05d}.bin"); shard += 1
        meta = {"dataset": f"synthetic_contrastive_binding_v4_{a.arm}", "generator": "scripts/gen_contrastive_v4.py",
                "tokenizer_path": a.tokenizer_path, "total_tokens": total, "unique_docs": n, "n_shards": shard,
                "dtype": "uint16", "eot_token": eot, "real_vocab_size": len(tok), "is_code": False, "fim_rate": 0.0,
                "seed": a.seed, "probe_exclusion_files": a.probe_files, "probe_v5_contexts_excluded": bool(a.probe_v5),
                "heldout_structures": ["relative_clause", "respectively", "object_relative (probe v6)"],
                "order_alternation_rate": ALT_RATE if a.arm == "fix" else 0.0, "docs_by_alternation": counts}
        (sd / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("total_tokens", "unique_docs", "n_shards")}))

    if a.probe_out:
        cases = build_probe_v6(random.Random(a.seed + 1), pool, names)
        Path(a.probe_out).write_text(json.dumps({
            "description": "Probe v6: entity-attribute binding on structures that neither contrastive v3 nor the v4 test corpus "
                           "trains on: passive with a held-out frame, object relatives (receiver mentioned first, no passive "
                           "verb) and inverse physical fit with a held-out frame. Training names and entities.",
            "categories": sorted({c["category"] for c in cases}), "cases": cases}, indent=2))
        print(f"[probe v6] wrote {len(cases)} cases to {a.probe_out}")


if __name__ == "__main__":
    main()
