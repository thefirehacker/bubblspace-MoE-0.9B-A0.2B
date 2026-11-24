"""Render knowledge-graph and NLI/commonsense sources into plain-text training documents (Phase 2.5a relations share).

Sources (downloaded and ClamAV-scanned under --raw first; no remote code is executed):
  kg_text pool
    conceptnet   conceptnet5/conceptnet5 (CC BY-SA 4.0): English edges for ~25 relations (IsA, PartOf, UsedFor,
                 CapableOf, Causes, HasProperty, AtLocation, MadeOf, Antonym, ...), grouped into concept-centric
                 paragraphs with several phrasings per relation and a direction-checking question
                 ("Is a wheel part of a car? Yes. Is a car part of a wheel? No.").
    wikidata5m   intfloat/wikidata5m (Wikidata, CC0): transductive triples verbalized with relation-specific templates.
                 NOT USED for Phase 2.5a: the alias lists are noisy ("Huamn", "CAT:CROATIA", "Franzoesische Republik"),
                 so entity names come out misspelled; full Wikipedia / fact_seeking / SYNTH cover these facts in clean prose.
  nli_commonsense pool
    esnli        e-SNLI train (premise, hypothesis, label, human explanation), from HF's parquet conversion.
    mnli         MultiNLI train.
    csqa         CommonsenseQA train (question, 5 options, answer).
    atomic       ATOMIC-2020 (CC BY 4.0) if-then commonsense; PersonX/PersonY replaced by varied names.

Each document is one JSONL row {"text", "source"}. Tokenize afterwards with src/data/prepare.py (--dataset json).

    python scripts/render_relational_sources.py --raw /data/raw/relsrc --out /data/raw/relational_text --sources all
"""

import argparse
import glob
import json
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

NAMES = ("Aarav Abigail Adaeze Aiko Alejandro Amara Amir Ana Anders Anika Arjun Astrid Ayesha Bao Beatriz Bilal Camila Chen "
         "Chloe Dante Darius Deepa Diego Dmitri Elena Elif Emeka Esther Fatima Felix Freya Gabriel Hana Hassan Helena Hiro "
         "Ibrahim Ines Isaac Ivan Jamal Jana Javier Jin Jonas Kai Kamala Karim Kavya Keiko Kenji Kofi Lars Layla Leo Lin Lucia "
         "Luis Maya Mehmet Mei Mira Nadia Nikhil Nina Noor Omar Oscar Pablo Paula Pedro Rafael Ravi Rosa Rohan Ruth Sakura "
         "Samir Sana Santiago Sara Selin Shreya Simone Sofia Sunil Tariq Tess Thabo Tomas Uma Vikram Wanjiru Yara Yusuf Zainab").split()


class Writer:
    def __init__(self, out, source, per_file=200_000):
        self.dir = Path(out) / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.source, self.per_file, self.n, self.part, self.f = source, per_file, 0, 0, None

    def write(self, text):
        if self.f is None or self.n % self.per_file == 0:
            if self.f:
                self.f.close()
            self.f = open(self.dir / f"part-{self.part:05d}.jsonl", "w")
            self.part += 1
        self.f.write(json.dumps({"text": text, "source": self.source}, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        if self.f:
            self.f.close()
        print(f"[{self.source}] {self.n:,} documents -> {self.dir}", flush=True)


# ---------------- ConceptNet ----------------
CN_TEMPLATES = {
    "IsA": (["{A} is a kind of {B}.", "{A} is a type of {B}.", "One kind of {B} is {A}."],
            "Is {B} a kind of {A}? No; {A} is a kind of {B}."),
    "PartOf": (["{A} is part of {B}.", "{B} has {A} as one of its parts.", "You can find {A} as a part of {B}."],
               "Is {B} part of {A}? No; {A} is part of {B}."),
    "HasA": (["{A} has {B}.", "{B} is something that {A} has."], "Does {B} have {A}? Not necessarily; {A} has {B}."),
    "UsedFor": (["{A} is used for {B}.", "People use {A} for {B}.", "One use of {A} is {B}."], "What is {A} used for? {B}."),
    "CapableOf": (["{A} can {B}.", "{A} is able to {B}."], "What can {A} do? It can {B}."),
    "NotCapableOf": (["{A} cannot {B}.", "{A} is not able to {B}."], "Can {A} {B}? No."),
    "Causes": (["{A} can cause {B}.", "{A} often leads to {B}.", "{B} can result from {A}."],
               "Does {B} cause {A}? No; {A} can cause {B}."),
    "HasProperty": (["{A} is {B}.", "{A} is usually {B}.", "A property of {A} is that it is {B}."], "What is {A} like? {B}."),
    "AtLocation": (["You are likely to find {A} in {B}.", "{A} can be found at {B}.", "{A} is often in {B}."],
                   "Where might you find {A}? In {B}."),
    "MadeOf": (["{A} is made of {B}.", "{A} is made from {B}."], "Is {B} made of {A}? No; {A} is made of {B}."),
    "HasPrerequisite": (["To {A}, you first need {B}.", "{A} requires {B}."], "What do you need before {A}? {B}."),
    "HasSubevent": (["Part of {A} is {B}.", "When you {A}, you may {B}."], "What happens during {A}? {B}."),
    "HasFirstSubevent": (["The first step of {A} is {B}."], "What comes first in {A}? {B}."),
    "HasLastSubevent": (["The last step of {A} is {B}."], "What comes last in {A}? {B}."),
    "MotivatedByGoal": (["Someone might {A} because they want {B}.", "A reason to {A} is {B}."], "Why would someone {A}? For {B}."),
    "Desires": (["{A} wants {B}.", "{A} likes {B}."], "What does {A} want? {B}."),
    "ReceivesAction": (["{A} can be {B}."], "What can be done to {A}? It can be {B}."),
    "CreatedBy": (["{A} is made by {B}.", "{B} creates {A}."], "Who or what makes {A}? {B}."),
    "Antonym": (["{A} is the opposite of {B}.", "The opposite of {A} is {B}."], "What is the opposite of {A}? {B}."),
    "DistinctFrom": (["{A} is not the same as {B}.", "{A} and {B} are different things."], "Is {A} the same as {B}? No."),
    "SymbolOf": (["{A} is a symbol of {B}.", "{A} stands for {B}."], "What does {A} symbolize? {B}."),
    "LocatedNear": (["{A} is often found near {B}.", "{A} is usually close to {B}."], "What is {A} usually near? {B}."),
    "DefinedAs": (["{A} is defined as {B}.", "{A} means {B}."], "What is {A}? {B}."),
    "Entails": (["{A} entails {B}.", "If you {A}, you also {B}."], "What does {A} involve? {B}."),
    "ObstructedBy": (["{A} can be prevented by {B}."], "What can stop {A}? {B}."),
    "CausesDesire": (["{A} makes you want to {B}."], "What does {A} make you want? To {B}."),
}


def cn_concept(uri):
    parts = uri.split("/")
    return parts[3].replace("_", " ") if len(parts) > 3 else None


def render_conceptnet(raw, out, rng, min_weight=1.0, max_edges=8):
    import pyarrow.parquet as pq
    by_head = defaultdict(list)
    files = sorted(glob.glob(f"{raw}/conceptnet5_conceptnet5/conceptnet5/*.parquet"))
    kept = 0
    for f in files:
        t = pq.read_table(f, columns=["rel", "arg1", "arg2", "lang", "weight"]).to_pydict()
        for rel, a1, a2, lang, w in zip(t["rel"], t["arg1"], t["arg2"], t["lang"], t["weight"]):
            r = rel.rsplit("/", 1)[-1]
            if lang != "en" or r not in CN_TEMPLATES or (w or 0) < min_weight:
                continue
            if not (a1.startswith("/c/en/") and a2.startswith("/c/en/")):
                continue
            A, B = cn_concept(a1), cn_concept(a2)
            if A and B and A != B:
                by_head[A].append((r, B))
                kept += 1
    print(f"[conceptnet] {kept:,} English edges, {len(by_head):,} head concepts", flush=True)
    w = Writer(out, "conceptnet")
    for A, edges in by_head.items():
        for rep in range(2 if len(edges) >= 3 else 1):   # two differently phrased documents for richer concepts
            sel = rng.sample(edges, min(len(edges), max_edges))
            sents = []
            for r, B in sel:
                sents.append(rng.choice(CN_TEMPLATES[r][0]).format(A=A, B=B))
            r, B = rng.choice(sel)
            sents.append(CN_TEMPLATES[r][1].format(A=A, B=B))
            text = " ".join(s[0].upper() + s[1:] for s in sents)
            w.write(text)
    w.close()


# ---------------- Wikidata5M ----------------
WD = {
    "P31": "{h} is an instance of {t}.", "P279": "{h} is a type of {t}.", "P17": "{h} is in the country of {t}.",
    "P27": "{h} is a citizen of {t}.", "P19": "{h} was born in {t}.", "P20": "{h} died in {t}.", "P106": "{h} worked as a {t}.",
    "P131": "{h} is located in {t}.", "P150": "{h} contains {t}.", "P47": "{h} shares a border with {t}.",
    "P36": "The capital of {h} is {t}.", "P1376": "{h} is the capital of {t}.", "P50": "{h} was written by {t}.",
    "P57": "{h} was directed by {t}.", "P161": "{t} was a cast member of {h}.", "P69": "{h} studied at {t}.",
    "P108": "{h} worked for {t}.", "P54": "{h} played for {t}.", "P136": "The genre of {h} is {t}.",
    "P495": "{h} originated in {t}.", "P361": "{h} is part of {t}.", "P527": "{h} includes {t}.", "P26": "{h} was married to {t}.",
    "P22": "The father of {h} was {t}.", "P25": "The mother of {h} was {t}.", "P40": "{t} is a child of {h}.",
    "P463": "{h} is a member of {t}.", "P102": "{h} is a member of the {t}.", "P166": "{h} received the {t}.",
    "P175": "{h} was performed by {t}.", "P264": "{h} was released on the label {t}.", "P407": "{h} is in the {t} language.",
    "P1412": "{h} spoke {t}.", "P30": "{h} is on the continent of {t}.", "P159": "{h} has its headquarters in {t}.",
    "P118": "{h} plays in the {t}.", "P641": "{h} is associated with the sport of {t}.", "P1303": "{h} played the {t}.",
    "P140": "The religion of {h} is {t}.", "P39": "{h} held the position of {t}.", "P1344": "{h} took part in {t}.",
    "P276": "{h} took place in {t}.", "P706": "{h} is located on {t}.", "P206": "{h} is next to {t}.", "P37": "The official language of {h} is {t}.",
    "P38": "The currency of {h} is {t}.", "P35": "The head of state of {h} is {t}.", "P6": "The head of government of {h} is {t}.",
    "P112": "{h} was founded by {t}.", "P127": "{h} is owned by {t}.", "P176": "{h} was manufactured by {t}.", "P86": "The music of {h} was composed by {t}.",
    "P58": "{h} was written by the screenwriter {t}.", "P737": "{h} was influenced by {t}.", "P1050": "{h} had the medical condition {t}.",
    "P101": "{h} worked in the field of {t}.", "P135": "{h} belongs to the movement {t}.", "P413": "{h} played as a {t}.",
    "P421": "{h} is in the {t} time zone.", "P1001": "{h} applies to {t}.", "P364": "The original language of {h} is {t}.",
    "P449": "{h} was first broadcast on {t}.", "P272": "{h} was produced by {t}.", "P1441": "{h} appears in {t}.",
    "P123": "{h} was published by {t}.", "P840": "{h} is set in {t}.", "P915": "{h} was filmed in {t}.", "P179": "{h} is part of the series {t}.",
    "P155": "{h} came after {t}.", "P156": "{h} was followed by {t}.", "P171": "{h} belongs to the parent taxon {t}.",
    "P105": "The taxon rank of {h} is {t}.", "P197": "{h} is next to the station {t}.", "P81": "{h} is on the {t} line.",
    "P1365": "{h} replaced {t}.", "P1366": "{h} was replaced by {t}.", "P551": "{h} lived in {t}.", "P937": "{h} worked in {t}.",
    "P3373": "{h} is a sibling of {t}.", "P1411": "{h} was nominated for the {t}.", "P1027": "{h} was conferred by {t}.",
    "P749": "The parent organization of {h} is {t}.", "P355": "{h} has the subsidiary {t}.", "P452": "{h} works in the {t} industry.",
}


def render_wikidata5m(raw, out, rng, max_facts=10):
    d = f"{raw}/intfloat_wikidata5m"
    ent, rel = {}, {}
    with tarfile.open(f"{d}/wikidata5m_alias.tar.gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            target = ent if "entity" in m.name else rel if "relation" in m.name else None
            if target is None:
                continue
            for line in tf.extractfile(m):
                parts = line.decode("utf-8", "replace").rstrip("\n").split("\t")
                if len(parts) >= 2:
                    aliases = [a.strip().rstrip(".") for a in parts[1:] if a.strip().rstrip(".")]
                    # prefer a capitalized alias (aliases are often lowercased variants)
                    cap = [a for a in aliases if a[:1].isupper()]
                    if aliases:
                        target[parts[0]] = (cap or aliases)[0]
    print(f"[wikidata5m] {len(ent):,} entity names, {len(rel):,} relation names", flush=True)
    by_head = defaultdict(list)
    with tarfile.open(f"{d}/wikidata5m_transductive.tar.gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or "train" not in m.name:
                continue
            for line in tf.extractfile(m):
                h, r, t = line.decode("utf-8", "replace").rstrip("\n").split("\t")
                if h in ent and t in ent and r in WD:  # only relations with a hand-written template
                    by_head[h].append((r, t))
    print(f"[wikidata5m] {sum(map(len, by_head.values())):,} triples, {len(by_head):,} head entities", flush=True)
    w = Writer(out, "wikidata5m")
    for h, facts in by_head.items():
        hn = ent[h]
        sel = rng.sample(facts, min(len(facts), max_facts))
        sents = [WD[r].format(h=hn, t=ent[t]) for r, t in sel]
        r, t = rng.choice(sel)
        cloze = WD[r].format(h=hn, t="___").rstrip(".")
        sents.append(f"Q: {cloze}? A: {ent[t]}.")
        w.write(" ".join(s[0].upper() + s[1:] for s in sents))
    w.close()


# ---------------- NLI / commonsense ----------------
NLI_LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}


def nli_text(rng, p, h, label, expl=None):
    lab = NLI_LABEL.get(label)
    if lab is None:
        return None
    p, h = p.strip(), h.strip()
    e = f" {expl.strip()}" if expl else ""
    fmt = rng.randrange(3)
    if fmt == 0:
        return f"Premise: {p}\nHypothesis: {h}\nRelation: {lab}.{(' Explanation:' + e) if e else ''}"
    if lab == "entailment":
        return rng.choice([f"{p} So it is true that {h[0].lower() + h[1:]}{e}", f"If \"{p}\", then \"{h}\" follows.{e}"])
    if lab == "contradiction":
        return rng.choice([f"{p} So it cannot be that {h[0].lower() + h[1:]}{e}", f"\"{p}\" rules out \"{h}\".{e}"])
    return rng.choice([f"{p} From this alone, we cannot tell whether {h[0].lower() + h[1:]}{e}",
                       f"\"{p}\" neither confirms nor rules out \"{h}\".{e}"])


def render_nli(raw, out, rng):
    import pyarrow.parquet as pq
    w = Writer(out, "esnli")
    for f in sorted(glob.glob(f"{raw}/esnli_esnli/**/train/*.parquet", recursive=True)):
        t = pq.read_table(f).to_pydict()
        for p, h, l, e in zip(t["premise"], t["hypothesis"], t["label"], t["explanation_1"]):
            x = nli_text(rng, p, h, l, e)
            if x:
                w.write(x)
    w.close()
    w = Writer(out, "mnli")
    for f in sorted(glob.glob(f"{raw}/nyu-mll_multi_nli/data/train-*.parquet")):
        t = pq.read_table(f).to_pydict()
        for p, h, l in zip(t["premise"], t["hypothesis"], t["label"]):
            x = nli_text(rng, p, h, l)
            if x:
                w.write(x)
    w.close()
    w = Writer(out, "csqa")
    for f in sorted(glob.glob(f"{raw}/tau_commonsense_qa/data/train-*.parquet")):
        t = pq.read_table(f).to_pylist()
        for ex in t:
            opts = " ".join(f"{l}. {x}" for l, x in zip(ex["choices"]["label"], ex["choices"]["text"]))
            ans = dict(zip(ex["choices"]["label"], ex["choices"]["text"])).get(ex["answerKey"])
            if ans:
                w.write(f"Question: {ex['question']}\n{opts}\nAnswer: {ex['answerKey']}. {ans}")
    w.close()


# ATOMIC tails are fragments ("to be abusive", "they are arrested"), so each relation is rendered as Q&A
ATOMIC_REL = {
    "xIntent": "Why did {X} do this?", "xNeed": "What did {X} need to do first?", "xAttr": "How would you describe {X}?",
    "xEffect": "What happens to {X} as a result?", "xWant": "What does {X} want to do next?", "xReact": "How does {X} feel?",
    "oEffect": "What happens to {Y} as a result?", "oReact": "How does {Y} feel?", "oWant": "What does {Y} want to do next?",
    "isAfter": "What happened before this?", "isBefore": "What happens after this?", "HinderedBy": "What could stop this from happening?",
    "Causes": "What does this cause?", "xReason": "What is the reason for this?",
}


def _atomic_tail(t, X, Y):
    t = re.sub(r"\bperson ?x\b", X, re.sub(r"\bperson ?y\b", Y, t.strip(), flags=re.I), flags=re.I)
    t = t[0].upper() + t[1:] if t else t
    return t if t.endswith((".", "!", "?")) else t + "."


def render_atomic(raw, out, rng):
    import pyarrow.parquet as pq
    events = defaultdict(list)
    for f in sorted(glob.glob(f"{raw}/Estwld_atomic2020-origin/data/train-*.parquet")):
        for ex in pq.read_table(f).to_pylist():
            tails = [t for t in (ex["tail"] or []) if t and t.strip().lower() != "none"]
            if tails and ex["relation"] in ATOMIC_REL:
                events[ex["event"]].append((ex["relation"], tails))
    w = Writer(out, "atomic")
    for ev, rels in events.items():
        X, Y = rng.sample(NAMES, 2)
        head = ev.replace("PersonX", X).replace("PersonY", Y).replace("___", "something")
        sents = [head[0].upper() + head[1:] + "."]
        for r, tails in rng.sample(rels, min(len(rels), 6)):
            sents.append(f"{ATOMIC_REL[r].format(X=X, Y=Y)} {_atomic_tail(rng.choice(tails), X, Y)}")
        w.write(" ".join(sents))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="all", help="comma list: conceptnet,wikidata5m,nli,atomic or all")
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()
    src = {"conceptnet", "wikidata5m", "nli", "atomic"} if args.sources == "all" else set(args.sources.split(","))
    rng = random.Random(args.seed)
    if "conceptnet" in src:
        render_conceptnet(args.raw, args.out, rng)
    if "wikidata5m" in src:
        render_wikidata5m(args.raw, args.out, rng)
    if "nli" in src:
        render_nli(args.raw, args.out, rng)
    if "atomic" in src:
        render_atomic(args.raw, args.out, rng)


if __name__ == "__main__":
    main()
