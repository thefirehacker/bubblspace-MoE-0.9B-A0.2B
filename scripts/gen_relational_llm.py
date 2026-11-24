"""Generate naturally phrased relational passages with an open LLM (vLLM), for Phase 2.5a.

Goal: teach semantic relations (which entity a property / role / cause belongs to) with far more
structural variety than template corpora. Contrastive v2 taught binding that transferred to new
words but not to new sentence structures (probe v3 flat), so every prompt combines a relation type x
format x reading level, and the model writes the text freely.

Two modes (v2 of this script, after a test showed factual errors in free generation; v3 defaults to
grounded only and drops the dialogue format, after test 3 showed staged-mistake dialogues confuse the
model and the check rejected ~88% of everyday scenes; test 4 dropped pronoun narratives and minimal
pairs, which forced fiction onto factual sources: 13% / 7% passed the check, vs 79% for Q&A):
  * grounded (default 70%): the prompt includes a real Wikipedia passage and the model may use only
    facts stated in it plus everyday knowledge (grounding idea as in PleIAs/SYNTH).
  * everyday (30%): invented people and everyday objects/scenes, so correctness = internal consistency.

Quality guard: the same model re-reads each output (with the source passage, when grounded), writes a
short critique and ends with "VERDICT: PASS|FAIL"; only PASS is kept. Outputs that copy any 15-word
sequence from the source are dropped (we want rephrasing, not copies).

Contamination guard: dropped if it shares any 6-word sequence with a probe context (v2/v3/v4 and the
facts probe), or mentions both entities of a v2/v3 probe pair.

Output: JSONL shards {"text", "mode", "relation", "format", "level", "topic", "source_title",
"source_url", "idx"}; resumable.

    python scripts/gen_relational_llm.py --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --wiki_dir /data/raw/wikipedia/20231101.en --out_dir /data/raw/relational_llm --n_prompts 900000
"""

import argparse
import glob
import json
import random
import re
import time
from pathlib import Path

RELATIONS = {
    "cause_effect": "what causes what, and what the effect is on which entity",
    "input_output": "what a process takes in versus what it produces (inputs vs outputs)",
    "part_whole": "which thing is a part of which whole, and what each part does",
    "comparison": "which of two entities has more or less of a property (size, speed, age, cost, strength...)",
    "temporal_order": "what happened before or after what, and why the order matters",
    "spatial": "where things are relative to each other (inside, above, behind, between...)",
    "agent_patient": "who did something to whom, and who was affected",
    "possession_transfer": "who owns, gives, receives, borrows or owes what",
    "social_roles": "relationships between people (teacher/student, parent/child, employer/employee...)",
    "category_membership": "what kind of thing something is, and how it differs from a similar category",
    "function_purpose": "what something is used for, and what a similar-looking thing is used for instead",
    "property_attribution": "which entity has which property when two entities are described together",
    "negation_exception": "what is NOT true, exceptions to a rule, and common misconceptions",
    "condition": "what happens under which condition (if/unless/only when)",
    "contrast_concession": "how two things differ despite seeming alike (although, whereas, despite)",
    "quantity_proportion": "how amounts relate (twice as many, a fraction of, more than half...)",
}

EVERYDAY_TOPICS = [
    "family life", "shopping at a market", "a school day", "an office workplace", "travelling by train", "cooking at home",
    "a neighbourhood cricket match", "friendship and arguments", "moving house", "household chores", "looking after pets",
    "a rainy day", "a village festival", "a wedding", "a hospital visit", "a bank and money", "a library", "a farm",
    "a bus journey", "a restaurant", "repairing a bicycle", "a birthday party", "a school sports day", "a garden",
    "an apartment building", "a small shop", "a job interview", "a road trip", "a camping trip", "a music lesson",
    "a football team", "a construction site", "a post office", "a classroom experiment", "a phone call", "a lost item",
]

FORMATS = {
    "explainer": "Write an explanatory paragraph or two about specific related entities from {subject}, focused on {rel}. "
                 "State clearly which property, role or effect belongs to which entity.",
    "confusion": "Write a short 'common confusion' explainer about two things from {subject} that people could mix up, "
                 "focused on {rel}. Say what each one really is, which way round the relation goes, and why the mix-up happens.",
    "qa": "Write 4-6 question-and-answer pairs about {subject}, each testing {rel}. Vary the question forms (which, who, why, "
          "what happens if, true or false). Each answer gives the answer and a one-sentence reason.",
    "process": "Describe a process or sequence of events from {subject} step by step, making {rel} explicit at each step "
               "(what goes in, what comes out, what acts on what).",
    "compare_prose": "Compare two or three related things from {subject} in flowing prose (no tables), focused on {rel}. Use "
                     "varied sentence structures: passive voice, relative clauses, 'the former/the latter', comparatives.",
}

LEVELS = ["a curious 12-year-old", "a high-school student", "a first-year university student", "a general adult reader"]

SYSTEM = ("You write clear, accurate educational text in English. Do not add a title, preamble or closing remarks; "
          "output only the requested text.")

GROUNDED_RULES = ("Use ONLY facts stated in the source passage, plus simple everyday knowledge. Do not add dates, numbers, "
                  "names or technical claims that are not in the passage. Do not copy sentences from the passage; rephrase. "
                  "If the passage does not support the requested relation, use another relation the passage clearly supports.")

# v4 (after EDA of shards 0-5): quantity/comparison/spatial passed only 14-37% in every domain because random Wikipedia
# passages rarely contain them, so the model invented numbers; ~15% of passages opened with a few stock phrases.
# Now the model CHOOSES one of 3 candidate relations that the source genuinely supports, and stock openings are banned.
CHOSEN_RULES = ("Use ONLY facts stated in the source passage, plus simple everyday knowledge. Do not add dates, numbers, "
                "names or technical claims that are not in the passage. Do not copy sentences from the passage; rephrase. "
                "Do not force a relation that is not genuinely present in the passage. Vary how you begin: do not start with "
                "'People often', 'People sometimes', 'A common confusion', 'One common confusion' or 'What happens if'.")
STOCK_OPENINGS = ("people often", "people sometimes", "a common confusion", "one common confusion", "what happens if")

EVERYDAY_RULES = ("Use invented people and ordinary objects and places. Do not rely on specialised facts; everything must "
                  "follow logically from what the text itself says.")

JUDGE_GROUNDED = ("You are a strict fact-checker. Compare the GENERATED TEXT with the SOURCE and with general knowledge. "
                  "Look for: any claim not supported by the source or common knowledge, wrong numbers, internal "
                  "contradictions, illogical events, or a property, role, cause or effect assigned to the wrong entity. "
                  "Write at most two short sentences of critique, then a final line exactly 'VERDICT: PASS' or 'VERDICT: FAIL'.")

JUDGE_CHOSEN = ("You are a strict fact-checker. Compare the GENERATED TEXT with the SOURCE and with general knowledge. "
                "Look for: any claim not supported by the source or common knowledge, wrong numbers, internal "
                "contradictions, illogical events, a property, role, cause or effect assigned to the wrong entity, or a "
                "relation ({rel}: {desc}) that is forced or contrived rather than genuinely present in the source. "
                "Write at most two short sentences of critique, then a final line exactly 'VERDICT: PASS' or 'VERDICT: FAIL'.")

JUDGE_EVERYDAY = ("You are a strict reviewer. Check the TEXT for internal contradictions, illogical or impossible events, "
                  "wrong arithmetic, pronouns that cannot be resolved as claimed, or a property, role, cause or effect "
                  "assigned to the wrong entity. Write at most two short sentences of critique, then a final line exactly "
                  "'VERDICT: PASS' or 'VERDICT: FAIL'.")

SKIP_SECTION = re.compile(r"^(see also|references|external links|further reading|notes|bibliography|sources)\b", re.I | re.M)


class WikiPassages:
    """Random 120-350-word passages of consecutive prose paragraphs from Wikipedia parquet files."""

    def __init__(self, wiki_dir, seed):
        import pyarrow.parquet as pq
        self.pq = pq
        self.files = sorted(glob.glob(f"{wiki_dir}/*.parquet"))
        assert self.files, f"no parquet in {wiki_dir}"
        self.r = random.Random(seed)

    def sample(self, n):
        out = []
        while len(out) < n:
            f = self.pq.ParquetFile(self.r.choice(self.files))
            tab = f.read_row_group(self.r.randrange(f.num_row_groups), columns=["title", "url", "text"]).to_pylist()
            self.r.shuffle(tab)
            for art in tab[: max(1, (n - len(out)) // 4 + 50)]:
                text = SKIP_SECTION.split(art["text"])[0]
                paras = [p.strip() for p in text.split("\n") if len(p.split()) >= 40 and "|" not in p]
                if not paras:
                    continue
                i = self.r.randrange(len(paras))
                chunk, words = [], 0
                while i < len(paras) and words < 120:
                    chunk.append(paras[i]); words += len(paras[i].split()); i += 1
                if 120 <= words <= 350:
                    out.append({"title": art["title"], "url": art["url"], "text": "\n\n".join(chunk)})
                if len(out) >= n:
                    break
        return out


def probe_guard(probe_paths, facts_path):
    grams, pairs = set(), []
    for p in probe_paths:
        for c in json.loads(Path(p).read_text())["cases"]:
            for k in ("ctxA", "ctxB"):
                w = re.findall(r"[a-z']+", c[k].lower())
                grams.update(" ".join(w[i:i + 6]) for i in range(len(w) - 5))
            if " / " in c.get("name", ""):
                a, b = [re.sub(r"\(.*?\)", "", x).strip().lower() for x in c["name"].split(" / ", 1)]
                pairs.append((a.split()[-1], b.split()[-1]))
    if facts_path:
        for it in json.loads(Path(facts_path).read_text())["items"]:
            w = re.findall(r"[a-z']+", it["q"].lower())
            grams.update(" ".join(w[i:i + 6]) for i in range(len(w) - 5))
    return grams, pairs


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def contaminated(text, grams, pairs):
    w = words(text)
    if any(" ".join(w[i:i + 6]) in grams for i in range(len(w) - 5)):
        return True
    ws = set(w)
    return any(a in ws and b in ws for a, b in pairs)


def copies_source(text, source, n=15):
    s = words(source)
    src = {" ".join(s[i:i + n]) for i in range(len(s) - n + 1)}
    w = words(text)
    return any(" ".join(w[i:i + n]) in src for i in range(len(w) - n + 1))


def build_meta(i, seed, grounded_frac):
    r = random.Random(seed * 1_000_003 + i)
    m = {"idx": i, "mode": "grounded" if r.random() < grounded_frac else "everyday",
         "relation": r.choice(list(RELATIONS)), "format": r.choice(list(FORMATS)), "level": r.choice(LEVELS),
         "topic": r.choice(EVERYDAY_TOPICS), "words": r.choice([150, 200, 250, 300])}
    others = [x for x in RELATIONS if x != m["relation"]]
    m["candidates"] = [m["relation"]] + r.sample(others, 2)
    return m


def user_prompt(m, src, chosen=False):
    rel = RELATIONS[m["relation"]]
    tail = f" Write for {m['level']}, about {m['words']} words."
    if chosen and m["mode"] == "grounded":
        cands = "\n".join(f"- {c}: {RELATIONS[c]}" for c in m["candidates"])
        task = FORMATS[m["format"]].format(subject="the subject of the source passage", rel="that relation")
        return (f"SOURCE (from the Wikipedia article \"{src['title']}\"):\n<<<\n{src['text']}\n>>>\n\n"
                f"CANDIDATE RELATIONS:\n{cands}\n\n"
                f"TASK: First choose the ONE candidate relation that the source most clearly and naturally supports, and "
                f"write it alone on the first line as 'RELATION: <name>'. Then, from the next line: {task} {CHOSEN_RULES}{tail}")
    if m["mode"] == "grounded":
        task = FORMATS[m["format"]].format(subject="the subject of the source passage", rel=rel)
        return (f"SOURCE (from the Wikipedia article \"{src['title']}\"):\n<<<\n{src['text']}\n>>>\n\n"
                f"TASK: {task} {GROUNDED_RULES}{tail}")
    return FORMATS[m["format"]].format(subject=m["topic"], rel=rel) + " " + EVERYDAY_RULES + tail


def judge_prompt(m, text, src, chosen=False):
    if chosen and m["mode"] == "grounded":
        j = JUDGE_CHOSEN.format(rel=m["relation"], desc=RELATIONS[m["relation"]])
        return f"{j}\n\nSOURCE:\n{src['text']}\n\nGENERATED TEXT:\n{text}"
    if m["mode"] == "grounded":
        return f"{JUDGE_GROUNDED}\n\nSOURCE:\n{src['text']}\n\nGENERATED TEXT:\n{text}"
    return f"{JUDGE_EVERYDAY}\n\nTEXT:\n{text}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--wiki_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_prompts", type=int, default=900_000)
    ap.add_argument("--chunk", type=int, default=20_000, help="prompts per shard")
    ap.add_argument("--grounded_frac", type=float, default=1.0,
                    help="share of grounded prompts (test 3: everyday scenes were mostly rejected, so default is grounded only)")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_num_seqs", type=int, default=512)
    ap.add_argument("--relation_mode", choices=["chosen", "requested"], default="chosen",
                    help="chosen (v4): model picks 1 of 3 candidate relations; requested: one relation is assigned (shards 0-5)")
    ap.add_argument("--probes", nargs="*", default=sorted(glob.glob("benchmark/data/probefinding-winograndestyle/*.json")))
    ap.add_argument("--facts", default="benchmark/data/facts-closedbook/facts-simple-v1-20261123.json")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams  # deferred: separate venv on the generation host

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grams, pairs = probe_guard(args.probes, args.facts)
    print(f"[guard] {len(grams):,} probe 6-grams, {len(pairs)} probe entity pairs", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, seed=args.seed, max_num_seqs=args.max_num_seqs)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
    judge_sp = SamplingParams(temperature=0.0, max_tokens=120)
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # noqa: E731

    done = {int(Path(p).stem.split("_")[1]) for p in glob.glob(str(out / "shard_*.jsonl"))}
    for start in range(0, args.n_prompts, args.chunk):
        sid = start // args.chunk
        if sid in done:
            continue
        t0 = time.time()
        meta = [build_meta(i, args.seed, args.grounded_frac) for i in range(start, min(start + args.chunk, args.n_prompts))]
        n_g = sum(m["mode"] == "grounded" for m in meta)
        srcs = iter(WikiPassages(args.wiki_dir, args.seed * 7919 + sid).sample(n_g))
        for m in meta:
            m["src"] = next(srcs) if m["mode"] == "grounded" else None
        chosen = args.relation_mode == "chosen"
        res = llm.generate([chat([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": user_prompt(m, m["src"], chosen)}]) for m in meta], sp, use_tqdm=False)
        cand, gen_tok = [], 0
        drop = {"truncated": 0, "short": 0, "probe": 0, "copied": 0, "no_relation": 0, "stock_opening": 0, "judge_fail": 0}
        for m, r in zip(meta, res):
            o = r.outputs[0]
            gen_tok += len(o.token_ids)
            text = o.text.strip()
            if chosen and m["mode"] == "grounded" and o.finish_reason == "stop":
                first, _, rest = text.partition("\n")
                g = re.match(r"\s*\**RELATION:\**\s*([a-z_]+)", first.strip(), re.I)
                if not g or g.group(1).lower() not in m["candidates"]:
                    drop["no_relation"] += 1
                    continue
                m["relation"], text = g.group(1).lower(), rest.strip()
                if text.lower().startswith(STOCK_OPENINGS):
                    drop["stock_opening"] += 1
                    continue
            if o.finish_reason != "stop":
                drop["truncated"] += 1
            elif len(text.split()) < 80:
                drop["short"] += 1
            elif contaminated(text, grams, pairs):
                drop["probe"] += 1
            elif m["src"] and copies_source(text, m["src"]["text"]):
                drop["copied"] += 1
            else:
                cand.append((m, text))
        jres = llm.generate([chat([{"role": "user", "content": judge_prompt(m, t, m["src"], chosen)}]) for m, t in cand],
                            judge_sp, use_tqdm=False)
        kept, rejects = [], []
        for (m, text), jr in zip(cand, jres):
            critique = jr.outputs[0].text.strip()
            v = re.findall(r"VERDICT:\s*(PASS|FAIL)", critique.upper())
            if v and v[-1] == "PASS":
                kept.append({"text": text, "mode": m["mode"], "relation": m["relation"], "format": m["format"],
                             "level": m["level"], "topic": None if m["src"] else m["topic"],
                             "source_title": m["src"]["title"] if m["src"] else None,
                             "source_url": m["src"]["url"] if m["src"] else None, "idx": m["idx"],
                             "relation_source": "chosen_from_3" if chosen else "requested",
                             "relation_candidates": m["candidates"] if chosen else None})
            else:
                drop["judge_fail"] += 1
                rejects.append({"text": text, "critique": critique, "mode": m["mode"], "relation": m["relation"],
                                "format": m["format"], "source_title": m["src"]["title"] if m["src"] else None,
                                "idx": m["idx"], "relation_source": "chosen_from_3" if chosen else "requested"})
        tmp = out / f"shard_{sid:05d}.jsonl.tmp"
        with open(tmp, "w") as f:
            for k in kept:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        with open(out / f"rejects_{sid:05d}.jsonl", "w") as f:  # kept for auditing the fact-check
            for k in rejects:
                f.write(json.dumps(k, ensure_ascii=False) + "\n")
        tmp.rename(out / f"shard_{sid:05d}.jsonl")
        dt = time.time() - t0
        kept_words = sum(len(k["text"].split()) for k in kept)
        print(f"[shard {sid}] kept {len(kept)}/{len(meta)} (grounded {sum(k['mode'] == 'grounded' for k in kept)}) "
              f"drop={drop} gen_tokens={gen_tok:,} {gen_tok / dt:,.0f} tok/s kept_words={kept_words:,} {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
