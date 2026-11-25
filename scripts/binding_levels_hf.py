"""Three-level binding evaluation for any Hugging Face causal LM (any revision / training checkpoint).

Levels (doc/wsp-experiments.md §6.1, §7.6):
  L1  single row   each context scored alone: is the correct ending more likely than the wrong one?
                   (how WinoGrande is officially scored; word-frequency shortcuts can pass)
  L2  pair flip    both halves of a minimal pair must be right (Elazar et al. 2021 "group scoring";
                   the BubblSpace probe rule)
  L3  generation   the model must WRITE the right answer itself (greedy, full vocabulary), in both halves

Suites:
  probes   BubblSpace minimal pairs (v2 / v3 / v4 / v5): ctxA/ctxB differ in the entity, c1/c2 are the endings.
           L3 = greedy continuation reproduces the correct ending up to its first token that differs from the
           wrong ending (same rule as scripts/eval_binding_generation.py).
  winogrande  WinoGrande-XL validation (1,267 rows), twins = adjacent rows with the same options and opposite answers.
           L1 = lm-eval partial scoring (log-prob of the text after the blank, given the prefix with each option).
           L3 = 5-shot "which option fills the blank" prompt, greedy; a row is a hit if the answer starts with the
           correct option. The deciding word in WinoGrande usually comes AFTER the blank, so plain continuation
           can't be used. `answered_with_option` reports how often the model answers in the expected format.

    python scripts/binding_levels_hf.py --model EleutherAI/pythia-1b --revision step143000 \
        --probes v4=path/v4.json v5=path/v5.json --winogrande --out results/pythia-1b_step143000.json
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def fetch(name, revision, cache_dir):
    """Download only what loading needs (safetensors if present, else pickle .bin) and return the local path.
    Pickle checkpoints are scanned with ClamAV before loading (project rule for unscanned files); transformers
    then loads them with torch's weights_only unpickler."""
    import subprocess
    from huggingface_hub import HfApi, snapshot_download
    sib = HfApi().model_info(name, revision=revision, files_metadata=True).siblings
    files = [f.rfilename for f in sib]
    has_st = any(f.endswith(".safetensors") for f in files)
    # weight-file hashes: lets the report catch hubs that store identical weights under different
    # revisions (found 24 Nov: EleutherAI/pythia-2.8b steps 1000-20000 and main share one .bin)
    wext = ".safetensors" if has_st else ".bin"
    global WEIGHT_SHA
    WEIGHT_SHA = {f.rfilename: f.lfs.sha256 for f in sib if f.rfilename.endswith(wext) and f.lfs}
    pats = ["*.json", "*.txt", "*.model", "*.tiktoken"] + (["*.safetensors"] if has_st else ["pytorch_model*.bin"])
    path = snapshot_download(name, revision=revision, cache_dir=cache_dir, allow_patterns=pats)
    if not has_st:
        if not shutil.which("clamscan"):
            raise SystemExit(f"{name}@{revision} ships pickle .bin only and clamscan is not installed here; run on fire01")
        r = subprocess.run(["clamscan", "-r", "--no-summary", "--infected", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"ClamAV flagged or failed on {name}@{revision}: {r.stdout} {r.stderr}")
        print(f"[scan] ClamAV clean: {name}@{revision}", flush=True)
    return path, ("safetensors" if has_st else "bin+clamav")


WEIGHT_SHA = {}


class LM:
    def __init__(self, name, revision, cache_dir, dtype):
        path, self.weights_format = fetch(name, revision, cache_dir)
        try:
            self.tok = AutoTokenizer.from_pretrained(path)
        except (ValueError, OSError):
            # some intermediate branches ship no tokenizer files (e.g. OLMo-2-1124-7B stage1-step101000);
            # the tokenizer is fixed for the whole run, so take it from the main branch
            self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.eos = self.tok.eos_token_id

    def ctx_ids(self, text):
        ids = self.tok(text, add_special_tokens=True)["input_ids"]  # BOS where the model expects one
        return ids or [self.eos]

    def cont_ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def logprob(self, ctx, cont):
        ids = torch.tensor([ctx + cont], device=self.device)
        logits = self.model(ids[:, :-1]).logits[0, -len(cont):].float()
        return F.log_softmax(logits, -1).gather(1, ids[0, -len(cont):, None]).sum().item()

    @torch.no_grad()
    def greedy(self, ctx, n):
        ids = torch.tensor([ctx], device=self.device)
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                                  pad_token_id=self.eos if self.eos is not None else 0)
        return out[0, len(ctx):].tolist()


def divergence(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) - 1)


def run_probes(lm, path):
    cases = json.loads(Path(path).read_text())["cases"]
    rows = []
    for c in cases:
        r = {"name": c["name"], "category": c.get("category")}
        e1, e2 = lm.cont_ids(c["c1"]), lm.cont_ids(c["c2"])
        k = divergence(e1, e2) + 1
        for side, ctx, gold, wrong, gids in (("A", c["ctxA"], c["c1"], c["c2"], e1), ("B", c["ctxB"], c["c2"], c["c1"], e2)):
            cid = lm.ctx_ids(ctx)
            lp_g, lp_w = lm.logprob(cid, gids), lm.logprob(cid, lm.cont_ids(wrong))
            gen = lm.greedy(cid, max(len(gids), 8))
            kk = min(k, len(gids))
            r[side] = {"l1": lp_g > lp_w, "l3": gen[:kk] == gids[:kk], "gen": lm.tok.decode(gen[:12])}
        r["l2"] = r["A"]["l1"] and r["B"]["l1"]
        r["l3_pair"] = r["A"]["l3"] and r["B"]["l3"]
        rows.append(r)
    return summarize(rows, "category")


def winogrande_pairs(ds):
    pairs, i = [], 0
    while i + 1 < len(ds):
        a, b = ds[i], ds[i + 1]
        if a["option1"] == b["option1"] and a["option2"] == b["option2"] and a["answer"] != b["answer"]:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def run_winogrande(lm, n_shot=5, seed=0):
    from datasets import load_dataset
    val = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    train = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    rng = random.Random(seed)
    shots = [train[j] for j in rng.sample(range(len(train)), n_shot)]

    def opt(r, k):
        return r["option1"] if k == "1" else r["option2"]

    def fewshot(r):
        return (f"Sentence: {r['sentence']}\nOptions: {r['option1']} or {r['option2']}\n"
                f"The blank _ refers to:")

    prompt = "".join(fewshot(s) + f" {opt(s, s['answer'])}\n\n" for s in shots)
    pairs = winogrande_pairs(val)
    in_pair = {i for p in pairs for i in p}
    res = {}
    for i, r in enumerate(val):
        pre, suf = r["sentence"].split("_", 1)
        scores = {}
        for k in ("1", "2"):  # lm-eval partial scoring: P(suffix | prefix + option)
            scores[k] = lm.logprob(lm.ctx_ids((pre + opt(r, k)).rstrip()), lm.cont_ids(" " + suf.strip())) if suf.strip() else 0.0
        l1 = scores[r["answer"]] > scores["2" if r["answer"] == "1" else "1"]
        gold, other = opt(r, r["answer"]), opt(r, "2" if r["answer"] == "1" else "1")
        gen = lm.tok.decode(lm.greedy(lm.ctx_ids(prompt + fewshot(r)), 8)).strip().split("\n")[0]
        g = gen.lower()
        hit = g.startswith(gold.lower()) and not (other.lower().startswith(gold.lower()) and g.startswith(other.lower()))
        # shortcut diagnostics: which name the model writes, by position in the sentence
        pos1 = r["sentence"].find(r["option1"]); pos2 = r["sentence"].find(r["option2"])
        first, second = (r["option1"], r["option2"]) if (pos1 != -1 and (pos2 == -1 or pos1 <= pos2)) else (r["option2"], r["option1"])
        wrote = first if g.startswith(first.lower()) else (second if g.startswith(second.lower()) else None)
        # margin = score(correct option) - score(wrong option); saved so pairs can be re-scored without the
        # model's general preference for one name (doc/binding-study-02-priors-and-word-order.md)
        margin = scores[r["answer"]] - scores["2" if r["answer"] == "1" else "1"]
        res[i] = {"l1": l1, "l3": hit, "answered_with_option": wrote is not None, "gen": gen[:40], "wrote": wrote,
                  "margin": round(margin, 4),
                  "wrote_first_mentioned": wrote == first, "wrote_second_mentioned": wrote == second,
                  "l1_prefers_first": scores["1" if first == r["option1"] else "2"] > scores["2" if first == r["option1"] else "1"]}
    rows = []
    for a, b in pairs:
        rows.append({"A": res[a], "B": res[b], "l2": res[a]["l1"] and res[b]["l1"],
                     "l3_pair": res[a]["l3"] and res[b]["l3"], "category": "winogrande"})
    out = summarize(rows, "category")
    out["n_rows"] = len(val)
    out["n_twin_pairs"] = len(pairs)
    out["l1_all_rows"] = round(sum(v["l1"] for v in res.values()) / len(res), 4)
    out["l3_all_rows"] = round(sum(v["l3"] for v in res.values()) / len(res), 4)
    out["answered_with_option"] = round(sum(v["answered_with_option"] for v in res.values()) / len(res), 4)
    out["rows_outside_pairs"] = len(val) - len(in_pair)
    # prior-free pair score: the twins share their two options and have opposite answers, so a general preference for
    # one name adds to one twin's margin and subtracts from the other's. Their SUM is > 0 only if the differing word
    # moves the preference the right way. Chance 50%.
    pf = [res[a]["margin"] + res[b]["margin"] for a, b in pairs]
    out["prior_free_pairs"] = round(sum(x > 0 for x in pf) / len(pf), 4)
    out["prior_free_median_effect_nats"] = round(sorted(pf)[len(pf) // 2], 4)
    # shortcut phases (Chan et al. 2026: frequency -> position -> correct)
    answered = [v for v in res.values() if v["answered_with_option"]]
    out["shortcuts"] = {
        "wrote_first_mentioned": round(sum(v["wrote_first_mentioned"] for v in res.values()) / len(res), 4),
        "wrote_second_mentioned": round(sum(v["wrote_second_mentioned"] for v in res.values()) / len(res), 4),
        "first_mentioned_share_of_answers": round(sum(v["wrote_first_mentioned"] for v in answered) / max(1, len(answered)), 4),
        "same_name_for_both_twins": round(sum(1 for a, b in pairs if res[a]["wrote"] and res[a]["wrote"] == res[b]["wrote"]) / len(pairs), 4),
        "l1_prefers_first_mentioned": round(sum(v["l1_prefers_first"] for v in res.values()) / len(res), 4),
    }
    out["rows"] = [{"i": i, **{k: v[k] for k in ("l1", "l3", "wrote", "gen", "margin")}} for i, v in res.items()]
    return out


def summarize(rows, key):
    n = len(rows)
    s = {"n_pairs": n,
         "L1": round(sum(r[x]["l1"] for r in rows for x in "AB") / (2 * n), 4),
         "L2": round(sum(r["l2"] for r in rows) / n, 4),
         "L3_rows": round(sum(r[x]["l3"] for r in rows for x in "AB") / (2 * n), 4),
         "L3": round(sum(r["l3_pair"] for r in rows) / n, 4)}
    cats = {}
    for r in rows:
        cats.setdefault(r[key], []).append(r)
    s["by_category"] = {c: {"n": len(v), "L2": round(sum(r["l2"] for r in v) / len(v), 3),
                            "L3": round(sum(r["l3_pair"] for r in v) / len(v), 3)} for c, v in cats.items()}
    s["examples"] = [{k: r[k] for k in ("A", "B")} for r in rows[:6]]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--probes", nargs="*", default=[], help="name=path pairs, e.g. v4=benchmark/data/.../v4.json")
    ap.add_argument("--winogrande", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache_dir", default=os.path.expanduser("~/hf_study_cache"))
    ap.add_argument("--keep_cache", action="store_true", help="don't delete the downloaded weights afterwards")
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    lm = LM(a.model, a.revision, a.cache_dir, dtype)
    res = {"model": a.model, "revision": a.revision, "n_params": sum(p.numel() for p in lm.model.parameters()),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "weights_format": lm.weights_format, "weight_sha256": WEIGHT_SHA, "suites": {}}
    for spec in a.probes:
        name, path = spec.split("=", 1)
        res["suites"][name] = run_probes(lm, path)
        s = res["suites"][name]
        print(f"{name}: L1 {s['L1']:.3f}  L2 {s['L2']:.3f}  L3 {s['L3']:.3f}  (n={s['n_pairs']})", flush=True)
    if a.winogrande:
        res["suites"]["winogrande"] = run_winogrande(lm)
        s = res["suites"]["winogrande"]
        print(f"winogrande: L1 {s['L1']:.3f} (all rows {s['l1_all_rows']:.3f})  L2 {s['L2']:.3f}  L3 {s['L3']:.3f}  "
              f"answered_with_option {s['answered_with_option']:.3f}  pairs {s['n_twin_pairs']}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    del lm
    torch.cuda.empty_cache()
    if not a.keep_cache:  # checkpoints are large; never keep more than one model on disk
        shutil.rmtree(a.cache_dir, ignore_errors=True)
    print(f"wrote {a.out} ({res['elapsed_s']} s)")


if __name__ == "__main__":
    main()
