"""Summary of the pass@k chat evals (scripts/eval_chat_passk.py + sandbox_exec.py), next to the greedy results.

Per task and model: greedy accuracy (benchmark/phase4, same prompts), pass@1 = mean over the n samples, pass@n = any sample
right, and the share of questions solved by 0, 1..n-1, and n of the n samples (plan-06's skill-vs-knowledge map). With
two models, also how questions moved between them: of the questions the first never solves, how many the second solves at
least once (gained), and the reverse (lost: the second never solves what the first sometimes did, i.e. narrowing).

    .venv/bin/python scripts/summarize_passk.py --dir benchmark/phase4/passk --greedy_dir benchmark/phase4 \
        --models phase3-run02b-step1500-tokens40B phase4-rlv1-step500-tokens40B
IFEval needs lm-eval (scored here, per sample, with its own checkers).
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

NAMES = {"phase3-step1500": "SFT run 01", "phase3-run02b-step1500-tokens40B": "SFT run 02b",
         "phase4-rlv1-step100-tokens40B": "RL v1 step 100", "phase4-rlv1-step300-tokens40B": "RL v1 step 300",
         "phase4-rlv1-step500-tokens40B": "RL v1 step 500"}
TASKS = [("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("gsm8k", "GSM8K"), ("ifeval", "IFEval (prompt, strict)"),
         ("bfcl:simple", "BFCL simple"), ("bfcl:multiple", "BFCL multiple"), ("bfcl:parallel", "BFCL parallel"),
         ("bfcl:irrelevance", "BFCL no tool needed")]


def load(p):
    """JSON file, or its .gz copy (the repo keeps the large gen/codeexec files gzipped)."""
    p = Path(p)
    if p.exists():
        return json.loads(p.read_text())
    gz = p.with_name(p.name + ".gz")
    return json.loads(gzip.decompress(gz.read_bytes())) if gz.exists() else None


def counts(gen, cx):
    """-> {task: {qid: number of correct samples}}"""
    out = defaultdict(dict)
    if cx:
        for task in ("humaneval", "mbpp"):
            per = defaultdict(int)
            for r in cx["results"].get(task, []):
                per[r["task_id"].rsplit("#", 1)[0]] += bool(r["passed"])
            out[task] = dict(per)
    if "gsm8k" in gen:
        out["gsm8k"] = {str(r["qid"]): sum(r["correct"]) for r in gen["gsm8k"]["rows"]}
    if "ifeval" in gen:
        from lm_eval.tasks.ifeval.utils import process_results
        out["ifeval"] = {str(r["qid"]): sum(bool(process_results(r["doc"], [x])["prompt_level_strict_acc"]) for x in r["responses"])
                         for r in gen["ifeval"]["rows"]}
    for cat, v in gen.get("bfcl", {}).items():
        out[f"bfcl:{cat}"] = {r["qid"]: sum(r["correct"]) for r in v["rows"]}
    return out


def greedy(chat, cx, task):
    if not chat:
        return None
    if task in ("humaneval", "mbpp"):
        return ((cx or {}).get("summary", {}).get(task) or {}).get("pass_at_1")
    if task == "gsm8k":
        return chat.get("gsm8k", {}).get("accuracy")
    if task == "ifeval":
        return chat.get("ifeval", {}).get("scores", {}).get("prompt_level_strict_acc")
    return chat.get("bfcl", {}).get(task.split(":")[1], {}).get("accuracy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--greedy_dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: <dir>/summary.md")
    a = ap.parse_args()
    d, gd = Path(a.dir), Path(a.greedy_dir)
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}"

    allc, n, refusal = {}, None, {}
    for m in a.models:
        gen = load(d / f"gen_{m}.json")
        if not gen:
            print(f"missing gen_{m}.json")
            continue
        n = gen["n"]
        allc[m] = c = counts(gen, load(d / f"codeexec_{m}.json"))
        (d / f"counts_{m}.json").write_text(json.dumps(c, indent=0))
        if "refusal" in gen:
            rows = gen["refusal"]["rows"]
            rate = lambda kind, key: (sum(sum(r[key]) for r in rows if r["kind"] == kind), sum(len(r[key]) for r in rows if r["kind"] == kind))
            refusal[m] = {"harmless": rate("harmless", "refused"), "harmful": rate("harmful", "refused"),
                          "realtime": rate("realtime", "no_live_data")}
        decoding = gen["decoding"]

    lines = [f"# Pass@{n} chat benchmarks: {', '.join(NAMES.get(m, m) for m in allc)}", "",
             f"Decoding: {decoding} (the RL v1 rollout settings), A10. Greedy = the single greedy answer from "
             f"`benchmark/phase4/` (same prompts and checkers). pass@1 = mean over the {n} samples; pass@{n} = at least one "
             f"of the {n} right. The last three columns split the questions by how many of the {n} samples were right: "
             f"0/{n} = the model doesn't produce it (knowledge or ability gap), 1–{n - 1}/{n} = it can but doesn't pick it "
             f"reliably (what RL sharpens), {n}/{n} = solved.", "",
             f"| Task | Model | Questions | Greedy % | pass@1 % (T=1) | pass@{n} % | 0/{n} % | 1–{n - 1}/{n} % | {n}/{n} % |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task, label in TASKS:
        for m, c in allc.items():
            v = list(c.get(task, {}).values())
            if not v:
                continue
            q = len(v)
            g = greedy(load(gd / f"chat_{m}.json"), load(gd / f"codeexec_{m}.json"), task)
            lines.append(f"| {label} | {NAMES.get(m, m)} | {q} | {pct(g)} | {pct(sum(v) / (n * q))} | {pct(sum(x > 0 for x in v) / q)} | "
                         f"{pct(sum(x == 0 for x in v) / q)} | {pct(sum(0 < x < n for x in v) / q)} | {pct(sum(x == n for x in v) / q)} |")

    if len(allc) == 2:
        (m0, c0), (m1, c1) = allc.items()
        n0, n1 = NAMES.get(m0, m0), NAMES.get(m1, m1)
        lines += ["", f"**{n0} → {n1}, per question** (gained: never solved by {n0}, solved at least once by {n1}; lost: the reverse)", "",
                  f"| Task | Never by {n0} | of those, gained | Solved ≥1 by {n0} | of those, lost | Mean right samples per question |",
                  "|---|---|---|---|---|---|"]
        for task, label in TASKS:
            a0, a1 = c0.get(task, {}), c1.get(task, {})
            qs = [k for k in a0 if k in a1]
            if not qs:
                continue
            never = [k for k in qs if a0[k] == 0]
            some = [k for k in qs if a0[k] > 0]
            gained = sum(a1[k] > 0 for k in never)
            lost = sum(a1[k] == 0 for k in some)
            lines.append(f"| {label} | {len(never)} | {gained} | {len(some)} | {lost} | "
                         f"{sum(a0[k] for k in qs) / len(qs):.2f} → {sum(a1[k] for k in qs) / len(qs):.2f} of {n} |")

    if refusal:
        lines += ["", f"**Refusal check, all {n} samples per prompt** (greedy: one answer per prompt)", "",
                  "| Model | Harmless refused (want low) | Harmful refused (want high) | Live data honest (want high) |", "|---|---|---|---|"]
        for m, r in refusal.items():
            f = lambda x: f"{x[0]}/{x[1]} ({100 * x[0] / max(1, x[1]):.0f}%)"
            lines.append(f"| {NAMES.get(m, m)} | {f(r['harmless'])} | {f(r['harmful'])} | {f(r['realtime'])} |")

    out = "\n".join(lines) + "\n"
    Path(a.out or d / "summary.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
