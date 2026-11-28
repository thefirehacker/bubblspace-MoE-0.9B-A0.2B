"""Build the Phase 4 RL task set (doc/plan-06-phase-4-rl.md §2): prompts + answer keys, decontaminated, one JSONL.

  maths        GSM8K train + MATH (numeric answers only) from allenai/RLVR-GSM-MATH-IF-Mixed-Constraints (ODC-BY);
               the 8-shot "Question: ... Answer:" wrapper is removed, only the last question is kept
  ifeval       the same set's instruction-constraint prompts (24 constraint types, checker in src/rl/rewards.py)
  code         MBPP train + validation + prompt splits (the test split stays an eval), tests run in the sandbox
  generated    time word problems, binding, tool calls / no-call (with answers where known), live-data questions,
               constraint + known-answer tasks (area ifeval_answer), simple code tasks (scripts/rl_task_gen.py)
  refusal      harmless commands with a known answer (expect an answer) and WildGuardMix prompts whose Tülu reply
               is a refusal (expect a refusal)

Every prompt is checked against the SFT eval index (13-token windows, scripts/phase3_decon.py index) and dropped on
overlap. Output: one task per line, {"id", "area", "kind", "messages", <answer key>}.

    python scripts/rl_build_tasks.py --rlvr ~/rl_raw/allenai__RLVR-GSM-MATH-IF-Mixed-Constraints --tulu ~/phase3_raw \
        --index ~/phase3_logs/sft_eval_index.npz --out ~/rl_tasks/tasks_v1.jsonl
"""
import argparse
import ast
import collections
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rl_task_gen  # noqa: E402
from decontaminate import LENGTHS, window_hashes  # noqa: E402
from phase3_gen_own_sets import CAPITALS  # noqa: E402
from src.rl.rewards import IF_CHECKS, REFUSAL  # noqa: E402


def last_question(user_text: str) -> str:
    q = user_text.split("Question:")[-1]
    return re.sub(r"\s*Answer:\s*$", "", q).strip()


def numeric(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def from_rlvr(path, rng, max_if_chars):
    out, skipped = [], collections.Counter()
    for r in pq.read_table(glob.glob(f"{path}/data/*.parquet")[0]).to_pylist():
        user = r["messages"][-1]["content"]
        if r["dataset"] in ("gsm8k", "MATH"):
            if not numeric(r["ground_truth"]):
                skipped["math_non_numeric"] += 1
                continue
            out.append({"area": "maths", "kind": "number", "source": r["dataset"],
                        "messages": [["user", last_question(user)]], "answer": str(r["ground_truth"]).replace(",", "")})
        else:
            c = json.loads(r["ground_truth"]) if isinstance(r["ground_truth"], str) else r["ground_truth"]
            if c.get("func_name") not in IF_CHECKS:
                skipped[f"if_unknown:{c.get('func_name')}"] += 1
                continue
            ascii_frac = sum(ch.isascii() for ch in user) / max(1, len(user))
            if len(user) > max_if_chars or ascii_frac < 0.95:
                skipped["if_long_or_non_english"] += 1
                continue
            c = {k: v for k, v in c.items() if v is not None}
            out.append({"area": "ifeval", "kind": "ifeval", "source": r.get("constraint_type"),
                        "messages": [["user", user]], "constraint": c})
    return out, skipped


def from_mbpp():
    from datasets import load_dataset
    out = []
    for split in ("train", "validation", "prompt"):
        for ex in load_dataset("google-research-datasets/mbpp", "full", split=split):
            out.append({"area": "code", "kind": "code", "source": f"mbpp/{split}",
                        "messages": [["user", f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}\n"
                                              "Write the Python function in a ```python code block."]],
                        "tests": "\n".join(ex["test_list"]), "setup": ex.get("test_setup_code") or ""})
    return out


def from_synthetic1_code(tulu_raw, max_difficulty, cap, rng):
    """SYNTHETIC-1 verifiable_code problems written for Python, with stdin/stdout test cases (kind code, tests_io)."""
    out = []
    for f in sorted(glob.glob(f"{tulu_raw}/PrimeIntellect__SYNTHETIC-1/data/*.parquet")):
        for r in pq.read_table(f, columns=["task_type", "prompt", "verification_info", "metadata"]).to_pylist():
            if r["task_type"] != "verifiable_code" or "programming language python" not in r["prompt"][:200].lower():
                continue
            try:
                vi = ast.literal_eval(r["verification_info"])
                md = ast.literal_eval(r["metadata"] or "{}")
            except (ValueError, SyntaxError):
                continue
            tests = [[c["input"], c["output"]] for c in vi.get("test_cases", []) if c.get("type") == "stdin_stdout"]
            try:
                diff = int(md.get("difficulty", 99))
            except (TypeError, ValueError):
                diff = 99  # unrated (or a text label): skip
            if not tests or diff > max_difficulty or len(r["prompt"]) > 4000:
                continue
            prompt = r["prompt"].strip() + "\n\nWrite a Python program that reads from standard input and writes to standard output, in a ```python code block."
            out.append({"area": "code", "kind": "code", "source": f"synthetic1/d{diff}",
                        "messages": [["user", prompt]], "tests_io": tests[:10]})
    rng.shuffle(out)
    return out[:cap]


def refusal_tasks(tulu_raw, rng, n_harmless, n_harmful, n_benign_edgy=0):
    """Refusal balance. harmless (capital commands, expect the answer), harmful (WildGuardMix prompts whose Tülu reply
    refuses: expect a refusal) and, added 27 Nov, benign_edgy: WildGuardMix prompts that SOUND harmful but whose Tülu reply
    is a normal answer (expect no refusal). The 45-min RL run without them learned to refuse "kill a Python process",
    "shoot a photo at night", "a regex for email addresses"."""
    out = []
    caps = [c for c in CAPITALS if c[0] not in ("India", "France")]
    for _ in range(n_harmless):
        (c, k) = rng.choice(caps)
        q = rng.choice([f"give me the capital of {c}", f"tell me the capital of {c} in one word", f"capital of {c} pls",
                        f"I need the capital of {c} as JSON"])
        out.append({"area": "refusal", "kind": "refusal", "messages": [["user", q]], "expect": "answer", "answer": k.split(",")[0]})
    harmful, benign = [], []
    for f in sorted(glob.glob(f"{tulu_raw}/allenai__tulu-3-sft-olmo-2-mixture-0225/data/*.parquet")):
        for r in pq.read_table(f, columns=["messages", "source"]).to_pylist():
            if "wildguardmix" not in (r["source"] or ""):
                continue
            m = r["messages"]
            if len(m) < 2 or m[0]["role"] != "user" or len(m[0]["content"]) >= 600:
                continue
            (harmful if REFUSAL.match(m[1]["content"] or "") else benign).append(m[0]["content"])
    rng.shuffle(harmful)
    rng.shuffle(benign)
    # keep our refusal check fair: no task may contain one of its prompts (scripts/eval_chat.py BENIGN/HARMFUL/REALTIME)
    from eval_chat import BENIGN, HARMFUL, REALTIME
    evals = [e.lower() for e in BENIGN + HARMFUL + REALTIME]
    clean = lambda p: not any(e in p.lower() or p.lower() in e for e in evals)
    out += [{"area": "refusal", "kind": "refusal", "messages": [["user", h]], "expect": "refuse"}
            for h in [h for h in harmful if clean(h)][:n_harmful]]
    out += [{"area": "refusal", "kind": "refusal", "source": "benign_edgy", "messages": [["user", b]], "expect": "answer"}
            for b in [b for b in benign if clean(b)][:n_benign_edgy]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rlvr", required=True)
    ap.add_argument("--tulu", required=True, help="dir holding allenai__tulu-3-sft-olmo-2-mixture-0225 (WildGuardMix prompts)")
    ap.add_argument("--index", required=True, help="eval n-gram index from scripts/phase3_decon.py index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_time", type=int, default=6000)
    ap.add_argument("--n_binding", type=int, default=4000)
    ap.add_argument("--n_tool", type=int, default=4000)
    ap.add_argument("--n_live", type=int, default=1000)
    ap.add_argument("--n_if_answer", type=int, default=6000, help="constraint + known answer tasks (27 Nov fix)")
    ap.add_argument("--n_gen_code", type=int, default=6000, help="generated simple code tasks (27 Nov: grow the code pool)")
    ap.add_argument("--n_harmless", type=int, default=1500)
    ap.add_argument("--n_harmful", type=int, default=1500)
    ap.add_argument("--n_benign_edgy", type=int, default=3000, help="harmful-sounding but harmless prompts: expect an answer")
    ap.add_argument("--max_if_chars", type=int, default=3000)
    ap.add_argument("--code_max_difficulty", type=int, default=7, help="SYNTHETIC-1 / CodeContests difficulty cap")
    ap.add_argument("--n_synthetic1_code", type=int, default=4000)
    ap.add_argument("--max_docs", type=int, default=10, help="windows hit by more tasks than this are generic, ignored")
    a = ap.parse_args()
    rng = random.Random(20261130)

    tasks, skipped = from_rlvr(a.rlvr, rng, a.max_if_chars)
    tasks += from_mbpp()
    tasks += from_synthetic1_code(a.tulu, a.code_max_difficulty, a.n_synthetic1_code, rng)
    tasks += rl_task_gen.generate(a.n_time, a.n_binding, a.n_tool, a.n_live, repo_root=str(Path(__file__).resolve().parents[1]),
                                  n_if_answer=a.n_if_answer, n_code=a.n_gen_code)
    tasks += refusal_tasks(a.tulu, rng, a.n_harmless, a.n_harmful, a.n_benign_edgy)

    # decontamination: any 13-token window (or whole short item) shared with an eval -> drop
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / "bubblspace-tokenizer"))
    idx = np.load(a.index)
    sets = {int(n): set(idx[f"h{n}"].tolist()) for n in LENGTHS if f"h{n}" in idx}
    # Pass 1: which eval windows does each task hit? Pass 2: ignore generic windows (hit by more than --max_docs tasks,
    # e.g. IFEval's constraint wording "Your response should contain N paragraphs", shared by design) and drop tasks
    # that hit any rarer one, as scripts/phase3_decon.py does for the SFT data.
    uniq, dropped, seen = [], collections.Counter(), set()
    for t in tasks:
        text = " ".join(m[1] for m in t["messages"] if m[0] == "user")
        h = hashlib.md5(json.dumps(t["messages"]).encode()).hexdigest()  # whole conversation: tool tasks differ in their tool list
        if h in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(h)
        t["id"] = f"{t['area']}-{h[:12]}"
        ids = np.asarray(tok.encode(text, add_special_tokens=False), dtype=np.int64)
        hits = set()
        for n, s in sets.items():
            hits |= {(n, w) for w in window_hashes(ids, n).tolist() if w in s}
        uniq.append((t, hits))
    freq = collections.Counter(w for _, hits in uniq for w in hits)
    kept = []
    for t, hits in uniq:
        if any(freq[w] <= a.max_docs for w in hits):
            dropped[f"decon:{t['area']}"] += 1
            continue
        kept.append(t)
    dropped["generic_windows_ignored"] = sum(1 for w, c in freq.items() if c > a.max_docs)
    rng.shuffle(kept)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for t in kept:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    stats = {"total": len(kept), "by_area_kind": collections.Counter(f"{t['area']}/{t['kind']}" for t in kept),
             "dropped_decon_or_dup": dropped, "skipped_while_reading": skipped}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1, default=dict))
    print(json.dumps(stats, indent=1, default=dict))


if __name__ == "__main__":
    main()
