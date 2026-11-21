"""Milestone Phase 2 evals that need NO execution of model-generated code
(see doc/eval-plan-phase2.md). HumanEval/MBPP (which do execute code) are
separate and sandboxed.

  cruxeval_o  CRUXEval output prediction (800 problems): given Python code + an
              input, predict the output literal. Official base-model "direct
              output" prompt with 2 shots. Scored by comparing
              ast.literal_eval(pred) == ast.literal_eval(gold) -- literal_eval
              parses literals only; nothing the model writes is executed.
  bfcl        Berkeley Function Calling Leaderboard v3 AST categories (simple,
              multiple, parallel), converted to our training format
              (<tools> JSON in the system turn, <tool_call>{"name","arguments"}</tool_call>
              in the assistant turn). Scored by BFCL-style AST matching: function
              name, required params present, no unknown params, values within the
              gold allowed-value lists (strings normalized).

Usage:
    python scripts/eval_phase2_milestone.py --checkpoint ckpt.pt --tasks cruxeval_o,bfcl \
        --out benchmark/phase2/milestone_step5000.json
"""

import argparse
import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from scripts.eval_phase2_quick import Gen
from src.config import load_config
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

CRUX_HEADER = """You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

"""


def literal(s):
    try:
        return True, ast.literal_eval(s.strip())
    except Exception:
        return False, None


def eval_cruxeval(gen, tok, stop_ids, limit, rng):
    path = hf_hub_download("cruxeval-org/cruxeval", "test.jsonl", repo_type="dataset")
    probs = [json.loads(l) for l in open(path)]
    if limit and limit < len(probs):
        probs = rng.sample(probs, limit)
    correct, parsed, rows = 0, 0, []
    for p in probs:
        prompt = CRUX_HEADER + f"[PYTHON]\n{p['code']}\nassert f({p['input']}) == ??\n[/PYTHON]\n[ANSWER]\n"
        out, _ = gen.greedy(tok.encode(prompt, add_special_tokens=False), 96, stop_ids=stop_ids, stop_text="[/ANSWER]")
        text = tok.decode(out).split("[/ANSWER]")[0].strip().split("\n")[0]
        pred = text.split("==", 1)[1] if "==" in text else text
        ok_p, vp = literal(pred)
        ok_g, vg = literal(p["output"])
        hit = ok_p and ok_g and vp == vg
        parsed += ok_p
        correct += hit
        rows.append({"id": p["id"], "gold": p["output"], "pred": pred.strip()[:120], "correct": hit})
    n = len(probs)
    return {"n": n, "pass_at_1": correct / n, "parsable_literal_rate": parsed / n, "examples": rows[:15]}


def _norm(v):
    if isinstance(v, str):
        return re.sub(r"[\s,./\-_*^]", "", v).lower()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _call_matches(call, gt_name, gt_params, func_schema):
    if call["name"] != gt_name:
        return False
    args = call["args"]
    allowed_keys = set(gt_params)
    if set(args) - allowed_keys:
        return False  # hallucinated parameter
    for k, allowed in gt_params.items():
        optional = "" in allowed
        if k not in args:
            if not optional:
                return False
            continue
        if _norm(args[k]) not in [_norm(a) for a in allowed if a != ""]:
            return False
    return True


def parse_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        if isinstance(obj, dict) and "name" in obj:
            a = obj.get("arguments", {})
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            calls.append({"name": obj["name"], "args": a if isinstance(a, dict) else {}})
    return calls


def eval_bfcl(gen, tok, stop_ids, per_category, rng, exclude=frozenset()):
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    out = {}
    for cat in ("simple", "multiple", "parallel"):
        qs = [json.loads(l) for l in open(hf_hub_download(repo, f"BFCL_v3_{cat}.json", repo_type="dataset"))]
        ans = {a["id"]: a["ground_truth"] for a in map(json.loads, open(hf_hub_download(repo, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")))}
        n_all = sum(1 for q in qs if q["id"] in ans)
        # drop questions flagged by scripts/contamination_check.py (13-gram overlap with training sources)
        qs = [q for q in qs if q["id"] in ans and q["id"] not in exclude]
        n_excluded = n_all - len(qs)
        if per_category and per_category < len(qs):
            qs = rng.sample(qs, per_category)
        hits, emits, valid, rows = 0, 0, 0, []
        for q in qs:
            user = q["question"][0][-1]["content"] if isinstance(q["question"][0], list) else q["question"][-1]["content"]
            tools = json.dumps(q["function"], ensure_ascii=False)
            prompt = (f"<|im_start|>system\n<tools>\n{tools}\n</tools>\n<|im_end|>\n"
                      f"<|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n")
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) > 1700:
                continue
            gen_ids, _ = gen.greedy(ids, 256, stop_ids=stop_ids)
            text = tok.decode(gen_ids)
            calls = parse_calls(text)
            emits += "<tool_call>" in text
            valid += bool(calls)
            gts = [(name, params) for g in ans[q["id"]] for name, params in g.items()]
            if cat == "parallel":
                remaining = list(gts)
                ok = len(calls) == len(gts)
                for c in calls:
                    m = next((i for i, (n, p) in enumerate(remaining) if _call_matches(c, n, p, None)), None)
                    if m is None:
                        ok = False
                        break
                    remaining.pop(m)
            else:
                ok = len(calls) == 1 and _call_matches(calls[0], gts[0][0], gts[0][1], None)
            hits += ok
            rows.append({"id": q["id"], "correct": ok, "pred": text[:300]})
        n = len(rows)
        out[cat] = {"n": n, "excluded_contaminated": n_excluded, "ast_accuracy": hits / max(1, n), "emits_tool_call": emits / max(1, n),
                    "valid_call_json": valid / max(1, n), "examples": rows[:8]}
    out["overall_ast_accuracy"] = sum(v["ast_accuracy"] * v["n"] for v in out.values() if isinstance(v, dict)) / max(1, sum(v["n"] for v in out.values() if isinstance(v, dict)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", default="cruxeval_o,bfcl")
    ap.add_argument("--crux_limit", type=int, default=0, help="0 = all 800")
    ap.add_argument("--bfcl_per_category", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bfcl_exclude_json", default="benchmark/phase2/contamination_bfcl.json",
                    help="contamination report; its contaminated_ids are excluded from BFCL scoring")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    model = Transformer(cfg.model).to("cuda").eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    gen = Gen(model, tok, "cuda", cfg.model.max_seq_len)
    stop_ids = (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
    rng = random.Random(args.seed)
    result = {"checkpoint": args.checkpoint, "step": ckpt.get("step")}
    tasks = [t.strip() for t in args.tasks.split(",")]
    if "cruxeval_o" in tasks:
        result["cruxeval_o"] = eval_cruxeval(gen, tok, stop_ids, args.crux_limit, rng)
        c = result["cruxeval_o"]
        print(f"[cruxeval_o] n={c['n']} pass@1={c['pass_at_1']:.3f} parsable={c['parsable_literal_rate']:.2f}")
    if "bfcl" in tasks:
        exclude = frozenset()
        if Path(args.bfcl_exclude_json).exists():
            exclude = frozenset(json.loads(Path(args.bfcl_exclude_json).read_text())["contaminated_ids"])
            print(f"[bfcl] excluding {len(exclude)} contaminated question ids")
        result["bfcl"] = eval_bfcl(gen, tok, stop_ids, args.bfcl_per_category, rng, exclude)
        result["bfcl"]["decontaminated"] = bool(exclude)
        for cat in ("simple", "multiple", "parallel"):
            b = result["bfcl"][cat]
            print(f"[bfcl:{cat}] n={b['n']} ast={b['ast_accuracy']:.3f} emits={b['emits_tool_call']:.2f} valid_json={b['valid_call_json']:.2f}")
        print(f"[bfcl] overall ast={result['bfcl']['overall_ast_accuracy']:.3f}")
    result["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[milestone] done in {result['elapsed_s']}s -> {args.out}")


if __name__ == "__main__":
    main()
