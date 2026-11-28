"""Pass@k chat evals (doc/plan-06-phase-4-rl.md, "Agreed next steps" step 3): n answers per question, sampled at
temperature 1.0 / top-p 1.0 (the RL v1 rollout and difficulty-filter settings), vLLM backend.

Same prompts, token budgets and checkers as scripts/eval_chat.py (greedy). Every sample is kept, so per question we know
how many of the n answers are right. scripts/summarize_passk.py turns that into pass@1 (mean over samples), pass@n (any
sample right) and the share of questions never / sometimes / always solved: the skill-vs-knowledge map of plan-06
(pass@n near 0 = the model doesn't know it; pass@1 low but pass@n high = a choice problem RL can sharpen).

Tasks: humaneval (164), mbpp (500), gsm8k (1,319), ifeval (541), bfcl simple / multiple / parallel (100 each, same seeded
sample and contamination exclusions as eval_chat.py) + irrelevance (240), refusal (45 harmless, 8 harmful, 8 live-data;
reported as rates over all samples, not pass@n).

    ~/vllm_env/bin/python scripts/eval_chat_passk.py --vllm_model_dir ~/vllm_models/<model> --out benchmark/phase4/passk/gen_<model>.json
    .venv/bin/python scripts/sandbox_exec.py --codegen benchmark/phase4/passk/gen_<model>.json --out benchmark/phase4/passk/codeexec_<model>.json --workers 16
    .venv/bin/python scripts/summarize_passk.py --dir benchmark/phase4/passk --greedy_dir benchmark/phase4 --models <model> ...
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_chat import (BENIGN, HARMFUL, REALTIME, _NO_LIVE, _REFUSAL, extract_code, gsm8k_pred, num_eq, render,  # noqa: E402
                       strip_think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm_model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="refusal,humaneval,mbpp,gsm8k,ifeval,bfcl")
    ap.add_argument("--limit", type=int, default=0, help="cap per task, for a quick check")
    ap.add_argument("--bfcl_per_category", type=int, default=100)
    ap.add_argument("--bfcl_exclude_json", default="benchmark/phase2/contamination_bfcl.json")
    ap.add_argument("--gpu_mem", type=float, default=0.6)
    a = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    from eval_phase2_milestone import _call_matches, parse_calls
    from src.rl.vllm_engine import VLLMEngine

    eng = VLLMEngine(a.vllm_model_dir, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, enable_sleep_mode=False)
    print(f"[model] vLLM {a.vllm_model_dir}  n={a.n} T={a.temperature} top_p={a.top_p} seed={a.seed}", flush=True)
    res = {"checkpoint": a.vllm_model_dir, "step": None, "n": a.n, "temperature": a.temperature, "top_p": a.top_p,
           "seed": a.seed, "decoding": f"{a.n} samples, T={a.temperature}, top-p={a.top_p}, vLLM", "tasks": {}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save = lambda: Path(a.out).write_text(json.dumps(res))
    tasks = a.tasks.split(",")
    lim = lambda xs: xs[:a.limit] if a.limit else xs

    def run(prompts, budget):
        """-> per prompt, a list of n (text, finished) samples."""
        t0 = time.time()
        groups = eng.generate(prompts, n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=budget, seed=a.seed)
        out = []
        for g in groups:
            samples = []
            for r in g:
                txt = r.text
                for s in ("<|im_end|>", "<|endoftext|>"):
                    txt = txt.removesuffix(s)
                samples.append((txt, r.finished))
            out.append(samples)
        n_tok = sum(len(r.completion_ids) for g in groups for r in g)
        dt = time.time() - t0
        print(f"  {len(prompts)} prompts x {a.n}, {n_tok:,} tokens in {dt:.0f}s ({n_tok / max(1e-9, dt):,.0f} tok/s)", flush=True)
        return out

    def stats(outs):
        flat = [x for g in outs for x in g]
        return {"stop_rate": sum(s for _, s in flat) / max(1, len(flat)),
                "think_rate": sum("<think>" in t for t, _ in flat) / max(1, len(flat))}

    if "refusal" in tasks:
        print("[refusal]", flush=True)
        qs = [(q, "harmless") for q in BENIGN] + [(q, "harmful") for q in HARMFUL] + [(q, "realtime") for q in REALTIME]
        outs = run([render(q) for q, _ in qs], 300)
        rows = []
        for (q, kind), g in zip(qs, outs):
            ans = [strip_think(t)[:400] for t, _ in g]
            rows.append({"prompt": q, "kind": kind, "refused": [bool(_REFUSAL.search(x)) for x in ans],
                         "no_live_data": [bool(_NO_LIVE.search(x)) for x in ans], "out": [t[:300] for t, _ in g]})
        res["refusal"] = {"rows": rows, **stats(outs)}
        for kind, key in (("harmless", "refused"), ("harmful", "refused"), ("realtime", "no_live_data")):
            v = [x for r in rows if r["kind"] == kind for x in r[key]]
            print(f"[refusal] {kind} {key}: {sum(v)}/{len(v)} samples", flush=True)
        save()

    def code_task(name, ds, prompts, assemble):
        print(f"[{name}]", flush=True)
        outs = run(prompts, 768)
        rows = []
        for ex, g in zip(ds, outs):
            for i, (t, s) in enumerate(g):
                qid, prog = assemble(ex, t)
                rows.append({"task_id": f"{qid}#{i}", "qid": qid, "sample": i, "completion": t, "stopped": s, "program": prog})
        res["tasks"][name] = rows
        res[f"{name}_stats"] = stats(outs)
        save()

    if "humaneval" in tasks:
        ds = lim(list(load_dataset("openai/openai_humaneval", split="test")))

        def he(ex, t):
            code = extract_code(strip_think(t))
            if f"def {ex['entry_point']}" not in code:
                code = ex["prompt"] + code
            elif "import" not in code:
                code = "\n".join(l for l in ex["prompt"].splitlines() if l.startswith(("import ", "from "))) + "\n" + code
            return ex["task_id"], code + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        code_task("humaneval", ds, [render("Complete the following Python function. Reply with the complete function in a "
                                           f"```python code block.\n\n```python\n{ex['prompt']}```") for ex in ds], he)

    if "mbpp" in tasks:
        ds = lim(list(load_dataset("google-research-datasets/mbpp", "full", split="test")))

        def mb(ex, t):
            setup = ex.get("test_setup_code") or ""
            code = extract_code(strip_think(t))
            return f"mbpp/{ex['task_id']}", (setup + "\n" if setup else "") + code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
        code_task("mbpp", ds, [render(f"{ex['text']}\nYour code should pass this test:\n{ex['test_list'][0]}\n"
                                      "Write the Python function in a ```python code block.") for ex in ds], mb)

    if "gsm8k" in tasks:
        ds = lim(list(load_dataset("openai/gsm8k", "main", split="test")))
        print("[gsm8k]", flush=True)
        outs = run([render(ex["question"]) for ex in ds], 768)
        rows = []
        for qi, (ex, g) in enumerate(zip(ds, outs)):
            gold = ex["answer"].split("####")[-1].strip().replace(",", "")
            preds = [gsm8k_pred(t) for t, _ in g]
            rows.append({"qid": qi, "q": ex["question"][:120], "gold": gold, "preds": preds,
                         "correct": [num_eq(p, gold) for p in preds], "out": [t[:400] for t, _ in g]})
        res["gsm8k"] = {"n": len(rows), **stats(outs), "rows": rows}
        c = [sum(r["correct"]) for r in rows]
        print(f"[gsm8k] pass@1 {sum(c) / (a.n * len(c)):.3f}  pass@{a.n} {sum(x > 0 for x in c) / len(c):.3f}", flush=True)
        save()

    if "ifeval" in tasks:
        ds = lim(list(load_dataset("google/IFEval", split="train")))
        print("[ifeval]", flush=True)
        outs = run([render(ex["prompt"]) for ex in ds], 1024)
        res["ifeval"] = {"n": len(ds), **stats(outs),
                         "rows": [{"qid": ex["key"], "doc": ex, "responses": [strip_think(t) for t, _ in g]}
                                  for ex, g in zip(ds, outs)]}
        save()

    if "bfcl" in tasks:
        repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
        exclude = set()
        if Path(a.bfcl_exclude_json).exists():
            exclude = set(json.loads(Path(a.bfcl_exclude_json).read_text())["contaminated_ids"])
        rng = random.Random(0)  # same seeded sampler (and call order) as eval_chat.py, so the same questions
        out = {}
        for cat in ("simple", "multiple", "parallel", "irrelevance"):
            qs = [json.loads(l) for l in open(hf_hub_download(repo, f"BFCL_v3_{cat}.json", repo_type="dataset"))]
            ans = {}
            if cat != "irrelevance":
                ans = {x["id"]: x["ground_truth"] for x in map(json.loads, open(hf_hub_download(
                    repo, f"possible_answer/BFCL_v3_{cat}.json", repo_type="dataset")))}
                qs = [q for q in qs if q["id"] in ans and q["id"] not in exclude]
                if a.bfcl_per_category and a.bfcl_per_category < len(qs):
                    qs = rng.sample(qs, a.bfcl_per_category)
            qs = lim(qs)
            user = lambda q: q["question"][0][-1]["content"] if isinstance(q["question"][0], list) else q["question"][-1]["content"]
            print(f"[bfcl:{cat}]", flush=True)
            outs = run([render(user(q), q["function"]) for q in qs], 1024)
            rows = []
            for q, g in zip(qs, outs):
                oks = []
                for t, _ in g:
                    calls = parse_calls(t)
                    if cat == "irrelevance":
                        ok = "<tool_call>" not in t
                    else:
                        gts = [(n, p) for gt in ans[q["id"]] for n, p in gt.items()]
                        if cat == "parallel":
                            remaining, ok = list(gts), len(calls) == len(gts)
                            for c in calls:
                                m = next((i for i, (n, p) in enumerate(remaining) if _call_matches(c, n, p, None)), None)
                                if m is None:
                                    ok = False
                                    break
                                remaining.pop(m)
                        else:
                            ok = len(calls) == 1 and _call_matches(calls[0], gts[0][0], gts[0][1], None)
                    oks.append(bool(ok))
                rows.append({"qid": q["id"], "correct": oks, "pred": [t[-300:] for t, _ in g]})
            out[cat] = {"n": len(rows), **stats(outs), "rows": rows}
            c = [sum(r["correct"]) for r in rows]
            print(f"[bfcl:{cat}] pass@1 {sum(c) / (a.n * len(c)):.3f}  pass@{a.n} {sum(x > 0 for x in c) / len(c):.3f}", flush=True)
        res["bfcl"] = out
        save()
    print("[done] saved", a.out, flush=True)


if __name__ == "__main__":
    main()
