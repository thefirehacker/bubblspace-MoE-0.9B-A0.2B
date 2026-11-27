"""Difficulty filter for Phase 4 RL (doc/rl-primer-phase4.md §6), and a per-area pass@1 / pass@8 map of the model.

The model answers every task n times (vLLM, temperature 1.0); each answer is scored with src/rl/rewards.py (code in the
docker sandbox). A task teaches GRPO something only if some answers pass and some fail, so tasks solved between 1 and
n-1 times are kept. The per-area summary is the skill-vs-knowledge map of plan-06: pass@1 low but pass@8 high = a
skill RL can sharpen; pass@8 near 0 = knowledge the model lacks.

Runs in the vLLM env (Python 3.12) with the plugin installed:
    python scripts/rl_filter.py --model_dir ~/vllm_models/phase3-step1500 --tasks ~/rl_tasks/tasks_v1.jsonl \
        --out ~/rl_tasks/filtered_phase3_step1500.jsonl --limit_per_area 1500
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rl.common import load_tasks, render, score_batch  # noqa: E402
from src.rl.vllm_engine import VLLMEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit_per_area", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512, help="tasks per generate() call")
    ap.add_argument("--gpu_mem", type=float, default=0.35)
    ap.add_argument("--sandbox_workers", type=int, default=16)
    a = ap.parse_args()

    tasks = load_tasks(a.tasks, limit_per_area=a.limit_per_area)
    eng = VLLMEngine(a.model_dir, gpu_memory_utilization=a.gpu_mem, max_model_len=8192, enable_sleep_mode=False)
    t0, done = time.time(), 0
    stats = defaultdict(lambda: {"tasks": 0, "pass_sum": 0.0, "any": 0, "all": 0, "kept": 0, "finished": 0, "samples": 0})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo, open(Path(a.out).with_suffix(".all.jsonl"), "w") as fa:
        for s in range(0, len(tasks), a.chunk):
            chunk = tasks[s:s + a.chunk]
            groups = eng.generate([render(t["messages"]) for t in chunk], n=a.n, temperature=1.0, max_tokens=a.max_tokens, seed=s)
            flat_t = [t for t, g in zip(chunk, groups) for _ in g]
            flat_r = [r for g in groups for r in g]
            rew = score_batch(flat_t, [r.text for r in flat_r], [r.finished for r in flat_r], a.sandbox_workers)
            for i, t in enumerate(chunk):
                rs = rew[i * a.n:(i + 1) * a.n]
                fin = sum(r.finished for r in flat_r[i * a.n:(i + 1) * a.n])
                k = sum(x >= 1.0 for x in rs)
                st = stats[t["area"]]
                st["tasks"] += 1; st["pass_sum"] += sum(rs) / a.n; st["any"] += k > 0; st["all"] += k == a.n
                st["finished"] += fin; st["samples"] += a.n
                rec = dict(t, sft_pass=k, sft_mean_reward=round(sum(rs) / a.n, 3))
                fa.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if 0 < k < a.n or (0 < sum(rs) < a.n):  # partial credit also counts as a mixed group
                    st["kept"] += 1
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += len(chunk)
            print(f"[filter] {done}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    summary = {area: {"tasks": st["tasks"], "pass@1": round(st["pass_sum"] / st["tasks"], 3),
                      f"pass@{a.n}": round(st["any"] / st["tasks"], 3), f"all_{a.n}_correct": round(st["all"] / st["tasks"], 3),
                      "kept_for_rl": st["kept"], "finished_frac": round(st["finished"] / st["samples"], 3)}
               for area, st in sorted(stats.items())}
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
