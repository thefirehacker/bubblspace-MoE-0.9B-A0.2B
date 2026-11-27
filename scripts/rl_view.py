"""Look inside an RL run (scripts/train_grpo.py output dir): per-step metrics, evals, and the actual rollouts.

    python3 scripts/rl_view.py <out_dir> --info                  # settings, task mix, task pool, what it is doing right now
    python3 scripts/rl_view.py <out_dir>                         # step table + eval table
    python3 scripts/rl_view.py <out_dir> --step 5                # one step: a few prompts with all 8 answers, rewards, advantages
    python3 scripts/rl_view.py <out_dir> --step 5 --area time --groups 3 --chars 400
    python3 scripts/rl_view.py <out_dir> --eval 20 --area tool   # greedy eval answers at an eval step
    python3 scripts/rl_view.py <out_dir> --follow                # live: current phase + step table, every 10 s (Ctrl-C)

Only the standard library: runs anywhere the output dir is visible (fire01, or after downloading it from HF).
"""
import argparse
import gzip
import json
import random
import time
from collections import defaultdict
from pathlib import Path


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def status_line(d):
    p = d / "status.json"
    if p.exists():
        s = json.loads(p.read_text())
        m, sec = divmod(s.get("since_start_s", 0), 60)
        print(f"NOW ({s.get('time_utc')} UTC, {m} min {sec} s since start): {s['phase']}\n")
    else:
        print("NOW: run not started yet (no status.json)\n")


def info(d):
    status_line(d)
    p = d / "run_config.json"
    if not p.exists():
        print("(run_config.json not written yet: the run is still in setup)")
        return
    r = json.loads(p.read_text())
    C = r["config"]
    print(f"run {C['run_name']}  started {r['started_utc']} UTC  ({r['steps']} steps)")
    print(f"start model : {C['init_checkpoint']}")
    print(f"tasks file  : {C['tasks']}")
    print(f"per step    : {C['prompts_per_step']} prompts x {C['n_samples']} answers = {C['prompts_per_step'] * C['n_samples']}, "
          f"max {C['max_tokens']} tokens, temperature {C.get('temperature', 1.0)}")
    print(f"training    : lr {C['lr']}, warmup {C.get('warmup_steps')}, KL beta {C['kl_beta']}, clip {C['clip_eps']}, "
          f"updates/batch {C.get('ppo_epochs', 1)}, micro-batch {C.get('micro_batch_tokens')} tokens")
    print(f"vLLM        : {C.get('vllm_gpu_mem')} of GPU memory; MoE backend {r.get('moe_backend')}")
    print(f"tests       : held-out set at step 0 and every {C.get('eval_every')} steps; checkpoint every {C.get('save_every')}")
    print(f"\n{'subject':<10} {'share':>6} {'≈/step':>7} {'pool':>7} {'held-out':>9}")
    w = r["area_weights_used"]
    tot = sum(w.values())
    for a in sorted(w, key=lambda k: -w[k]):
        print(f"{a:<10} {w[a] / tot:6.0%} {C['prompts_per_step'] * w[a] / tot:7.1f} {r['task_pool_per_area'].get(a, 0):7,} "
              f"{r['heldout_per_area'].get(a, 0):9}")


def step_table(d):
    rows = read_jsonl(d / "train_log.jsonl")
    if not rows:
        print("(no steps logged yet)")
        return
    areas = sorted({k[7:] for r in rows for k in r if k.startswith("reward/")})
    print(f"{'step':>4} {'reward':>6} {'0-adv':>5} {'kl':>7} {'len':>5} {'fin':>4} {'gnorm':>6} {'gen s':>5} {'rew s':>5} {'step s':>6} | "
          + " ".join(f"{a[:9]:>9}" for a in areas))
    for r in rows:
        print(f"{r['step']:>4} {r['reward_mean']:6.3f} {r['zero_adv_groups']:5.2f} {r.get('kl', 0):7.4f} {r['answer_len']:5.0f} "
              f"{r['finished_frac']:4.2f} {r.get('grad_norm', 0):6.3f} {r['t_gen']:5.0f} {r['t_reward']:5.0f} {r['t_step']:6.0f} | "
              + " ".join(f"{r.get('reward/' + a, float('nan')):5.2f}({r.get('n/' + a, 0):2d})" for a in areas))
    print("  (subject columns: reward this step (number of prompts this step))")
    evals = read_jsonl(d / "evals.jsonl")
    if evals:
        keys = sorted(k for k in evals[0] if k.startswith("eval/"))
        print("\nheld-out evals (greedy):\n" + f"{'step':>4} " + " ".join(f"{k[5:][:8]:>8}" for k in keys))
        for e in evals:
            print(f"{e['step']:>4} " + " ".join(f"{e.get(k, float('nan')):8.3f}" for k in keys))


def show_step(d, step, area, groups, chars, seed):
    rows = read_jsonl(d / "rollouts" / f"step_{step:04d}.jsonl.gz")
    if not rows:
        print(f"no rollouts for step {step}")
        return
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    ids = [i for i, g in by.items() if not area or g[0]["area"] == area]
    random.Random(seed).shuffle(ids)
    for i in ids[:groups]:
        g = sorted(by[i], key=lambda r: -r["reward"])
        print("=" * 100)
        print(f"[{g[0]['area']}/{g[0]['kind']}] {i}   rewards {[round(r['reward'], 2) for r in by[i]]}")
        print("PROMPT:", g[0]["prompt"][-1][1][:chars].replace("\n", " "))
        for r in g:
            print(f"  reward {r['reward']:.2f}  adv {r['advantage']:+.2f}  {r['n_tokens']:4d} tok  {'' if r['finished'] else '(cut off) '}"
                  f"| {r['answer'][-chars:].replace(chr(10), ' ')}")


def show_eval(d, step, area, groups, chars):
    rows = [r for r in read_jsonl(d / "eval_samples" / f"step_{step:04d}.jsonl.gz") if not area or r["area"] == area]
    for r in rows[:groups]:
        print(f"[{r['area']}] reward {r['reward']:.2f} | Q: {r['prompt'][-1][1][:200].replace(chr(10), ' ')}\n   A: {r['answer'][-chars:].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--step", type=int)
    ap.add_argument("--eval", type=int)
    ap.add_argument("--area", default=None)
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args()
    d = Path(a.out_dir)
    if a.step is not None:
        show_step(d, a.step, a.area, a.groups, a.chars, a.seed)
    elif a.eval is not None:
        show_eval(d, a.eval, a.area, a.groups, a.chars)
    elif a.info:
        info(d)
    elif a.follow:
        while True:
            print("\033[2J\033[H", end="")
            status_line(d)
            step_table(d)
            time.sleep(10)
    else:
        step_table(d)


if __name__ == "__main__":
    main()
