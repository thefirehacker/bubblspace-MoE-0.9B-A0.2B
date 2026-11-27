"""Phase 4 RL: GRPO with verifiable rewards (doc/rl-primer-phase4.md). One process, one GPU:

  policy      our Transformer (src/model), fp32 master weights, AdamW
  reference   a frozen copy of the starting SFT model (KL penalty)
  rollouts    vLLM + our plugin (src/rl/vllm_engine.py): weights synced after every update, asleep while training
  rewards     plain-code checkers (src/rl/rewards.py; code in the docker sandbox via src/rl/common.py)

One step: sample prompts -> wake vLLM, sync weights, generate n answers each -> sleep vLLM -> score -> group advantages
-> policy-gradient loss (clipped ratio) + KL to the reference on the answer tokens -> AdamW step.
With one update per batch the ratio starts at exactly 1 (its value is the PyTorch policy's own log-prob, not vLLM's,
which differ by ~0.03 in bf16), so clipping only acts when ppo_epochs > 1.

Runs in the vLLM env (Python 3.12, plugin installed):
    python scripts/train_grpo.py configs/rl_v1.yaml
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from src.config import load_config  # noqa: E402
from src.model.moe import probe_grouped_mm  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402
from src.rl.common import load_tasks, render, score_batch  # noqa: E402


def load_state(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    ck = torch.load(path, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def token_logprobs(model, seqs, n_prompt, device, ptoken):
    """Per-token log-probs of each sequence's answer tokens (positions >= n_prompt). seqs: list of id lists, padded here.
    Returns (list of 1-D tensors, total load-balance loss)."""
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), ptoken, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, lb, _, _ = model(x)
    out = []
    for i, (s, p) in enumerate(zip(seqs, n_prompt)):
        lg = logits[i, p - 1:len(s) - 1].float()           # position t predicts token t+1
        tgt = x[i, p:len(s)]
        out.append(torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).squeeze(-1))
    return out, lb


def microbatches(items, budget):
    """Group (seq, ...) items into micro-batches of at most `budget` padded tokens, longest first."""
    items = sorted(items, key=lambda it: -len(it["seq"]))
    mb, cur, cur_max = [], [], 0
    for it in items:
        m = max(cur_max, len(it["seq"]))
        if cur and m * (len(cur) + 1) > budget:
            mb.append(cur)
            cur, m = [], len(it["seq"])
        cur.append(it)
        cur_max = m
    if cur:
        mb.append(cur)
    return mb


def dump_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max_steps", type=int, default=None, help="override, e.g. for a smoke test")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value (value parsed as YAML)")
    a = ap.parse_args()
    C = yaml.safe_load(open(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        C[k] = yaml.safe_load(v)
    torch.manual_seed(C.get("seed", 0))
    rng = random.Random(C.get("seed", 0))
    out_dir = Path(C["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = a.max_steps or C["steps"]
    dev = "cuda"
    t_start = time.time()

    def set_status(phase, **kw):
        """What the run is doing right now, for a second terminal: scripts/rl_view.py <out_dir> --follow / --info."""
        (out_dir / "status.json").write_text(json.dumps({"phase": phase, "since_start_s": round(time.time() - t_start),
                                                         "time_utc": time.strftime("%H:%M:%S", time.gmtime()), **kw}))

    set_status("setup: loading vLLM")

    # ---- rollout engine first, so vLLM profiles memory before the policy and optimizer are allocated
    from src.rl.vllm_engine import VLLMEngine
    eng = VLLMEngine(C["vllm_model_dir"], gpu_memory_utilization=C.get("vllm_gpu_mem", 0.3),
                     max_model_len=C.get("max_model_len", 6144), enable_sleep_mode=True, seed=C.get("seed", 0))
    tok = eng.tok
    stop_ids = set(eng.stop_ids)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    pad_id = tok.convert_tokens_to_ids("<|fim_pad|>")
    eng.sleep()

    # ---- policy + reference
    set_status("setup: loading policy, reference model, optimizer")
    mcfg = load_config(C["model_config"]).model
    mcfg.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    state = load_state(C["init_checkpoint"])
    policy = Transformer(mcfg).to(dev)
    policy.load_state_dict(state)
    policy.train()
    ref = Transformer(mcfg).to(dev)
    ref.load_state_dict(state)
    ref = ref.eval().requires_grad_(False)  # fp32 like the policy (bf16 would also round its RoPE tables: fake KL at step 0)
    del state
    opt = torch.optim.AdamW(policy.parameters(), lr=C["lr"], betas=tuple(C.get("betas", [0.9, 0.99])),
                            weight_decay=C.get("weight_decay", 0.0), eps=1e-8)
    warm = C.get("warmup_steps", 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    lb_coef = float(load_config(C["model_config"]).model.moe.load_balance_coef)

    # ---- tasks: a fixed held-out slice per area for evals, the rest for training
    tasks = load_tasks(C["tasks"])
    by_area = defaultdict(list)
    for t in tasks:
        by_area[t["area"]].append(t)
    val, train = [], defaultdict(list)
    for area, ts in by_area.items():
        k = min(C.get("val_per_area", 40), len(ts) // 5)
        val += ts[:k]
        train[area] = ts[k:]
    weights = {ar: w for ar, w in C["area_weights"].items() if train.get(ar)}
    print(f"[tasks] train {sum(len(v) for v in train.values()):,} ({ {k: len(v) for k, v in train.items()} }), val {len(val)}", flush=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "config": C, "steps": steps, "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "task_pool_per_area": {k: len(v) for k, v in train.items()},
        "heldout_per_area": {k: sum(1 for t in val if t["area"] == k) for k in train},
        "area_weights_used": weights, "moe_backend": mcfg.moe.backend}, indent=1))

    wb = None
    if C.get("wandb_project"):
        import wandb
        wb = wandb.init(project=C["wandb_project"], name=C["run_name"], config=C)

    def evaluate(step):
        set_status(f"held-out test at step {step}: generating", step=step)
        eng.wake()
        eng.sync_weights(policy)
        g = eng.generate([render(t["messages"]) for t in val], n=1, temperature=0.0, max_tokens=C["max_tokens"])
        eng.sleep()
        set_status(f"held-out test at step {step}: scoring", step=step)
        r = score_batch(val, [x[0].text for x in g], [x[0].finished for x in g], C.get("sandbox_workers", 16))
        dump_jsonl_gz(out_dir / "eval_samples" / f"step_{step:04d}.jsonl.gz",
                      [{"step": step, "id": t["id"], "area": t["area"], "kind": t["kind"], "prompt": t["messages"],
                        "answer": x[0].text, "finished": x[0].finished, "reward": rr} for t, x, rr in zip(val, g, r)])
        res = defaultdict(list)
        for t, x in zip(val, r):
            res[t["area"]].append(x)
        res = {f"eval/{k}": sum(v) / len(v) for k, v in res.items()}
        res["eval/mean"] = sum(res.values()) / len(res)
        print(f"[eval] step {step}: " + " ".join(f"{k[5:]}={v:.3f}" for k, v in sorted(res.items())), flush=True)
        with open(out_dir / "evals.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **res}) + "\n")
        return res

    log_f = open(out_dir / "train_log.jsonl", "a")
    if C.get("eval_at_start", True):
        ev = evaluate(0)
        if wb:
            wb.log(ev, step=0)

    for step in range(1, steps + 1):
        t0 = time.time()
        areas = rng.choices(list(weights), weights=list(weights.values()), k=C["prompts_per_step"])
        batch = [rng.choice(train[ar]) for ar in areas]

        # 1-3. rollouts
        set_status(f"step {step}/{steps}: generating {len(batch) * C['n_samples']} answers (vLLM awake)", step=step)
        eng.wake()
        sync_s = eng.sync_weights(policy)
        groups = eng.generate([render(t["messages"]) for t in batch], n=C["n_samples"], temperature=C.get("temperature", 1.0),
                              top_p=C.get("top_p", 1.0), max_tokens=C["max_tokens"])
        eng.sleep()
        t_gen = time.time() - t0

        # 4. rewards
        n_code = sum(1 for t in batch if t["kind"] == "code") * C["n_samples"]
        set_status(f"step {step}/{steps}: scoring (vLLM asleep; {n_code} code answers in docker, 16 at a time)", step=step)
        flat_t = [t for t, g in zip(batch, groups) for _ in g]
        flat_r = [r for g in groups for r in g]
        rew = score_batch(flat_t, [r.text for r in flat_r], [r.finished for r in flat_r], C.get("sandbox_workers", 16))
        t_rew = time.time() - t0 - t_gen

        # 5. group advantages
        n = C["n_samples"]
        items, zero_groups = [], 0
        adv_all = [0.0] * len(flat_r)
        for gi in range(len(batch)):
            rs = torch.tensor(rew[gi * n:(gi + 1) * n])
            if rs.std() < 1e-6:
                zero_groups += 1
                continue
            adv = rs - rs.mean()
            if C.get("adv_std_norm", True):
                adv = adv / (rs.std() + 1e-4)
            for j in range(n):
                adv_all[gi * n + j] = float(adv[j])
                r = flat_r[gi * n + j]
                comp = list(r.completion_ids)
                if r.finished and (not comp or comp[-1] not in stop_ids):
                    comp.append(im_end)  # learn to stop: the stop token is part of the answer
                if not comp:
                    continue
                items.append({"seq": list(r.prompt_ids) + comp, "n_prompt": len(r.prompt_ids), "adv": float(adv[j])})

        # 6-7. loss and update
        set_status(f"step {step}/{steps}: training update on {len(items)} answers", step=step)
        stats = defaultdict(float)
        if items:
            n_tok = sum(len(it["seq"]) - it["n_prompt"] for it in items)
            for epoch in range(C.get("ppo_epochs", 1)):
                for mb in microbatches(items, C.get("micro_batch_tokens", 32768)):
                    seqs, npr = [it["seq"] for it in mb], [it["n_prompt"] for it in mb]
                    lps, lb = token_logprobs(policy, seqs, npr, dev, pad_id)
                    with torch.no_grad():
                        ref_lps, _ = token_logprobs(ref, seqs, npr, dev, pad_id)
                    loss = 0.0
                    for it, lp, rlp in zip(mb, lps, ref_lps):
                        if "old" not in it:
                            it["old"] = lp.detach()
                        ratio = torch.exp(lp - it["old"])
                        A = it["adv"]
                        pg = -torch.minimum(ratio * A, torch.clamp(ratio, 1 - C["clip_eps"], 1 + C["clip_eps"]) * A)
                        d = rlp - lp
                        kl = torch.exp(d) - d - 1          # k3 estimator, >= 0
                        loss = loss + (pg + C["kl_beta"] * kl).sum() / n_tok
                        stats["kl"] += kl.detach().sum().item() / n_tok
                        stats["entropy_proxy"] += (-lp.detach()).sum().item() / n_tok
                        stats["clip_frac"] += ((ratio - 1).abs() > C["clip_eps"]).float().sum().item() / n_tok
                    loss = loss + lb_coef * lb * len(mb) / len(items)
                    loss.backward()
                    stats["loss"] += float(loss)
                gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), C.get("grad_clip", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                stats["grad_norm"] = float(gn)
            sched.step()

        # every rollout of the step (prompt, answer, reward, advantage), for later analysis (pushed to HF with the checkpoints)
        dump_jsonl_gz(out_dir / "rollouts" / f"step_{step:04d}.jsonl.gz",
                      [{"step": step, "id": t["id"], "area": t["area"], "kind": t["kind"], "prompt": t["messages"],
                        "sample": j % n, "answer": r.text, "finished": r.finished, "n_tokens": len(r.completion_ids),
                        "reward": rw, "advantage": round(ad, 4)}
                       for j, (t, r, rw, ad) in enumerate(zip(flat_t, flat_r, rew, adv_all))])

        # 8. log
        area_r = defaultdict(list)
        for t, x in zip(flat_t, rew):
            area_r[t["area"]].append(x)
        rec = {"step": step, "lr": sched.get_last_lr()[0], "reward_mean": sum(rew) / len(rew),
               "zero_adv_groups": zero_groups / len(batch), "trained_answers": len(items),
               "finished_frac": sum(r.finished for r in flat_r) / len(flat_r),
               "answer_len": sum(len(r.completion_ids) for r in flat_r) / len(flat_r),
               "t_gen": round(t_gen, 1), "t_reward": round(t_rew, 1), "t_step": round(time.time() - t0, 1), "t_sync": round(sync_s, 2),
               **{f"reward/{k}": sum(v) / len(v) for k, v in area_r.items()},
               **{f"n/{k}": c for k, c in sorted(Counter(areas).items())}, "code_answers": n_code, **stats}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"[step {step}] reward {rec['reward_mean']:.3f} zero-adv {rec['zero_adv_groups']:.2f} kl {stats['kl']:.4f} "
              f"len {rec['answer_len']:.0f} fin {rec['finished_frac']:.2f} gn {stats.get('grad_norm', 0):.3f} "
              f"| gen {t_gen:.0f}s reward {t_rew:.0f}s step {rec['t_step']:.0f}s", flush=True)
        if wb:
            wb.log(rec, step=step)
        if step % C.get("eval_every", 25) == 0 or step == steps:
            ev = evaluate(step)
            if wb:
                wb.log(ev, step=step)
        if step % C.get("save_every", 100) == 0 or step == steps:
            torch.save({"model": policy.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                        "step": step, "rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state(),
                        "config": C, "rl_info": {"init_checkpoint": C["init_checkpoint"], "tasks": C["tasks"],
                                                 "run_name": C["run_name"]}},
                       out_dir / f"step_{step}.pt")
            print(f"[save] {out_dir}/step_{step}.pt", flush=True)
    set_status("finished", step=steps)


if __name__ == "__main__":
    main()
