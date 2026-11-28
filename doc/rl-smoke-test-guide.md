# RL smoke test: what runs when, and how to watch it

A hands-on guide to the 20-step Phase 4 GRPO smoke test (27 Nov): the order of everything, what one step does, which environments exist, and how to watch it live from a second terminal. The method itself (GRPO, checkers, why RL) is in [rl-primer-phase4.md](rl-primer-phase4.md); the task list and status in [plan-06-phase-4-rl.md](plan-06-phase-4-rl.md).

**Purpose:** test the RL pipeline end to end on Phase 3 run 01 (step 1,500) before RL v1. Nothing here decides the RL start model; that comes from the run 01 vs run 02b evals.

| | |
|---|---|
| Machine | fire01 (H100) |
| Started by | `scripts/rl_smoke_chain.sh` with `START_AFTER=train`: right after Phase 3 run 02b training exits, **alongside the run 02b evals** on the same GPU |
| Config | `configs/rl_smoke.yaml` (= `rl_v1.yaml` with 20 steps, test every 10, checkpoint at 20, W&B run `Phase4_rl_smoke`); overrides from the chain: run 01 weights, the filtered task file, `vllm_gpu_mem=0.25`, `micro_batch_tokens=16384` (so both jobs fit in 80 GB) |
| Output folder | `/lambda/nfs/h100-f1/checkpoints/Phase4_rl_smoke/` |
| Log | `/lambda/nfs/h100-f1/train_rl_smoke.log`; chain log `/lambda/nfs/h100-f1/rl_smoke_chain.log` |

---

## 1. The evening's order

| # | What | Where | Starts (IST) | Takes |
|---|---|---|---|---|
| 1 | SFT run 02b training | H100 | running | until ~19:10 |
| 2 | Difficulty filter (run 01 answers the 38k RL tasks 8 times each) | A10 | running | until ~20:30 |
| 3 | Run 02b checkpoints → HF (training-states repo + runnable model branch) | A10 | ~19:15 | ~30 min |
| 4 | Run 02b evals (refusal check, chat test, HumanEval/MBPP, GSM8K, IFEval, BFCL) | H100 | ~19:10 | ~1.5–2 h |
| 5 | vLLM environment setup on fire01, then **this smoke test** | H100, alongside #4 | ~19:15 + ~15 min setup | ~30 min |
| 6 | Report: run 01 vs run 02b, smoke result, pass@1/pass@8 map → the user picks the RL start model and gives the go | — | ~21:00–21:30 | — |
| 7 | If run 02b is chosen: rerun the difficulty filter for it (H100, vLLM) | H100 | after the go | ~30–40 min |
| 8 | RL v1 (500 steps) overnight; key checkpoints + all rollouts → HF | H100 | after #7 | ~6 h |

The smoke test uses the tasks the A10 filter has kept **so far** (several thousand by 19:15); for 20 steps that's plenty.

---

## 2. Inside the smoke test, in order

| # | What happens | Time (approx.) |
|---|---|---|
| 1 | **Setup:** load vLLM (fast answer generation), then the policy model, the frozen reference model and the optimizer | ~2–3 min |
| 2 | **Load tasks;** set aside up to 40 per subject as a held-out test set (never trained on) | seconds |
| 3 | **Step-0 test:** one greedy answer per held-out question, scored: the "before RL" score per subject | ~1–2 min |
| 4 | **Training steps 1–20** (§3) | ~30–40 s each (slower while the evals share the GPU) |
| 5 | **Test again at steps 10 and 20** on the same held-out questions | ~1–2 min each |
| 6 | **Save `step_20.pt`**; status becomes `finished` | ~1 min |

There is **no subject-by-subject order**: every step mixes all subjects (training one subject at a time makes the model forget the others). Planned share of the 64 prompts per step:

| Subject | Share | ≈ prompts/step | over 20 steps |
|---|---|---|---|
| Maths | 20% | 13 | ~256 |
| Instruction following | 18% | 12 | ~230 |
| Code | 15% | 10 | ~192 |
| Refusal balance | 13% | 8 | ~166 |
| Time problems | 12% | 8 | ~154 |
| Tool calls | 12% | 8 | ~154 |
| Who did what | 10% | 6 | ~128 |

Each step draws at random with these shares, so the actual counts vary a little; the step table shows them.

---

## 3. One step

```
 0 s            ~15 s              ~20 s                ~32 s
 │──────────────│──────────────────│────────────────────│
 │ wake vLLM,   │ SCORE 512 answers│ TRAIN (vLLM asleep) │ → next step
 │ sync weights │  • ~430 in Python│  • compare 8 answers│
 │ (0.5 s),     │    (instant)     │    per question     │
 │ generate 512 │  • ~80 code →    │  • one update to the│
 │ answers      │    docker, 16 at │    model            │
 │              │    a time        │  • save the 512     │
 │              │    (~5–10 s)     │    answers to disk  │
```

1. Pick 64 questions at random with the shares above.
2. Copy the policy's current weights into vLLM (0.5 s).
3. vLLM writes **8 answers per question**: 512 answers, all subjects in one batch. Writing is the same for every subject: it's all text.
4. **Score all 512, in the same step, before training** (the update needs the scores). Each answer goes to its subject's checker (§4); every checker returns a number from 0 to 1.
5. Within each question, compare the 8 scores with their average (the advantage); questions where all 8 scored the same teach nothing and are skipped.
6. **One training update for all subjects together:** push better-than-average answers up and worse ones down, stay close to the SFT model (KL), one optimizer step on the one shared model.
7. Log the numbers; save all 512 answers with their rewards and advantages to `rollouts/step_00NN.jsonl.gz`.

The next step starts with the updated model and fresh questions.

---

## 4. Environments: how many, and when they're up

**v1 has one real environment: the docker sandbox for code.** Everything else is a function inside the trainer process.

| Component | What it is | Up when? |
|---|---|---|
| **Trainer** (`scripts/train_grpo.py`) | one Python process: policy, reference model, optimizer, all Python checkers | the whole run |
| **vLLM engine** | inside the trainer process; writes the answers | **awake** only while generating; **asleep** during training (memory freed) |
| **Python checkers** (maths, time, names, instruction constraints, tool-call JSON, refusal) | functions in `src/rl/rewards.py`; nothing to start | each step's scoring phase |
| **Docker sandbox** (code only) | one **disposable container per code answer**, started and deleted automatically | only during scoring: **up to 16 at once**, each ~1–2 s |
| **Run 02b evals** | a separate job on the same H100 | in parallel the whole time |

```
time ─────────────────────────────────────────────────────────────────────────────────────────►
       0:00        0:03       0:05            each step ≈ 30–40 s  (×20)                ≈0:20  ≈0:22
        │ setup     │ step-0   │ step 1          step 2          …  step 10 + test …  step 20 │ save │
GPU     │█ load    │▓▓ vLLM   │▓▓▓▓░░░░████  ▓▓▓▓░░░░████  …                                │█ ckpt│
CPU     │          │ ▒ score  │     ▒▒▒▒          ▒▒▒▒       ← docker bursts (code answers)  │      │
docker  │  0       │ ≤16      │  0  ≤16  0        ≤16   0    containers alive at once       │  0   │
evals   │═══════════════════════════════════════════════════════════════════════════════════════│
```
▓ vLLM awake, generating · ░ scoring (vLLM asleep) · █ training update · ▒ code containers · ═ the parallel eval job

**Why mixing subjects works with a code environment:** the environment is used only at **scoring** time, not while writing. Nothing is set up per subject or per step: the checkers are loaded once, the docker image is pulled once during setup, and each code answer simply gets its own fresh container (one bad program can't affect another).

**What a code container does:** the trainer takes the code out of the reply, appends the task's tests (MBPP asserts, or a small runner for SYNTHETIC-1 input/output cases), and runs

```
sudo docker run --rm -i --network none --cpus 1 --memory 512m --pids-limit 64 --read-only \
     --user nobody --cap-drop ALL ... python:3.11-slim python3 -I -        (program on stdin, 10 s limit)
```

Exit code 0 (all tests pass) = reward 1; any error, failed test or timeout = 0. The container is deleted at once (`--rm`). It's the same sandbox that scored all our HumanEval/MBPP evals.

**Not in v1:** library tasks (`transformers`, `torch`, … need a custom image with tiny offline models) and **multi-turn** environments (call a tool, read the real result, continue; or run code, see the error, fix it), where generation pauses for the environment mid-answer. Both are planned for RL v2.

---

## 5. Watching it from a second terminal

SSH into fire01 (host and key are in your private notes), then:

```bash
cd ~/moe0.9b-a0.2b
D=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_smoke
```

| Want to see | Command |
|---|---|
| Has it started? What phase is the chain in? | `tail -f /lambda/nfs/h100-f1/rl_smoke_chain.log` |
| **Settings, task mix, task pool, and what it's doing right now** | `python3 scripts/rl_view.py $D --info` |
| **Live:** current phase + step table, refreshed every 10 s | `python3 scripts/rl_view.py $D --follow` (Ctrl-C to stop) |
| The raw training log (one line per step, test results, saves) | `tail -f /lambda/nfs/h100-f1/train_rl_smoke.log` |
| **The actual answers of a step**, with rewards and advantages | `python3 scripts/rl_view.py $D --step 5 --area time --groups 3` |
| Held-out test answers, before vs after | `python3 scripts/rl_view.py $D --eval 0 --area tool` / `--eval 20` |
| The code containers appearing and disappearing | `watch -n 1 'sudo docker ps --filter name=sbx- --format "{{.Names}}  {{.RunningFor}}"'` |
| GPU memory and load (smoke + evals together) | `watch -n 2 nvidia-smi` |
| Charts | W&B project `moe0-9b-a0-2b`, run `Phase4_rl_smoke` |

`--step` options: `--area` (maths, time, code, tool, binding, ifeval, refusal), `--groups N` (questions shown), `--chars N` (answer length shown), `--seed N` (another random pick). Each question shows all 8 answers sorted best first:

```
[time/time] time-abc   rewards [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
PROMPT: I arrived at 2:00, 30 minutes late. When did it start?
  reward 1.00  adv +1.20    12 tok  | It started at 1:30.
  reward 0.00  adv -0.60    12 tok  | 2 + 30 = 32
```

`reward` is the checker's score; `adv` (advantage) is how strongly training pushes that answer **up (+)** or **down (−)**.

### Files in the output folder

| File | Contents | Written |
|---|---|---|
| `status.json` | the current phase (setup / test / generating / scoring / training / finished), time since start | continuously |
| `run_config.json` | all settings, start model, task file, task pool and held-out size per subject | after setup |
| `train_log.jsonl` | one line per step (the numbers in §6) | every step |
| `evals.jsonl` | held-out test score per subject | steps 0, 10, 20 |
| `rollouts/step_NNNN.jsonl.gz` | every answer of the step: question, subject, task ID, answer, finished, length, reward, advantage | every step |
| `eval_samples/step_NNNN.jsonl.gz` | every held-out test answer with its score | steps 0, 10, 20 |
| `step_20.pt` | the checkpoint (weights, optimizer, scheduler, RNG, config, start model) | end |

---

## 6. Reading the numbers

| Column (step table) | Meaning | Healthy | Warning sign |
|---|---|---|---|
| `reward` | average score of the 512 answers | creeps up (noisy over 20 steps) | falls steadily |
| `0-adv` | share of questions where all 8 answers scored the same (no learning signal) | well below 0.5 (the filter keeps it down) | near 1: nothing to learn from |
| `kl` | drift from the SFT model | starts ~0, grows slowly | sudden jumps |
| `len` | average answer length (tokens) | stable | keeps growing (rambling) |
| `fin` | share of answers that ended properly | near 1 | low: answers cut off at 1,024 tokens |
| `gnorm` | size of the update | steady | spikes (instability) |
| `gen s / rew s / step s` | seconds generating / scoring / whole step | ~10–25 / ~5–10 / ~30–40 (more while the evals share the GPU) | far above: a bottleneck |
| subject columns `0.42(13)` | that subject's average reward this step (prompts this step) | individual subjects move up | one subject collapses |

**What the smoke test must show** before RL v1: every phase runs (setup, tests, all 20 steps, save); rewards from all seven subjects including code (the sandbox works on fire01); `0-adv` well below 1; KL small and finite; no crash or out-of-memory with the evals alongside; step time close to the estimate. **Spot-check rewards by reading rollouts** (`--step`): a wrong answer with reward 1, or a right one with 0, means a checker needs fixing before the long run.

After the smoke test, the report goes to the user; RL v1 starts only on the user's go.
