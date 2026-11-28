# Benchmark results

Standard base-LM benchmark suite via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`src/eval/lm_eval_adapter.py` + `scripts/run_benchmarks.py` — see those files for
how a non-`transformers`-native architecture is wired into the harness).

This page covers **Phases 1 and 2** (base models), scored with loglikelihood
multiple-choice tasks only. Later phases have their own summaries:

| Phase | Results |
|---|---|
| 2.5a (knowledge, relations) | [`phase25a/summary.md`](phase25a/summary.md) |
| 2.5b (16k context: passkey, 2-hop retrieval) | [`phase25b/summary.md`](phase25b/summary.md) |
| 3 SFT (chat, refusal check, hand test) | [`phase3/`](phase3/) (run 01 hand test: [`chat_smoke_phase3_step1500.md`](phase3/chat_smoke_phase3_step1500.md)) |
| 3 SFT runs 01 / 02b vs 4 RL v1 (chat evals, BFCL, IFEval, GSM8K, English 6) | [`phase4/summary.md`](phase4/summary.md) |
| Pass@8 (8 samples, T=1.0): SFT run 02b vs RL v1 step 500, skill vs knowledge gaps | [`phase4/passk/summary.md`](phase4/passk/summary.md) |
| Hand tests vs OLMo-2 1B, binding levels | [`handtest/`](handtest/), [`binding_levels/report.md`](binding_levels/report.md) |

## step_7000 (2026-11-20)

Checkpoint: `step_7000.pt` — step 7,000 / 19,073 planned steps (≈37% through
the single 10B-token FineWeb-Edu pretraining run), loss ≈2.8 at this point.
Raw results: [step_7000.json](step_7000.json).

| Task | Metric | Score | Random baseline |
|---|---|---|---|
| PIQA | acc_norm | 64.7% | 50% (2-choice) |
| ARC-Easy | acc_norm | 51.1% | ≈25% (multi-choice) |
| HellaSwag | acc_norm | 37.0% | 25% (4-choice) |
| LAMBADA (openai) | acc (exact last-word match) | 29.0% | ≈0% (strict exact-match) |
| LAMBADA (openai) | perplexity | 58.1 | — |
| WinoGrande | acc | 52.4% | 50% (2-choice) |
| ARC-Challenge | acc_norm | 28.0% | 25% (4-choice) |

**Read:** tasks that reward broad surface-pattern learning (PIQA, ARC-Easy,
HellaSwag) are clearly above chance already. LAMBADA's strict exact-match
accuracy (29%) is a genuinely positive signal of real language-modeling
quality this early. The two hardest, most reasoning-heavy benchmarks
(ARC-Challenge, WinoGrande) sitting near chance is expected at this model
size (≈915M total / ≈211M active) and training stage — those benchmarks are
designed to be hard even for much larger, fully-trained models — not a sign
of a problem.

This is the first of what should become a series of checkpoints evaluated
the same way, to track real improvement over the course of training rather
than judging a single number in isolation.

## step_8000 (2026-11-20)

Checkpoint: `step_8000.pt` — step 8,000 / 19,073 planned steps (≈42% through
the run). Raw results: [step_8000.json](step_8000.json).

| Task | Metric | step_7000 | step_8000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 64.7% | 64.5% | -0.2 (noise) |
| ARC-Easy | acc_norm | 51.1% | 50.7% | -0.4 (noise) |
| HellaSwag | acc_norm | 37.0% | 38.0% | +1.0 |
| LAMBADA (openai) | acc | 29.0% | 31.0% | **+2.0** |
| LAMBADA (openai) | perplexity | 58.1 | **50.3** | **-7.8 (better)** |
| WinoGrande | acc | 52.4% | 51.7% | -0.7 (noise) |
| ARC-Challenge | acc_norm | 28.0% | 27.6% | -0.4 (noise) |

**Read:** most metrics are flat within statistical noise over just 1,000
steps (stderr on most of these tasks is ≈1-1.4%, so ±1 point swings aren't
meaningful). LAMBADA is the exception — a real, consistent improvement
(perplexity down 13%, accuracy up 2 points). This is the metric flagged in
the `step_7000` comparison-against-other-models analysis as most dependent
on total tokens seen, so seeing it move first and most clearly as training
continues is a good, expected signal rather than a surprise.

## step_9000 (2026-11-20)

Checkpoint: `step_9000.pt` — step 9,000 / 19,073 planned steps (≈47% through
the run). Raw results: [step_9000.json](step_9000.json).

| Task | Metric | step_8000 | step_9000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 64.5% | **65.9%** | **+1.4** |
| ARC-Easy | acc_norm | 50.7% | 50.8% | +0.1 (noise) |
| HellaSwag | acc_norm | 38.0% | 38.5% | +0.5 |
| LAMBADA (openai) | acc | 31.0% | 30.6% | -0.4 (noise) |
| LAMBADA (openai) | perplexity | 50.3 | **49.3** | -1.0 (better) |
| WinoGrande | acc | 51.7% | 51.5% | -0.2 (noise) |
| ARC-Challenge | acc_norm | 27.6% | 27.2% | -0.4 (noise) |

**Read:** PIQA posts the clearest real gain this step (+1.4, above the
≈1.1% stderr noise floor on that task) alongside HellaSwag's continued
slow climb. LAMBADA perplexity keeps improving (now down 15% from the
step_7000 baseline of 58.1), reinforcing the step_8000 read that it's the
metric most sensitive to additional tokens seen. Everything else remains
flat within stderr — consistent with a model still mid-run rather than any
regression.

## step_18000 (2026-11-21)

Checkpoint: `step_18000.pt` — step 18,000 / 19,073 planned steps (≈94% through
the run), ≈9.44B of 10B planned tokens seen. Raw results:
[step_18000.json](step_18000.json). This jumps 9,000 steps ahead of the
last logged checkpoint (`step_9000`), so deltas below reflect ≈4.7B
additional tokens of training, not the usual 1,000-step increment.

| Task | Metric | step_9000 | step_18000 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 65.9% | **68.8%** | **+2.9** |
| ARC-Easy | acc_norm | 50.8% | **56.6%** | **+5.8** |
| HellaSwag | acc_norm | 38.5% | **42.6%** | **+4.1** |
| LAMBADA (openai) | acc | 30.6% | **35.3%** | **+4.7** |
| LAMBADA (openai) | perplexity | 49.3 | **31.0** | **-18.3 (better)** |
| WinoGrande | acc | 51.5% | 50.7% | -0.8 (noise) |
| ARC-Challenge | acc_norm | 27.2% | **29.9%** | **+2.7** |

**Read:** every task except WinoGrande posts a move well clear of its
stderr (≈1-1.4% on most tasks) — the clearest sign yet of real learning
rather than noise, consistent with nearly doubling total tokens seen since
the last checkpoint. LAMBADA perplexity in particular has fallen 37% from
the step_7000 baseline (58.1 → 31.0) and its accuracy is up 6.3 points
(29.0% → 35.3%), reaffirming it as the most token-sensitive metric in this
suite. ARC-Easy (+5.8) and HellaSwag (+4.1) also show their largest single
jumps of the run so far. WinoGrande remains flat within noise — expected,
since it is one of the two hardest, most reasoning-heavy tasks in this
suite (along with ARC-Challenge, which did move here) and is known to need
much more scale/training to shift meaningfully.

## step_19073 (2026-11-21) — Phase 1 final checkpoint

Checkpoint: `step_19073.pt` — step 19,073 / 19,073 planned steps (**100% —
Phase 1 complete**), full 10.0B-token FineWeb-Edu pretraining run finished.
Raw results: [step_19073.json](step_19073.json).

| Task | Metric | step_18000 | step_19073 | Δ |
|---|---|---|---|---|
| PIQA | acc_norm | 68.8% | 68.2% | -0.6 (noise) |
| ARC-Easy | acc_norm | 56.6% | 56.1% | -0.5 (noise) |
| HellaSwag | acc_norm | 42.6% | 42.9% | +0.3 (noise) |
| LAMBADA (openai) | acc | 35.3% | 34.8% | -0.5 (noise) |
| LAMBADA (openai) | perplexity | 31.0 | 32.0 | +1.0 (noise) |
| WinoGrande | acc | 50.7% | 51.4% | +0.7 (noise) |
| ARC-Challenge | acc_norm | 29.9% | 29.8% | -0.1 (noise) |

**Read:** every metric is flat within stderr over the final 1,073 steps —
expected, since the LR schedule had already decayed to its floor
(`min_lr_ratio: 0.1`) well before this point, so these last steps make very
small, low-magnitude updates. This is the final Phase 1 base-pretraining
checkpoint; continued improvement from here requires Phase 2's continual
training on the expanded English + code + agentic mixture
(`doc/plan-02-phase-2-continual-training.md`), not more steps on the same
10B-token FineWeb-Edu-only run.

## Phase 2 — complete (step 22,888, 12.0B tokens)

Phase 2 continues from `step_19073` on English + code + agentic +
contrastive data (12B tokens). Raw results are in [`phase2/`](phase2/), one
JSON per eval per checkpoint. How each eval is scored is in
[`doc/eval-notes.md`](../doc/eval-notes.md#phase-2-evaluation-continual-training-english--code--agentic--contrastive).
All evals are deterministic (bit-identical across reruns). On Hugging Face: `main` holds step 19,000 (end of the plateau); step 9,000 is on branch `phase2-step9000` and Phase 1 final on `phase1-final`. Full training states (weights + optimizer + schedulers + RNG) for Phase 1 final, Phase 2 step 19,000 and Phase 2 final are in [`bubblspace-moe0.9b-a0.2b-training-states`](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states).

| Eval | Phase 1 | P2 1k | 3k | 5k | 7k | 9k | 11k | 13k | 15k | 17k | 19k | 21k | **22,888** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Agentic: valid call, name = gold (23 eps) | 0% | 17% | 39% | 30% | 48% | 30% | 26% | 35% | 43% | 35% | 35% | 39% | **39%** |
| Reasoning: closes `<think>` ≤512 tok | 0% | 0% | 0% | 0% | 0% | 12% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| FIM exact match (60 lines) | 0% | 0% | 0% | 3.3% | 5.0% | 1.7% | 3.3% | 1.7% | 1.7% | 3.3% | 3.3% | 3.3% | **5.0%** |
| Binding probe v2 (/100) | 24 | 25 | 25 | 25 | 28 | 24 | 24 | 22 | 29 | 28 | 30 | 35 | **37** |
| Binding probe v3 (/70) | 4 | 6 | 6 | 5 | 6 | 3 | 1 | 4 | 4 | 4 | 4 | 6 | 6 |
| Binding probe v4 (/112, chance ≈ 28) | 13 | 21 | — | — | — | 11 | 17 | 13 | 24 | 18 | 17 | **82** | 77 |
| Routing: mean load CV | 0.623 | 0.385 | 0.318 | 0.293 | 0.285 | 0.274 | 0.264 | 0.260 | 0.252 | 0.252 | 0.246 | 0.229 | **0.210** |

(Steps 2k, 4k and 6k are in `phase2/`; omitted here for width.)

**English forgetting gate** (6-task, `phase2/english6_phase2_step*.json`):

| Task | Phase 1 | 5k | 7k | 9k | 11k | 13k | 15k | 17k | 19k | 21k | **22,888** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HellaSwag | 42.86 | 40.39 | 40.69 | 40.64 | 40.28 | 40.61 | 40.65 | 40.72 | 40.26 | 41.07 | 41.74 |
| PIQA | 68.17 | 67.30 | 67.52 | 67.90 | 67.19 | 67.57 | 67.46 | 66.92 | 66.87 | 68.82 | 67.85 |
| ARC-Easy | 56.10 | 52.78 | 51.18 | 52.57 | 53.20 | 52.90 | 51.64 | 52.15 | 55.09 | 54.92 | 54.12 |
| ARC-Challenge | 29.78 | 27.99 | 27.39 | 27.99 | 27.99 | 29.18 | 27.90 | 29.61 | 29.18 | 29.10 | 28.58 |
| WinoGrande | 51.38 | 51.07 | 51.70 | 52.72 | 52.57 | 51.78 | 50.67 | 53.75 | 52.25 | 52.96 | **54.30** |
| LAMBADA | 34.80 | 32.85 | 33.03 | 35.30 | 34.54 | 34.02 | 32.08 | 33.96 | 32.21 | 33.01 | **34.91** |
| **Average** | **47.18** | 45.40 | 45.25 | 46.19 | 45.96 | 46.01 | 45.07 | 46.18 | 45.98 | 46.65 | **46.92** |

| Milestones | Phase 1 final | P2 step 5,000 | P2 step 19,000 | **P2 step 22,888** |
|---|---|---|---|---|
| CRUXEval-O pass@1 | 0.0% | 8.5% | 17.7% (98% parsable) | **20.7%** (96% parsable) |
| BFCL simple / multiple / parallel (AST, decontaminated) | 0 / 0 / 0% | 9 / 6 / 0% | 25 / 13 / 0% | **25 / 17 / 1%** |

**Code generation with execution** (`scripts/eval_code_gen.py` + `scripts/sandbox_exec.py`; greedy, deterministic; every program runs in its own network-less, non-root, read-only container with CPU/memory/process/10 s limits):

| | Phase 1 final | P2 step 9,000 | P2 step 19,000 | **P2 step 22,888** |
|---|---|---|---|---|
| HumanEval pass@1 (164) | 0/164 = 0.0% | 15/164 = 9.1% | 19/164 = 11.6% | **24/164 = 14.6%** |
| MBPP pass@1 (500, full test split) | not run (no code in Phase 1 training) | 42/500 = 8.4% | 63/500 = 12.6% | **91/500 = 18.2%** |

Raw: `phase2/codegen_*.json` (completions + assembled programs) and `phase2/codeexec_*.json` (per-task pass/fail). The step-19k HumanEval run was repeated on the H100 and passed exactly the same 19 problems. Contamination check vs the Python training pools: passed problems are not more overlapping than failed ones (HumanEval 14% vs 23%, MBPP 24% vs 20%; `phase2/contamination_code.json`).

**Closed-book knowledge** (added for Phase 2.5; greedy, deterministic):

| | Phase 1 final | P2 22,888 |
|---|---|---|
| Simple-facts probe (95) | 81/95 | 82/95 |
| TriviaQA (5-shot, first 2,000) | 8.5% | 11.6% |
| NQ-open (5-shot, 3,610) | 2.9% | 3.9% |

**Read:**
- **Code and tool calling grew through the plateau and again in the anneal.** CRUXEval-O 8.5% → 17.7% → 20.7%. HumanEval 9.1% → 11.6% → 14.6%. MBPP 8.4% → 12.6% → 18.2%. BFCL multiple 6% → 13% → 17%.
- **English recovered in the anneal**: 45.98 → 46.92, within 0.3 of Phase 1. WinoGrande (54.3) and LAMBADA (34.9) are now above Phase 1.
- **Binding transferred to new words, not to new structures.** Probe v4 went 17 → 82 when contrastive v2 entered (77 at the final step). Probe v3 stayed at 6. Phase 2.5a's contrastive v3 targets structure.
- **Facts barely moved**: TriviaQA 8.5% → 11.6%, NQ 2.9% → 3.9%. That's the focus of Phase 2.5a.
- **Still near zero:** BFCL parallel calls (1%) and closing `</think>`.
- **Routing at its most even:** load CV 0.62 → 0.21, with no starved or overloaded experts.

## How to reproduce

```bash
pip install -e ".[eval]"
python scripts/run_benchmarks.py \
    --checkpoint /path/to/step_N.pt \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --output benchmark/step_N.json
```
