# Benchmark results

Standard base-LM benchmark suite via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`src/eval/lm_eval_adapter.py` + `scripts/run_benchmarks.py` — see those files for
how a non-`transformers`-native architecture is wired into the harness).

Only loglikelihood-scored, multiple-choice tasks are used — appropriate for a
**base-pretrained, non-instruction-tuned model**. Chat/instruction-following
benchmarks (MMLU-chat-style, open-ended QA scoring) would not be meaningful
here since this model has had no SFT.

## step_7000 (2026-11-20)

Checkpoint: `step_7000.pt` — step 7,000 / 19,073 planned steps (~37% through
the single 10B-token FineWeb-Edu pretraining run), loss ≈2.8 at this point.
Raw results: [step_7000.json](step_7000.json).

| Task | Metric | Score | Random baseline |
|---|---|---|---|
| PIQA | acc_norm | 64.7% | 50% (2-choice) |
| ARC-Easy | acc_norm | 51.1% | ~25% (multi-choice) |
| HellaSwag | acc_norm | 37.0% | 25% (4-choice) |
| LAMBADA (openai) | acc (exact last-word match) | 29.0% | ~0% (strict exact-match) |
| LAMBADA (openai) | perplexity | 58.1 | — |
| WinoGrande | acc | 52.4% | 50% (2-choice) |
| ARC-Challenge | acc_norm | 28.0% | 25% (4-choice) |

**Read:** tasks that reward broad surface-pattern learning (PIQA, ARC-Easy,
HellaSwag) are clearly above chance already. LAMBADA's strict exact-match
accuracy (29%) is a genuinely positive signal of real language-modeling
quality this early. The two hardest, most reasoning-heavy benchmarks
(ARC-Challenge, WinoGrande) sitting near chance is expected at this model
size (~915M total / ~211M active) and training stage — those benchmarks are
designed to be hard even for much larger, fully-trained models — not a sign
of a problem.

This is the first of what should become a series of checkpoints evaluated
the same way, to track real improvement over the course of training rather
than judging a single number in isolation.

## step_8000 (2026-11-20)

Checkpoint: `step_8000.pt` — step 8,000 / 19,073 planned steps (~42% through
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
steps (stderr on most of these tasks is ~1-1.4%, so ±1 point swings aren't
meaningful). LAMBADA is the exception — a real, consistent improvement
(perplexity down 13%, accuracy up 2 points). This is the metric flagged in
the `step_7000` comparison-against-other-models analysis as most dependent
on total tokens seen, so seeing it move first and most clearly as training
continues is a good, expected signal rather than a surprise.

## step_9000 (2026-11-20)

Checkpoint: `step_9000.pt` — step 9,000 / 19,073 planned steps (~47% through
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
~1.1% stderr noise floor on that task) alongside HellaSwag's continued
slow climb. LAMBADA perplexity keeps improving (now down 15% from the
step_7000 baseline of 58.1), reinforcing the step_8000 read that it's the
metric most sensitive to additional tokens seen. Everything else remains
flat within stderr — consistent with a model still mid-run rather than any
regression.

## step_18000 (2026-11-21)

Checkpoint: `step_18000.pt` — step 18,000 / 19,073 planned steps (~94% through
the run), ~9.44B of 10B planned tokens seen. Raw results:
[step_18000.json](step_18000.json). This jumps 9,000 steps ahead of the
last logged checkpoint (`step_9000`), so deltas below reflect ~4.7B
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
stderr (~1-1.4% on most tasks) — the clearest sign yet of real learning
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

## How to reproduce

```bash
pip install -e ".[eval]"
python scripts/run_benchmarks.py \
    --checkpoint /path/to/step_N.pt \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --output benchmark/step_N.json
```
