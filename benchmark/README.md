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

## How to reproduce

```bash
pip install -e ".[eval]"
python scripts/run_benchmarks.py \
    --checkpoint /path/to/step_N.pt \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --output benchmark/step_N.json
```
