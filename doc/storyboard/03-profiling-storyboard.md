# Storyboard 03 — Phase 2 pre-flight: smoke tests, profiling, and the prefetch fix

This covers the window between "Phase 1 is done, start Phase 2" and the Phase 2 launch itself: building the Phase 2 training code, the first smoke test, a profiling pass that contradicted my guess, one fix, and a second smoke test. Chronological, including the mistakes.

## 1. The ask

Phase 1 finished at `step_19073` (10B FineWeb-Edu tokens, ~87k tok/s, ~32h on one H100). You said: start Phase 2 on `fire01`. The plan (`doc/plan-02-phase-2-continual-training.md`, revised after the entity-binding probe work in [`doc/review-01-pre-phase2-semantic-diagnosis.md`](../review-01-pre-phase2-semantic-diagnosis.md)) was 12B tokens, continuing from `step_19073` with a two-stage WSD schedule.

But the training code to do that didn't exist yet. `scripts/train.py` had no way to continue a *finished* run (`--resume` would exit immediately because `start_step == total_steps`), the data loader read a single pool with uniform random windows, and attention had no document mask. So Phase 2 started with writing code, not launching it.

## 2. What got built

All of it is opt-in: a config without a `phase2:` block runs the original Phase 1 path, unchanged.

- **Document mask** (`src/model/attention.py`): `build_doc_block_mask` derives a document id per token from `<|endoftext|>` positions and passes a causal + same-document `BlockMask` to `flex_attention`. With no mask passed, the original `F.scaled_dot_product_attention(is_causal=True)` path runs.
- **Multi-pool dataset** (`src/data/dataset.py`, `MultiPoolTokenDataset`): 16 named pools (English/STEM, code, agentic, contrastive). Each training row is drawn from a pool by stage-dependent group weights. Agentic pools are read **block-aligned** so every sample starts at an episode start, with padding targets set to −100. Each pool holds out the tail of its last shard for a fixed per-pool validation set.
- **Window-level FIM** (`src/data/fim.py`, `apply_window_fim`): 50/50 PSM/SPM, applied to one document span *inside* an already-sampled 2048-token window, so all three FIM sentinels always land in the same window. Phase 1's file-level version split them across windows for any file longer than 2048 tokens.
- **Phase 2 schedule** (`src/optim/schedule.py`): re-warm 6e-5 → 2.4e-4 over 300 steps, stable through step 19,073, linear decay to 6e-6 by step 22,888.
- **`train.py`**: `--init_from` loads weights plus Muon/AdamW optimizer state but starts at step 0 with the new schedule; `ignore_index=-100`; per-pool validation every 250 steps; W&B `global_step` continuing from 19,073.
- **Contrastive corpus** (`scripts/gen_contrastive.py`): 61.9M unique tokens of minimal-pair documents targeting the probe's zero categories, with probe entity pairs and probe names excluded.
- **Tests** (`tests/test_phase2.py`): doc-mask isolation, FIM invariants, block alignment and pad masking, stage mixture, schedule boundaries.

The most important test: perturbing document A's tokens leaves document B's logits **bit-for-bit unchanged** with the mask, and visibly changed without it (so the test can actually fail). With one document per row, masked output matches the Phase 1 causal path.

### Bug #1 — a pool-weight override that was never read

The config allows a per-pool `weight` inside a group (Wikipedia 1.5, Cosmopedia 1.5, FineMath 1.0, FineWeb 0.8), but `_Pool` never stored it; the sampler looked up an attribute that was never set and silently fell back to token-count weighting. Caught on re-read before any test ran. **Fix**: `_Pool` stores `spec.get("weight")` and the sampler uses it.

### Bug #2 — divide by zero when a stage empties a group

The new stage-mixture test zeroed every pool in a group with a stage `boost`, and the sampler divided by zero. The real config never does this (in Stage 2B, FineWeb is boosted to 0 but Wikipedia, Cosmopedia and FineMath remain), but it's a real edge case. **Fix**: skip a group whose total weight is zero.

## 3. Data, and two false alarms

The revised plan added Wikipedia (1.5B tokens) and FineMath (1.0B) to the English pool. Both were tokenized in parallel on the A10.

### False alarm — "unsafe" files that were safe

Following the malware-scan practice from Storyboard 02, I scanned both datasets with `HfApi().get_paths_info(..., expand=True)`. It reported 5 of 41 Wikipedia files and 13 of 64 FineMath files as unsafe. Inspecting the individual files showed `safe=True, status='safe'`. The bug was mine: `BlobSecurityInfo` subclasses `dict`, so my `isinstance(s, dict)` branch read the wrong field. Re-checked with attribute access: **all 105 files clean.** Lesson: when a scan result contradicts itself, inspect one raw record before acting on the aggregate.

### Contrastive repetition

The first contrastive test run was only 78% unique at 20k documents. The physical-fit template family has only ~80 possible distinct documents and would have repeated thousands of times. **Fix**: the generator writes only unique documents; repetition is controlled in one place, the pool's sampling weight (lowered to 1.5% in Stage 2A / 6% in 2B, about 4.4× total).

## 4. Why a smoke test before a ~40h run

Phase 2 is roughly two days of H100 time. A crash at hour 30, or a silent bug that wastes the whole run, costs far more than 100 steps. The smoke test (`configs/phase2_smoke.yaml`: same code path, 100 steps, short warmup, separate W&B run) answers the correctness questions cheaply:

| Check | What it guards against | Smoke #1 result |
|---|---|---|
| `--init_from` loads | wrong weights / reset optimizer → loss spike | ✅ first English val loss 2.52, consistent with Phase 1's ~2.6 |
| `flex_attention` under `torch.compile` | graph breaks, crashes, compile blowups | ✅ compiled, no breaks |
| Stability through re-warm | LR shock, divergence | ✅ `grad_norm` 2.84 → ~0.19 by step 80 |
| Checkpoint save | losing hours of training to a save bug | ✅ |
| Throughput | a run that takes far longer than planned | ⚠️ **~69–71k tok/s** |

Everything passed except the one number that decides wall-clock: at ~70k tok/s, 12B tokens is **~48h**, not the 38–39h planned. Smoke #1 also flagged two things to watch during the real run: MoE load-balance loss ~1.15 (vs 1.00 in Phase 1, the router adapting to code/agentic tokens), and contrastive val loss dropping 3.87 → 0.70 in 50 steps (template predictability, not memorization, since validation is held out).

## 5. Why profile, and why my guess was wrong

The tempting move was to launch anyway, or to "optimize the obvious thing." My obvious thing was the document mask: Phase 1 used PyTorch's fused causal attention (`is_causal=True`), the fastest path available. `flex_attention` sits at the other end of the trade-off. Its flexibility is **high: it supports arbitrary combinations of masks and biases via `score_mod` and `mask_mod` functions.** Ours is a two-line `mask_mod`:

```python
def mask_mod(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])
```

That gets compiled into a block-sparse kernel, with no dense 2048×2048 mask ever materialized. Flexible usually means slower, so I expected most of the gap to be here.

You chose option (b), profile and fix before launching, on the condition that each step asked for permission first. `scripts/profile_phase2.py` timed each piece of a step in isolation on `fire01` (CUDA-synchronized medians):

| Per micro-batch (16 per step) | Time |
|---|---|
| Data loading (`get_batch`) | **85.8 ms** |
| Mask build (`create_block_mask`) | 8.0 ms |
| Forward+backward **with** doc mask (incl. mask) | 379.3 ms |
| Forward+backward **without** mask (Phase 1 path) | 365.6 ms |
| Optimizer step, Muon+AdamW (once per step) | 130 ms |

- **The document mask costs ~3.7%.** My guess was wrong; `flex_attention` is nearly as fast as the fused causal kernel here.
- **Data loading is the whole gap:** 86 ms × 16 = **~1.37 s per step with the GPU idle**, because the loader ran serially on the main thread. The likely cause is random reads from network storage (`/lambda/nfs`), with 16 rows per micro-batch coming from different pools.
- **Recompiles were a non-issue:** a one-time per-layer-index specialization at warm-up, inherited from Phase 1 code.
- **Ceiling if loading overlapped compute:** ~6.2 s/step, **~84.6k tok/s**, about 39h.

### Mistake — `pkill -f` killing its own shell, twice

Twice I ran a remote command that ended with `pkill -f <pattern>`, where the pattern also appeared in that same command line, so `pkill` killed the shell running it and ssh returned exit 255. The first time was while stopping the auto-launch chain, the second while cleaning up the GPU-utilization sampler after profiling. Both times the intended work had already happened and I verified the end state, but it's an easy footgun. Use `pgrep`/`ps` with a bracketed pattern (`[p]hase2_launch`) or kill by PID.

## 6. The fix and its effect

Don't make the loader faster; take it off the critical path. `BatchPrefetcher` (`src/data/dataset.py`) runs `get_batch_cpu` on a background thread into a bounded queue (depth 8, pinned CPU tensors); the training loop pops the next ready batch. One thread is enough (86 ms of loading per 379 ms of compute), and the GIL isn't a concern because the thread mostly waits on NFS reads.

The non-negotiable constraint: **the fix must not change what the model sees.** The mixture depends on the step, so the producer walks steps in exactly the training loop's order. `test_prefetcher_matches_inline_order` checks that the prefetched batches are bit-for-bit identical, in order, to inline `get_batch`, across a stage boundary. 16/16 tests pass, including the original Phase 1 tests.

Smoke test #2, same config as #1:

| | Smoke #1 (serial loader) | Smoke #2 (prefetch) |
|---|---|---|
| Steady-state throughput | ~69–71k tok/s | **~85.5k tok/s** |
| Projected Phase 2 wall-clock | ~48h | **~39h** |
| Loss / val / `grad_norm` | baseline | tracks #1 |

The measured ~85.5k tok/s lands on the profiler's ~84.6k ceiling, within ~2% of Phase 1. **About 9 hours of H100 time saved**, and the model trains on exactly the same data in exactly the same order.

## 7. Where things stand now

- Phase 2 code: done and tested (16/16), Phase 1 path unchanged.
- All 16 pools are on `fire01`; Wikipedia (1,500,000,000 tokens) and FineMath (1,000,000,000) were synced and token-count-verified against the A10 originals. `/lambda/nfs/h100-f1/data/phase2/` is 21GB.
- The smoke-test checkpoint was deleted; nothing is running on `fire01`.
- The earlier auto-launch chain was stopped when you chose option (b). Launching Phase 2 now waits on your explicit go-ahead:

```bash
python3 scripts/train.py configs/phase2.yaml \
  --init_from /lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-base/step_19073.pt
```

  It will appear in W&B as a new run, `moe0-9b-a0-2b-phase02`, in the same project and group as phase01, with `global_step` continuing from 19,073. Expected duration: ~39h for 22,888 steps.
- The write-up of this window is published to the landing site as the FireHacker post `phase-2-profiling-before-launch-data-loading`.
