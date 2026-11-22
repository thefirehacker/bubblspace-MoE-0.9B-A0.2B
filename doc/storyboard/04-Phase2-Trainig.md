# Storyboard 04 — Phase 2: continual training, from design to step 7,000

This covers Phase 2 from the decision to start it through step ~7,000 of 22,888, roughly 13 hours into the run. It includes the data-mix design, the code changes, how the run is split into a plateau and an anneal for continual learning, the evals built to watch it, and what those evals show so far. It's chronological where that helps, and includes the mistakes. The pre-flight profiling has its own storyboard ([03](03-profiling-storyboard.md)), and Phase 1 is in [02.5](02.5-Phase1.md).

## 1. Where Phase 1 left us

Phase 1 finished at `step_19073`: 10B FineWeb-Edu tokens, a coherent English base model, with PIQA 68.2, ARC-Easy 56.1, HellaSwag 42.9 and LAMBADA 34.8. Phase 1 also exposed two problems that shaped every Phase 2 decision.

1. **The model had never seen code, a tool call or a reasoning trace.** The 12 special tokens (FIM, `<tools>`, `<tool_call>`, `<think>`, …) were in the vocabulary but untrained. Worse, they were actively suppressed: on `step_9000` their embedding norms measured ~1.10, against ~1.28 for normal vocabulary. Nineteen thousand steps of softmax pressure and weight decay had pushed them down with nothing pulling them up.
2. **Entity-attribute binding was weak.** The model activated the right neighbourhood but bound the wrong edge. "The capital of the United States is" → "Washington, DC." followed by "This city is in California". WinoGrande stayed at chance (51%) all run. Our own 100-case minimal-pair probe scored 25/100, with three categories (comparative value, emotional attribution, possession/obligation) at a flat, confound-free zero.

You rejected "small model capacity" as an explanation: if a model can't bind entities at this scale, scale alone won't fix it. So Phase 2 had two jobs: teach code and agentic formats, and attack binding directly. All without losing the English base.

## 2. Designing the data mix

The mix went through three versions. Each was corrected by evidence, not taste.

### Version 1 — the first draft (rejected)

71.5% English, about 22% code and 6.6% agentic, on a 5B budget. The audit ([`issue-01-phase2.md`](../issue-01-phase2.md)) showed this would leave the finished model at >90% English overall, and would spread ~1.1B code tokens across 13 languages (~85M each). Continual-pretraining practice puts replay of the old distribution at roughly 20–30%, not 70%. The audit also found the planned code datasets unusable: `the-stack-v2-dedup` and `python-edu` ship only Software Heritage blob IDs, not file text.

### Version 2 — the audited plan

12B tokens: 26% English replay, 63% code and 11% agentic ([`plan-02`](../plan-02-phase-2-continual-training.md)). Code was narrowed to 5 languages (Python, TypeScript, JavaScript, Go, SQL/shell/config) so each gets 4–6× more signal, sourced from `Avelina/python-edu-cleaned` (hydrated text) and `bigcode/starcoderdata`.

The budget tier was an explicit cost decision. 12B is enough for each of 64 experts to see roughly 1B code tokens under top-8 routing, without a multi-day commitment. It works out to about 39 hours on one H100.

The agentic pool came in far smaller than estimated. Episodes are hard-filtered to ≤2048 tokens (dropped, never truncated), and OpenThoughts-114k kept only 8,575 of 114,000 traces because full chains of thought average ~8k tokens. Two sources were added to close the gap, glaive-function-calling-v2 (58.9M tokens, 0.5% dropped) and Bespoke-Stratos-17k (5.5M), for a total of ~375M unique tokens. Rather than repeat that 6×, the agentic target was cut to 1.3B (~3.5× repetition) and the difference moved to code.

Security checks changed the plan twice:
- `OpenCoder-LLM/opc-annealing-corpus` was dropped entirely. Hugging Face's ClamAV scan flagged 59 of 85 sampled files in exactly the configs we'd selected.
- `starcoderdata`'s 4 flagged files were confirmed to sit outside our 5 languages.

### Version 3 — rebalanced for binding (what's running)

The binding diagnosis changed the English pool. Version 2's English was half reused FineWeb-Edu, the same noisy web distribution that had produced the problem. Its anneal also dropped English to 16% with zero encyclopedic or STEM text. The pre-Phase 2 review ([`review-01-pre-phase2-semantic-diagnosis.md`](../review-01-pre-phase2-semantic-diagnosis.md)) rebuilt the English group around dense, contradiction-free sources:

| English pool | Unique tokens | Within-group weight | Why |
|---|---|---|---|
| Wikipedia `20231101.en` | 1.5B | 1.5 | clean entity–attribute facts (capitals, history, geography) |
| Cosmopedia-v2 | 1.52B | 1.5 | textbook-style explanations |
| FineMath-4plus | 1.0B | 1.0 | mathematical and STEM prose, multi-step reasoning |
| FineWeb-Edu replay (Phase 1 shards) | 10B available | 0.8 | anti-forgetting anchor, deliberately small |

Then a fourth group was added: **contrastive binding**. `scripts/gen_contrastive.py` writes synthetic minimal-pair documents, the same fact stated with the entities swapped and the attribute following the right entity. It covers five relation families aimed at the probe's zero categories, written in six document styles. Two guards keep it from inflating the probes:
- Any document that pairs two entities appearing together in a probe context is removed.
- Every probe name is removed (Carlos, Emma, Grace, Priya and Wei were dropped).

The first test run was only 78% unique: the physical-fit family has only ~80 possible documents. So the generator now writes only unique documents (61.9M tokens), and repetition is controlled in one place, the sampling weight.

**Final group weights** (`configs/phase2.yaml`):

| Group | Stage 2A (10B tokens) | Stage 2B (2B tokens) |
|---|---|---|
| English + knowledge + STEM | 41.5% | 34% (originally FineWeb-Edu boost 0; changed to ~5.7% at step 19,000) |
| Code | 49% | 38% (python-edu boost ×3) |
| Agentic | 8% | 22% |
| Contrastive | 1.5% (~150M drawn) | 6% (~120M drawn; v2 corpus from step 19,000) |

Two ideas from the review were deliberately left out:
- **Shared-expert grafting** was rejected. As you pointed out, all 64 routed experts already share the residual stream and every attention layer, so there's no structural gap for a shared expert to fill.
- **A multi-token-prediction head and SmolLM2-1.7B logit distillation** were kept off, as a decision for mid-run. The MTP evidence is modest and comes from much larger models. Distillation would cut throughput by 35–45%.

## 3. How the schedule is built for continual learning

Phase 2 isn't a fresh run. It continues a model that already went through a full warmup and cosine decay. The schedule exists to let new skills in without washing old ones out. That takes four mechanisms working together.

**1. Start from Phase 1 with its optimizer state.**
- `--init_from step_19073.pt` loads the weights and the Muon momentum and AdamW moment buffers, then starts Phase 2's own step counter at 0.
- It's separate from `--resume`, which would have exited immediately because Phase 1's loop was already at `total_steps`.
- Carrying the optimizer state avoids the shock of an optimizer that has forgotten the loss surface it's on.

**2. Re-warm, don't restart.**
- Learning rate goes from 6e-5 to 2.4e-4 over 300 steps. 6e-5 is exactly where Phase 1 ended (`min_lr_ratio` 0.1 × 6e-4), so there's no discontinuity.
- The 2.4e-4 peak is 0.4× Phase 1's peak. It's large enough for Muon to re-specialize experts toward code, and for AdamW to pull the suppressed special-token embeddings back into use. It's small enough not to overwrite what Phase 1 learned.

**3. A long stable plateau with replay (Stage 2A, steps 0–19,073, 10B tokens).**
- The learning rate holds at 2.4e-4.
- Every batch is a blend of English, code, agentic and contrastive data, never 100% of anything. Every optimizer step still carries English gradient, which is the standard defence against catastrophic forgetting.
- In an MoE it also protects the router: experts that specialized on English keep receiving English tokens instead of being reassigned wholesale to code syntax.

**4. A special annealing phase (Stage 2B, steps 19,073–22,888, 2B tokens).**
- The learning rate decays linearly 2.4e-4 → 6e-6, and the data shifts at the same moment toward the highest-value material:
  - agentic 8% → 22%
  - contrastive 1.5% → 6%
  - code reweighted ×3 toward curated `python-edu-cleaned`
  - English kept at 34%, but only Wikipedia, Cosmopedia and FineMath; raw FineWeb-Edu is switched off *(changed mid-run: FineWeb-Edu kept at ~5.7%, and contrastive v1 replaced by v2; see section 12)*
- The reasoning comes from Warmup-Stable-Decay training (used by SmolLM2 and others): data seen during the low-learning-rate tail has an outsized effect on the final checkpoint. Updates there are small and precise, so concentrated specialized data imprints its patterns without the large updates that cause forgetting at high learning rate.
- This is why the anneal is a separate phase rather than just the end of a curve: the plateau builds broad competence on a mixed stream, and the anneal consolidates the skills we care about most at the point where the model is most receptive and least prone to drift.

Splitting by token count: 10B in the plateau and 2B in the anneal (~17%), in line with the 10–20% decay fraction common in WSD schedules. Stage 2A and 2B are the same run with one config. The sampler reads the current step and switches mixtures at step 19,073 by itself.

## 4. What files changed

Everything is opt-in. A config without a `phase2:` block runs the original Phase 1 code path unchanged, and the Phase 1 tests still pass.

| File | Change |
|---|---|
| `src/model/attention.py` | `build_doc_block_mask(idx, eot_id)`: each token's document id comes from `<|endoftext|>` positions, and a causal + same-document `BlockMask` is compiled into `flex_attention`. Later, a KV-cache path for fast generation in evals (prefill with causal SDPA, then one token at a time against cached keys and values). With neither, the original fused causal kernel runs. |
| `src/model/block.py`, `src/model/transformer.py` | Pass `block_mask` / `layer_cache` through; `KVCache` + `new_kv_cache()`; RoPE offset by cached length. |
| `src/data/dataset.py` | `MultiPoolTokenDataset`: 16 named pools, stage-dependent group weights, per-pool weights and stage `boost`, held-out validation tail per pool. Flat pools sample random windows. Agentic pools are block-aligned (every sample starts at an episode), with pad targets set to −100. `BatchPrefetcher`: a background thread that produces batches in exactly the training loop's order (the profiling fix, [03](03-profiling-storyboard.md)). |
| `src/data/fim.py` | `apply_window_fim`: FIM on one document span *inside* the sampled 2048-token window, 50/50 PSM/SPM. Phase 1's file-level FIM split the three sentinels across windows for any file over 2048 tokens. |
| `src/optim/schedule.py` | `build_phase2_scheduler`: re-warm, plateau, linear anneal. |
| `scripts/train.py` | `--init_from`; Phase 2 branch with multi-pool data, prefetcher, doc mask per micro-batch, `ignore_index=-100`, per-pool validation every 250 steps, W&B `global_step` = step + 19,073. |
| `configs/phase2.yaml`, `configs/phase2_smoke.yaml` | Full run and 100-step smoke variant. |
| `scripts/gen_contrastive.py` | Contrastive binding corpus with contamination guard. |
| `tests/test_phase2.py` | Doc-mask isolation (perturbing document A leaves document B's logits bit-identical), FIM invariants, block alignment and pad masking, stage mixture, schedule boundaries, prefetcher order equivalence, KV cache = full forward. 17 tests pass. |

Two bugs were caught before any GPU time:
- A per-pool `weight` that `_Pool` never stored, so the sampler silently fell back to token-count weighting.
- A divide-by-zero when a stage boosted every pool in a group to 0.

## 5. Pre-flight: smoke test and profiling

Covered in detail in [03](03-profiling-storyboard.md):
- Smoke test #1 passed every correctness check but ran at ~70k tok/s, which projected ~48h.
- Profiling showed the document mask costs only ~3.7%. My guess that it was the bottleneck was wrong. The real cause was serial data loading from network storage, 86 ms × 16 micro-batches per step with the GPU idle.
- The background prefetcher brought throughput to ~85.5k tok/s on exactly the same data in the same order, saving ~9 hours.

## 6. Launch

Launched with your go-ahead at **18:01 UTC, 21 Nov**:

```bash
python3 scripts/train.py configs/phase2.yaml \
  --init_from /lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-base/step_19073.pt
```

It appears in W&B as run `moe0-9b-a0-2b-phase02` (`lec9ww5h`), in the same project and group as Phase 1, with `global_step` continuing from 19,073 so the two runs line up on one axis.

**Mistake:** I launched without `python -u`, so the log file is block-buffered and shows nothing for long stretches. Two monitors expired with no events before I switched to reading metrics from the W&B API. Future launches use `-u`.

## 7. Training health

| Steps | Loss | Load-balance loss | z-loss | grad_norm mean / max | tok/s |
|---|---|---|---|---|---|
| 0–500 (re-warm) | 1.93 | 1.16 | 0.116 | 0.35 / 3.1 | 84.2k |
| 1,000–1,500 | 1.64 | 1.05 | 0.096 | 0.19 / 0.21 | 84.4k |
| 2,500–3,000 | 1.56 | 1.03 | 0.090 | 0.19 / 0.77 | 84.0k |
| 4,000–4,500 | 1.52 | 1.02 | 0.085 | 0.17 / 0.18 | 84.3k |
| 6,500–7,000 | 1.50 | 1.01 | 0.080 | 0.17 / 0.21 | 84.0k |

- No crashes and no restarts.
- A liveness check (every 10, then 40 minutes) never fired.
- The one grad-norm spike (3.1) is the re-warm, and it settled within ~80 steps, as in the smoke test.
- Load-balance loss started at ~1.16 as the router adapted to code and agentic tokens, and is now back near the ideal 1.0.
- The finish is projected for ~08:00–09:30 UTC, 23 Nov.

**Per-pool validation loss** (held-out, every 250 steps):

| Group | Step 0 | Step 1,000 | Step 6,750 |
|---|---|---|---|
| Code | 3.51 | 1.35 | 1.16 |
| Agentic | 2.93 | 1.22 | 1.09 |
| English | 2.67 | 2.26 | 2.21 |
| Contrastive | 3.87 | 0.59 | 0.59 |
| FineWeb-Edu alone | 2.50 | 2.65 | 2.76 ↑ |

## 8. Reading the router properly

You asked how to read W&B's `tokens_per_expert` histogram. Explaining it turned up a real flaw in the chart: it sums counts by expert *index* across all 16 layers. Expert 37 in layer 0 and expert 37 in layer 9 are unrelated, so the chart can look balanced while individual layers are badly skewed, or the reverse.

`scripts/analyze_routing.py` measures routing per (layer, expert), separately for each data group, on held-out windows. It reports load CV, entropy, starved (<0.1× fair share) and overloaded (>3×) experts, and domain specialization (an expert's share of a domain vs. the rest). It runs on the A10 against relayed checkpoints, so training is never touched.

| | Phase 1 final | P2 1k | 3k | 5k | 7k |
|---|---|---|---|---|---|
| Mean load CV | 0.623 | 0.385 | 0.318 | 0.293 | 0.285 |
| Overloaded experts | 21 | 5 | 0 | 0 | 0 |
| Starved experts | 0 | 2 | 0 | 1 | 0 |

At step 1,000, layer-0 experts 39 and 58 were starved (~0.09× load), and I flagged them as a death risk. They recovered. Since then one layer-0 expert is briefly starved at some checkpoints (17 at step 2,000, 31 at step 5,000), a different one each time. That's rotation, not death.

The analysis also shows real specialization emerging. For example, a layer-1 expert receives code tokens at ~16× the rate of other domains, and a layer-7 expert takes contrastive text at ~12×.

## 9. Building the eval suite

Phase 1's six English benchmarks can't see code, tool calls or binding. You asked for more evals for agentic coding. The plan is in [`eval-plan-phase2.md`](../eval-plan-phase2.md), and every eval runs on the A10, never on `fire01`.

**What was built:**
- `scripts/eval_phase2_quick.py`, every 1,000 steps:
  - fill-in-the-middle exact match on held-out code
  - agentic tool-call format: emits, closes, valid JSON, name in tools, name = gold, argument keys = gold, ends turn
  - reasoning `<think>` open/close
  - fixed greedy prompts for side-by-side inspection
- Probe v3: 70 binding cases whose sentence structures are disjoint from both v2 and the contrastive corpus. If v2 rises and v3 doesn't, the model memorized templates instead of learning to bind.
- `scripts/eval_phase2_milestone.py`: CRUXEval-O (predict a program's output, 2-shot official prompt) and BFCL v3 simple / multiple / parallel, converted to our `<tools>`/`<tool_call>` format and scored by AST match. Neither executes model-written code. HumanEval/MBPP do, so they wait for a sandbox.
- `scripts/contamination_check.py`: 13-gram overlap of BFCL against the raw ToolMind, Hermes and Glaive files. It found 11 of 603 questions, which are excluded from scoring.

**The determinism discovery.** While adding the KV cache for faster generation, cached and uncached greedy outputs diverged after a few tokens. That looked like a cache bug. It wasn't. Two *identical uncached* forward passes also differed by up to ~0.1 in the logits: non-deterministic CUDA kernels introduce tiny float differences, and those flip near-tied top-8 router choices (0–12 tokens per layer got a different expert set between runs). For evals that means the same checkpoint scores differently run to run. `src/eval/determinism.py` (deterministic algorithms plus a fixed cuBLAS workspace) makes runs bit-identical, and every eval script calls it first. The Phase 1 baseline was re-scored under it: v2 25 → 24/100, the honest number.

**The eval pipeline:**

```
fire01 (training)                         A10 (evals)
step_N.pt saved ──► relay_ckpts.sh ──rsync, 40 MB/s cap──► phase2_step_N.pt
                                                         │
                                       eval_watch.sh: probes v2/v3, quick, routing
                                       (+ CRUXEval/BFCL, English at milestones)
                                                         │
                                         results ► benchmark/phase2/*.json
                                         checkpoint deleted after evals
```

The bandwidth cap keeps the relay from competing with training's data reads on network storage. After the step-5,000 English result, the pipeline was extended: the English suite now runs at steps 7,000, 9,000, …, 21,000 and the final 22,888, with CRUXEval/BFCL again at 19,000 and the final step.

## 10. Results so far

The full tables are in [`eval-notes.md`](../eval-notes.md#phase-2-evaluation-continual-training-english--code--agentic--contrastive). In short:

| | Phase 1 final | Phase 2 best so far |
|---|---|---|
| CRUXEval-O pass@1 | 0.0% | 8.5% (step 5k, 94% parsable) |
| BFCL simple / multiple / parallel | 0 / 0 / 0% | 9 / 6 / 0% (step 5k) |
| Agentic: valid call, name = gold | 0% | 48% (step 7k) |
| FIM exact match | 0% | 5% (steps 6k–7k) |
| Reasoning: closes `<think>` | 0% | 0% |
| Probe v2 / v3 | 24 / 4 | 25–31 / 3–6 (noise) |
| English 6-task average | 47.18 | 45.40 (step 5k, −1.78) |

**Honest read:**
1. **Code and agentic formats are being learned.** These moved off zero on external benchmarks. On held-out episodes, step 1,000 generations went from echoing the user to writing well-formed calls like `<tool_call>{"name": "generate_random_color", "arguments": {}}</tool_call>`.
2. **Reasoning opens `<think>` and rambles.** Many BFCL misses are the model starting a ToolMind-style think block and never reaching the call.
3. **Binding hasn't moved.** Contrastive validation loss flattened at ~0.59 by step ~500, so the model fitted the corpus's surface statistics almost immediately, but neither probe shows transfer. The anneal (contrastive ×4) is the last chance within this run. If it fails, binding needs a different approach after Phase 2, not more of the same data.
4. **English forgetting is real.**
   - HellaSwag −2.5 and ARC-Easy −3.3 are several standard errors.
   - FineWeb-Edu validation loss rose 0.26, while Wikipedia and Cosmopedia improved.
   - Part of the gap is the comparison: Phase 1 was measured after its learning rate decayed, and step 5,000 is at peak learning rate. The anneal should recover some of it.
   - I first framed this as a decision for you, then corrected it to "watch". When the numbers crossed my own 1-point threshold, I moved it to "measure the trend". The decision point is step 19,000: if the English trend is still falling, Stage 2B restarts from the step-19,000 checkpoint with a small FineWeb-Edu share instead of zero, losing essentially no progress.

## 11. Mistakes and lessons

- **`pkill -f` / `pgrep -f` matched the remote shell itself** (three times across Phase 2), so the command killed its own shell. Use bracketed patterns (`[t]rain.py`) or kill by PID.
- **Unbuffered logs:** use `python -u` for long runs, and monitor through W&B, not the log file.
- **The KV-cache "bug" was MoE nondeterminism.** Measure the baseline's own variance before blaming new code.
- **A watcher's exit code isn't a verdict.** The eval-queue watcher exited 1 because its final error-grep found no tracebacks. Read the output, not the status.
- **Escalation discipline:** I first presented the FineWeb question as a decision needed today. When you asked what it rested on, the honest answer was one validation-loss signal and a decision ~20 hours away. Present watch items as watch items.

## 12. Mid-run changes at step ~9,000–10,000

### The English trend turned

The English suite moved to every 2,000 steps. The overnight eval scripts covered steps 2,000–7,000 and exited normally at the end of their range, deleting each checkpoint after evaluation, so step 7,000 was copied over again. The trend:

| | Phase 1 | 5k | 7k | 9k |
|---|---|---|---|---|
| 6-task average | 47.18 | 45.40 | 45.25 | **46.19** |

The loss happened early, when the data mix changed, then started recovering at peak learning rate. LAMBADA passed Phase 1 at step 9,000. HellaSwag (−2.2) stayed down.

### Decision: FineWeb-Edu back into the anneal

You decided to keep FineWeb-Edu in Stage 2B at its Stage 2A weight: ~5.7% of the anneal's tokens, taken only from Wikipedia, Cosmopedia and FineMath. Code, agentic and contrastive are untouched. Total tokens and steps don't change.

You asked whether stopping and restarting causes forgetting. It doesn't. `--resume` restores the weights, both optimizers' state, both learning-rate schedulers and the step counter, so it's an exact continuation, unlike the Phase 1 → 2 transition, which re-warmed the learning rate and changed the data. The only change is which data Stage 2B samples.

### Contrastive v2

Free generations from the step-9,000 model showed the v1 phrase "Put the other way around:" being copied. Together with flat v3 and a validation loss that plateaued at step ~500, that showed v1 was teaching templates. `scripts/gen_contrastive_v2.py` rebuilds the corpus:

| | v1 | v2 |
|---|---|---|
| Sentence frames for event + reason | 1, inside 6 wrappers | 15 (fronted, consequence, cleft, semicolon, …) |
| Reference resolution | none | "it" and "the former/latter", with explicit resolution |
| Question formats | 1 | 9, including either/or |
| Document shapes | 6 | 9 + possession chains, narrative distractors, two situations per document |
| Most frequent wrapper phrase | ~17% of documents | 6.4% |
| Unique tokens | 61.9M (~4.4× repetition over Phase 2) | 133.5M (~0.9× in the anneal) |

Its guards assert that:
- no probe v2/v3 entity pair or name appears
- no probe v3 frame phrase appears ("The one that was", "of the two", …)
- no held-out frame appears

The held-out frames, entity pairs and names build **probe v4** (112 cases), the first probe that is template-free by construction. Baselines: Phase 1 13/112, step 1,000 21/112, step 9,000 11/112, all below the ~28 a coin flip scores. That's the honest baseline the anneal has to beat.

### How the restart is set up

`configs/phase2.yaml`:
- adds the `contrastive_v2` pool, boosted to 0 in Stage 2A so steps 19,000–19,073 replay exactly
- sets v1 to 0 in Stage 2B
- keeps FineWeb-Edu in Stage 2B

A sampler check on fire01 confirmed Stage 2A is bit-for-bit the same mix and Stage 2B is as intended.

Before arming anything, `--resume` was tested on the A10 with a real Phase 2 checkpoint (step 1,000, tiny batch):
- **Test 1** started while the step-9,000 probes were running, because my wait condition also matched the English check's "done" line, and it ran out of memory.
- **Test 2** passed: resumed at step 1,000 and trained 168 steps cleanly.
- **Test 3** used the final config with the v2 pool loaded and passed.

`~/switch_2b.sh` on fire01 runs at the step-19,000 checkpoint (~08:25 IST, 23 Nov):
1. Waits for the file to be complete.
2. Checks that the process it will stop really is the training process, and that the final config and v2 data are in place.
3. Stops training and resumes with the same W&B run id.
4. Confirms after 15 minutes. If the new launch fails, it falls back to resuming with the original config, so training doesn't sit stopped overnight.

### Generation controls and step 9,000 on HF

- **Chat REPL** ([`scripts/chat_repl.py`](../../scripts/chat_repl.py)):
  - New controls: repetition penalty (default 1.15), no-repeat 4-grams, min-p, and a `/chat` mode that wraps input in the `<|im_start|>` format and stops at `<|im_end|>`.
  - On the India prompt, the loop ("a country that is very diverse in its culture…" ×3) is replaced by a varied paragraph.
  - In chat mode the model answers "The capital of France is Paris." and stops by itself.
  - Removing the loops exposed the remaining binding errors ("ruled by the Aryans for over 2,000 years").
- **Weights:** the step-9,000 weights went to Hugging Face `main`, and the Phase 1 final weights were preserved on a `phase1-final` branch first. The model card now says which checkpoint is on `main` and where Phase 1 lives.

**Mistakes and lessons today:**
- A `pkill -f` again matched its own shell. The pattern appeared in the config name on the same command line.
- A wait condition matched a substring of a different log line. Anchor log matches.

## 13. Where things stand now

- **Training:** step ~10,000 / 22,888, healthy, Stage 2A. The switch to the new Stage 2B mix is armed for step 19,000.
- **Evals:** English, probes v2/v3/v4, the quick suite and routing every 2,000 steps. CRUXEval/BFCL at step 19,000 and at the final step.
- **Still pending with you:**
  - HumanEval/MBPP in a Docker sandbox on the A10.
  - Fixing the eval's "name in tools" check, which scores below the stricter "name = gold".
  - After Phase 2, the binding plan:
    - knowledge augmentation (many paraphrases per fact)
    - SmolLM2-1.7B distillation as a short Phase 2C
    - SFT, then DPO on consistent vs. contradictory answers
