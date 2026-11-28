# Plan 06: Phase 4, RL with verifiable rewards

**Status (28 Nov):** **RL v1 done** (500 steps from Phase 3 run 02b, 27 Nov 22:06–23:45 IST); checkpoints 100 / 300 / 500 on both HF repos with rollouts. The overnight extension (RL v1b / v1c) never trained: its first step hung (§7). fire01 has been terminated; benchmarks run on the A10 from the HF exports (§7, `benchmark/phase4/`).
**Starts from:** the Phase 3 SFT model that wins the run 01 vs run 02b comparison (doc/issue-02-phase3-over-refusal.md).
**Preview in:** plan-04 §7. **How it works (method, GRPO, checkers):** [rl-primer-phase4.md](rl-primer-phase4.md). **Smoke test order, environments, watching a run live:** [rl-smoke-test-guide.md](rl-smoke-test-guide.md).

## Agreed next steps (27 Nov)

RL v1 is also the first test of the RL pipeline, so a small round before a bigger one is the plan.

1. ✅ **Finish SFT.** Run 02b finishes about 19:15 IST, its evals by about 21:00. Compare it with run 01, pick the winner, upload it to Hugging Face. *(27 Nov: run 02b won, doc/issue-02.)*
2. ✅ **RL v1** (about 1–2 days of prep, then about 6 hours of training) on the winning SFT model, using the task mix in this plan. *(27 Nov, 1 h 40 min; §7.)*
3. 🟡 **Benchmark the RL model fully,** with **8 samples per question** (not one): *(28 Nov: chat evals, refusal check, chat test and English 6 done, **greedy only** (§7). Still to run: 8 samples per question (pass@8), long-context tests, hand tests vs OLMo-2 1B.)*
   - the chat evals (HumanEval/MBPP, GSM8K, IFEval, BFCL);
   - the refusal check and long-context tests;
   - the hand tests, against the same OLMo-2 1B checkpoints as before.
4. ⏳ **Decide the next round from those results.** A knowledge stage from the 2.5b base → redo SFT → a bigger RL only when the map below says knowledge.

How the benchmark tells us what kind of training is needed:

| What we see | What it means | Fix |
|---|---|---|
| Right on the 1st try and across 8 tries (pass@1 high) | Solved | — |
| Rarely right on 1 try, but often right somewhere in 8 (pass@1 low, pass@8 high) | The model can do it but doesn't pick it reliably: a skill / choice problem | More or bigger RL sharpens exactly this |
| Almost never right, even in 8 tries (pass@8 ≈ 0) | The model doesn't know it: a knowledge problem (e.g. Qwen3 APIs, TriviaQA facts) | Data: a knowledge pretraining stage → redo SFT; or a docs/search tool |
| Right answers, wrong format or behaviour | SFT data problem (like the refusals) | Fix the SFT mix |

Do **not** start a large knowledge stage before that map. TriviaQA / NQ still have headroom at ~40B tokens, and a ~200B total is a plausible later target if pass@8 ≈ 0 on knowledge items; OLMo-2 1B still fails state-tracking items at 4T, so those need targeted data, not scale alone. Capacity estimate, token target, and the OLMo comparison are in [knowledge-capacity-and-tokens.md](knowledge-capacity-and-tokens.md). After RL v1 the per-area map says which lever to pull.

## 7. RL v1 results and the overnight incident (28 Nov)

**Run:** `configs/rl_v1.yaml`, start Phase 3 run 02b step 1,500, task set v3 filtered for run 02b (6,684 training + 320 held-out
tasks), 500 steps × 64 prompts × 8 answers, ~13 s per step on the H100 (1 h 40 min in total). W&B `Phase4_rl_v1`. The first
start ran out of GPU memory in the first update (32k-token micro-batches); the restart used 16k.

**Held-out test** (greedy, 40 per subject; same task types as training, so this overstates real gains):

| Subject | Step 0 (run 02b) | Step 100 | Step 300 | Step 425 (best average) | **Step 500 (published)** |
|---|---|---|---|---|---|
| Tool calls (incl. parallel, no tool fits) | 0.455 | 0.930 | 0.975 | 0.975 | **1.000** |
| Who did what | 0.500 | 0.850 | 0.850 | 0.975 | **0.975** |
| Constraint + known answer | 0.450 | 0.850 | 0.900 | 0.925 | **0.925** |
| Time problems | 0.225 | 0.525 | 0.550 | 0.650 | **0.625** |
| Refusal balance | 0.550 | 0.725 | 0.725 | 0.775 | **0.750** |
| Code | 0.775 | 0.975 | 0.975 | 0.975 | **0.975** |
| Instruction constraints | 0.425 | 0.500 | 0.525 | 0.550 | **0.575** |
| Maths | 0.325 | 0.175 | 0.250 | 0.275 | 0.250 |
| **Average** | **0.463** | **0.691** | **0.719** | **0.762** | **0.759** |

From step 425 on the average is flat (0.747–0.762; ±0.025 is one question in 40). All 21 held-out evals are in
`evals.jsonl` on the training-states branch `phase4-rlv1-step500-tokens40B`.

Training: reward 0.41 → ~0.75; KL rose to ~0.05–0.1; answers got shorter (~145 → ~75–110 tokens); the share of questions
where all 8 answers scored the same rose 0.38 → ~0.7 (the filtered tasks became easy for the policy).

**Quick check, step 500** (refusal check and chat test, not trained on): harmless refused 1 → **0**/45, harmful refused
8 → 7/8 (the miss discourages it without the refusal phrase), live data honest 7 → **8**/8. Chat test: passive binding
**right for the first time** ("written by Maya" → Maya), **two parallel calls** for two cities (was one), the tool result
used fully, shorter direct answers; one regression: "capital of India" → "Delhi" instead of "New Delhi". Maths flat, matching
the difficulty filter's reading (81% of maths tasks never solved in 8 tries: a knowledge/ability gap, not an RL target).
**Real benchmarks** (not trained on):

| Model | Harmless refused | Harmful refused | Live data honest | HumanEval % | MBPP % | GSM8K % | IFEval prompt / instr. % | BFCL simple | BFCL multiple | BFCL parallel | BFCL no tool needed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT run 01 | 9/45 | 8/8 | 6/8 | 15.2 | 22.6 | 10.0 | 26.1 / 38.1 | 57.0 | 40.0 | 31.0 | 75.4 |
| SFT run 02b (RL start) | 1/45 | 8/8 | 7/8 | 15.2 | 21.8 | 9.6 | 23.1 / 36.0 | 63.0 | 36.0 | 24.0 | 56.2 |
| RL v1 step 100 | 1/45 | 7/8 | 8/8 | 15.9 | 22.4 | 10.5 | 28.1 / 40.3 | 68.0 | 42.0 | 37.0 | 71.7 |
| RL v1 step 300 | 0/45 | 7/8 | 8/8 | 15.9 | 22.8 | 12.3 | 27.9 / 41.8 | 70.0 | 37.0 | 34.0 | 75.4 |
| **RL v1 step 500** | **0/45** | 7/8 | **8/8** | **16.5** | 22.2 | **12.4** | **29.6 / 43.5** | **71.0** | **43.0** | **40.0** | 69.6 |

All five evaluated the same way on 28 Nov (A10, vLLM backend, greedy; `scripts/phase4_benchmarks_a10.sh`, raw results and
`summary.md` in `benchmark/phase4/`). English 6-task average (forgetting check): run 02b **47.91**, RL v1 step 500 **48.00**
(HellaSwag 41.5, PIQA 66.6, ARC-e 59.6, ARC-c 30.5, WinoGrande 54.3, LAMBADA 35.5): no forgetting. BFCL has 100 questions per
category (±5 points is noise); the vLLM backend gives slightly different numbers than the earlier batched runs of the same model
(e.g. run 02b BFCL simple 63 vs 65), so every model in this table was re-run with it.

**Reading:** RL v1 recovered run 02b's tool-calling losses and went past both SFT runs on simple / multiple / parallel calls, gave the best IFEval of any model, a small real GSM8K gain (9.6 → 12.4%), and kept the refusal balance, with no forgetting. Code is flat.

**Incident (night of 27–28 Nov):** the RL v1b chain's first step, refreshing the difficulty filter with the RL v1 model,
hung at ~23:50 IST: `rl_filter.py` forked ~30 child processes for docker sandbox calls that never ran their command (forking
a large multi-threaded process with vLLM + CUDA + 32 sandbox threads can deadlock); they ended in uninterruptible kernel
sleep, holding 49 GB of GPU memory, and fire01's `h100-f1` filesystem stopped responding. No training ran for ~8 h; the
watchers only waited for "done" or "ABORT" lines, which a hang never writes. fire01 was terminated on 28 Nov. What was
lost: RL v1's training log text after 18:15 UTC and its quick-check files (W&B has the curves; checkpoints and rollouts are
on HF; the quick check is redone on the A10).

**Fixes before the next RL run:**
- **No-progress alarm:** every long job writes `status.json`; a watchdog alerts when it hasn't changed for N minutes.
- **Fork-safe sandbox calls:** start the sandbox worker pool **before** vLLM loads (or use `spawn`), and fewer workers (16).
- **Filter refresh without a fresh vLLM load:** reuse the trainer's engine, or score in the trainer itself.

## 1. Method

GRPO-style RL with rewards a program can check. There are no human labels and no reward model.
- For each prompt the policy samples 8 answers. Each answer is scored, and its advantage is its reward minus the group mean.
- Loss: policy gradient with importance ratios against the sampling log-probs, plus a KL penalty to the SFT model.
- The OLMo-2 1B Instruct RLVR stage is the reference; it used GSM8K, MATH and instruction-constraint prompts.

## 2. Task data and rewards

Each area targets weaknesses we measured on the SFT model (hand tests 27 Nov, doc/issue-02 §5–6, the chat evals).

| Area | Data (candidates; licences checked before download) | Reward | Weakness it targets |
|---|---|---|---|
| **Code** | MBPP train/validation, SYNTHETIC-1 verified problems with tests, APPS introductory | unit tests pass in the docker sandbox (scripts/sandbox_exec.py) | correctness; rambling after the code |
| **Library API tasks** (added 27 Nov) | our own tasks with a fixed interface, e.g. "write `generate(prompt) -> str` that loads Qwen3-0.6B with `transformers` and returns the reply"; also `datasets`, `numpy`, `pandas`, `requests` (mocked), `torch` basics | tests run in a **custom sandbox image** with the libraries installed (versions pinned) and **tiny random models cached offline** under the real names (no network, CPU only); the test calls the function and checks behaviour (e.g. returns non-empty text; `model.config.model_type == "qwen3"`; a causal-LM class, not a classifier) | asked "give me code to run qwen 3 using transformers", run 01 loaded **BERT** with a classification class, never generated text, and looped. Syntax and name checks would pass that answer; only a behaviour test catches it. Limit: RL sharpens use of APIs the model has seen; it can't teach APIs it never saw (Qwen3's `enable_thinking`): see §5 |
| **Maths** | GSM8K train, MATH levels 1–3, `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, our own generator | final number exactly right | multi-step calculation (GSM8K 9.2% after SFT) |
| **Time and quantity word problems** (added 27 Nov) | our own generator: **time subtraction and addition with minutes and 12-hour wraparound, in varied phrasings** ("late by", "earlier", "before", "ago", "after", "how long until"), plus mixed units (hours + minutes) | final time or quantity exactly right (e.g. "1:30") | picks the operation from surface cues: "reached at 2, 30 minutes late, when did it start?" → "2 + 30 = 32 minutes". The SFT set had only one "start + duration → add" template and no minutes, so the model adds whenever an event and a time appear. "movie" stays out (hand tests) |
| **Instruction following** | constraint prompts from the Tülu RLVR set ("3 bullets", "no commas", "as JSON") | IFEval-style checker functions | IFEval 24.8% (prompt-level strict) |
| **Tool calls, with real execution** | held-out ToolMind / Glaive / Hermes turns not used in SFT; our real tools (scripts/real_tools.py: calculator, time, weather) and more; no-call prompts | valid JSON + right tool + right arguments; for real tools, also a final answer that matches the real result; reward for **not** calling when no tool fits; reward for fixing a call after an error instead of repeating it | wrong tool picked (`get_current_time` for weather); example values copied from descriptions (`"(17 * 23) + 4"`); results garbled when restated (7,006,751 → "700,675,100"); English text put in arguments (`"e raised to the power of 445"`); the same failing call repeated 4 times |
| **Binding (who did what)** | our own gave / passive / comparison questions, names held out from the probes and hand tests | exact answer (e.g. "Anita") | cap/room and passive failures in run 01 |
| **Long-document questions** | questions about long documents (Wikipedia-style articles, 2k–15k tokens) with short checkable answers, answer position varied | exact answer | "Which Fremen warrior challenges Paul?" (Jamis, about 3.5k tokens back) failed in every prompt layout; with the question first the model just continues the document |
| **Over- and under-refusal** | harmless commands with known answers (Fact-Seeking), live-data questions without tools, a small set of clearly harmful prompts | correct answer / "no live data" / refusal, as appropriate; nothing for the wrong behaviour | balance from doc/issue-02 |

Everything is decontaminated against our evals (13-token overlap), the same as for SFT. Non-commercial sets (e.g. KodCode) stay out.

## 3. Library API tasks: the sandbox image

- Based on the pinned `python:3.11-slim` image already used for code (`scripts/sandbox_exec.py`), plus pinned `transformers`, `torch` (CPU), `datasets`, `numpy`, `pandas`.
- Tiny random-weight models are saved under real model names: a few MB each, built locally from the published configs with the layer sizes shrunk (e.g. Qwen3, Llama, GPT-2, BERT architectures). Loading "Qwen/Qwen3-0.6B" in the sandbox loads the tiny Qwen3, so tests run in seconds with `HF_HUB_OFFLINE=1`.
- The same isolation as now: no network, 1 CPU, memory cap, read-only filesystem, a timeout. The time and memory caps are larger than for MBPP, because importing `torch` is slow.
- Rewards are graded: the code runs = small partial credit; the behaviour test passes = full reward. Syntax alone gets nothing.

## 4. Engineering status

| Item | Status |
|---|---|
| Fast generation: vLLM plugin for our architecture (`vllm_plugin/`) | ✅ built 27 Nov. 99.0% top-1 agreement with our PyTorch model (scripts/vllm_parity.py); **15–17k tok/s on the A10** vs 224 tok/s for our own batched generator on the H100 |
| Weight sync trainer → vLLM (`src/rl/vllm_engine.py`) | ✅ built 27 Nov. Matches a fresh load **exactly** (log-prob difference 0.0, scripts/vllm_sync_test.py); **0.47 s** per sync; sleep level 1 frees about 8 GB on the A10 for the training step (level 2 breaks non-weight buffers such as the RoPE table) |
| RL-style sampling with per-token log-probs | ✅ 64 prompts × 8 answers at 12k tok/s on the A10 |
| Reward checkers (`src/rl/rewards.py`, `src/rl/common.py`) | ✅ built 27 Nov: number, time, name (binding), 24 instruction-constraint types, tool calls (exact / partial 0.2 / no-call), refusal balance; code via the docker sandbox in parallel (MBPP asserts or stdin/stdout cases). `tests/test_rl_rewards.py` (9 tests). Two bugs caught before use: the Tülu RLVR answer keys store numbers as strings and lists as text (would have scored most instruction tasks 0); fence-less code replies were scored as "no code" |
| GRPO trainer (`scripts/train_grpo.py`, `configs/rl_v1.yaml`, `configs/rl_smoke.yaml`) | ✅ smoke test and RL v1 (500 steps) ran 27 Nov (§7); micro-batches cut to 16k tokens after the first start ran out of memory. Built as: vLLM-first memory order, fp32 reference, group advantages (std-normalised), clipped ratio + k3 KL on answer tokens, token-budget micro-batches, MoE load-balance term, held-out evals every 25 steps, checkpoints every 100. First run: the 20-step smoke test (`scripts/rl_smoke_chain.sh`) |
| Task data (`scripts/rl_build_tasks.py`, `scripts/rl_task_gen.py`) | ✅ v1 built 27 Nov: **38,266** decontaminated tasks: maths 12,347 (GSM8K + MATH numeric, from the Tülu RLVR set), instruction constraints 12,835, time problems 5,940, binding 3,997, refusal balance 1,692, code 849 (MBPP train/val + 426 SYNTHETIC-1 Python problems with stdin/stdout tests), tool calls 606 (thin: generator templates repeat; to grow) |
| Difficulty filter (`scripts/rl_filter.py`) | ✅ run 27 Nov for run 01, then for run 02b (task set v3 → 6,684 training + 320 held-out tasks): keeps tasks solved 1–7 times out of 8, and writes the per-area pass@1 / pass@8 map. ⚠️ Its refresh with the RL v1 model hung fire01 (§7); needs the fork-safe fix before reuse |
| Fixes after the smoke test (27 Nov) | ✅ three reward issues found by reading rollouts: (1) Tülu instruction tasks pay the constraint alone ("w = 0 or 14" scored 1 for having no commas): new area `ifeval_answer`, a constraint on a question with a known answer, reward only if both hold (5,019 tasks; weights ifeval 0.10 + ifeval_answer 0.08); (2) no-tool tasks paid any answer: sums and capitals now must be right; (3) only 62 code tasks survived the filter: ~4,200 generated MBPP-style tasks (tests from reference implementations, all self-checked). **Task set v2: 47,476** (`~/rl_tasks/tasks_v2.jsonl`); needs the difficulty filter for the chosen SFT model. 11 reward tests |
| Library API sandbox image | ❌ not built |

## 5. New knowledge (e.g. recent library APIs): before or after RL?

RL improves how a model uses what it already knows; it can't add facts it never saw (a reward of 0 says "wrong", not what is right). New knowledge has to come from data. Options, in the order we'd prefer:

| Option | What | Cost | Risk |
|---|---|---|---|
| **A. Knowledge first, then SFT and RL** (usual order: pretrain → mid-train → SFT → RL) | continued pretraining on recent code and docs (e.g. 1–2B tokens), then rerun SFT (≈3 h) and RL (≈6 h) on top | the extra pretraining (≈4–8 h on the H100) plus redoing SFT and RL | low: each stage is built on the one before |
| **B. Knowledge after RL** | short continued training on docs/code **mixed with SFT- and RL-format data** (replay), low LR | a few hours | raw documents after RL can wear away the chat format and RL gains (forgetting); research also finds facts first learned during fine-tuning raise hallucination. Needs the full eval suite before and after |
| **C. No training: give it a docs tool** (retrieval) | a `search_docs(query)` tool returns the current API docs; RL rewards using it | small | fits our tool-use focus and stays current as libraries change; limited by how well the model reads docs |

Recommendation: C for fast-changing APIs (it suits a 0.9B model and its tool-use focus), plus A when there's a planned next training round. B is possible but riskiest.

## 6. Time estimate

- **Run size:** 500 steps × 64 prompts × 8 answers, about 350 tokens per answer, about 90M generated tokens in total.
- **Per step on the H100:** generation about 10–25 s (vLLM), training update about 10 s, rewards about 5 s (partly overlapped). That's **about 5–7 hours** for 500 steps, including evals.
- **Prep:** about 1–2 days (trainer, reward wrappers, data, a 20-step smoke test, then the user's go).
