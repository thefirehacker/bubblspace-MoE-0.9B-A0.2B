# Plan 06: Phase 4, RL with verifiable rewards

**Status (27 Nov, evening):** rollout engine, weight sync, rewards, task set and GRPO trainer built; difficulty filter running on the A10 (run 01); 20-step RL smoke test armed on fire01 for after run 02b + its evals. RL v1 waits for the user's go.
**Starts from:** the Phase 3 SFT model that wins the run 01 vs run 02b comparison (doc/issue-02-phase3-over-refusal.md).
**Preview in:** plan-04 §7. **How it works (method, GRPO, checkers):** [rl-primer-phase4.md](rl-primer-phase4.md). **Smoke test order, environments, watching a run live:** [rl-smoke-test-guide.md](rl-smoke-test-guide.md).

## Agreed next steps (27 Nov)

RL v1 is also the first test of the RL pipeline, so a small round before a bigger one is the plan.

1. **Finish SFT.** Run 02b finishes about 19:15 IST, its evals by about 21:00. Compare it with run 01, pick the winner, upload it to Hugging Face.
2. **RL v1** (about 1–2 days of prep, then about 6 hours of training) on the winning SFT model, using the task mix in this plan.
3. **Benchmark the RL model fully,** with **8 samples per question** (not one):
   - the chat evals (HumanEval/MBPP, GSM8K, IFEval, BFCL);
   - the refusal check and long-context tests;
   - the hand tests, against the same OLMo-2 1B checkpoints as before.
4. **Decide the next round from those results.** A knowledge stage from the 2.5b base → redo SFT → a bigger RL only when the map below says knowledge.

How the benchmark tells us what kind of training is needed:

| What we see | What it means | Fix |
|---|---|---|
| Right on the 1st try and across 8 tries (pass@1 high) | Solved | — |
| Rarely right on 1 try, but often right somewhere in 8 (pass@1 low, pass@8 high) | The model can do it but doesn't pick it reliably: a skill / choice problem | More or bigger RL sharpens exactly this |
| Almost never right, even in 8 tries (pass@8 ≈ 0) | The model doesn't know it: a knowledge problem (e.g. Qwen3 APIs, TriviaQA facts) | Data: a knowledge pretraining stage → redo SFT; or a docs/search tool |
| Right answers, wrong format or behaviour | SFT data problem (like the refusals) | Fix the SFT mix |

Do **not** start a large knowledge stage before that map. TriviaQA / NQ still have headroom at ~40B tokens, and a ~200B total is a plausible later target if pass@8 ≈ 0 on knowledge items; OLMo-2 1B still fails state-tracking items at 4T, so those need targeted data, not scale alone. Capacity estimate, token target, and the OLMo comparison are in [knowledge-capacity-and-tokens.md](knowledge-capacity-and-tokens.md). After RL v1 the per-area map says which lever to pull.

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
| GRPO trainer (`scripts/train_grpo.py`, `configs/rl_v1.yaml`, `configs/rl_smoke.yaml`) | 🟡 written 27 Nov, not yet run: vLLM-first memory order, fp32 reference, group advantages (std-normalised), clipped ratio + k3 KL on answer tokens, token-budget micro-batches, MoE load-balance term, held-out evals every 25 steps, checkpoints every 100. First run: the 20-step smoke test (`scripts/rl_smoke_chain.sh`) |
| Task data (`scripts/rl_build_tasks.py`, `scripts/rl_task_gen.py`) | ✅ v1 built 27 Nov: **38,266** decontaminated tasks: maths 12,347 (GSM8K + MATH numeric, from the Tülu RLVR set), instruction constraints 12,835, time problems 5,940, binding 3,997, refusal balance 1,692, code 849 (MBPP train/val + 426 SYNTHETIC-1 Python problems with stdin/stdout tests), tool calls 606 (thin: generator templates repeat; to grow) |
| Difficulty filter (`scripts/rl_filter.py`) | 🔄 running 27 Nov on the A10 for run 01 (8 answers per task): keeps tasks solved 1–7 times out of 8, and writes the per-area pass@1 / pass@8 map. Rerun for the chosen SFT model |
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
