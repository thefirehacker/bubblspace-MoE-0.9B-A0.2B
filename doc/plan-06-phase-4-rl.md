# Plan 06: Phase 4, RL with verifiable rewards

**Status (27 Nov):** planning; rollout engine and weight sync built and verified; trainer, rewards and task data not built.
**Starts from:** the Phase 3 SFT model that wins the run 01 vs run 02b comparison (doc/issue-02-phase3-over-refusal.md).
**Preview in:** plan-04 §7. **How it works (method, GRPO, checkers):** [rl-primer-phase4.md](rl-primer-phase4.md).

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
| Reward checkers | 🟡 exist inside the evals (sandbox, GSM8K extraction, BFCL matching, IFEval checkers, real tools). They need wrapping as fast reward functions; the sandbox needs to run many programs in parallel. Candidate: Prime Intellect `verifiers` (MIT) environments where they fit |
| GRPO trainer (loss, KL to the SFT model, rollout → reward → update loop) | ❌ not built |
| Task data | ❌ not downloaded or built |

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
