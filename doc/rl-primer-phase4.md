# Phase 4 RL primer: how our RL setup works

A walk-through of the reinforcement-learning stage planned after SFT: what each part does, which tools it uses, how GRPO works, who scores the answers, and what can go wrong. The task list and status are in [plan-06-phase-4-rl.md](plan-06-phase-4-rl.md); this document explains the method.

---

## 1. The idea

SFT (Phase 3) is **imitation**. We show the model complete example answers and train it to copy them token by token.

RL is **trial and error**:
1. The model answers a question several times.
2. A program checks each answer.
3. Training makes the answers that passed more likely and the ones that failed less likely.

Nobody writes the solution the model learns from. We only need a **checker**: unit tests for code, "is the final time 1:30?" for a word problem, "did it call `get_weather` with the right city?" for tools. That's why this is called **RL with verifiable rewards (RLVR)**.

**Why bother after SFT?** SFT teaches the *form* of good answers but not always the *substance*. Our own example: SFT taught "event + time + number → write `a + b = c`", so the model answers "I arrived at 2:00, 30 minutes late, when did it start?" with "2 + 30 = 32 minutes". In RL, an answer only earns reward if its final time is actually right, so the model is pushed toward answers that work, not answers that look like the training examples.

---

## 2. The parts

```
                ┌──────────────────────── one H100 (fire01), one Python process ────────────────────────┐
 task data ───► │  ROLLOUT ENGINE: vLLM + our plugin       512 answers per step, ~15k+ tok/s            │
 (prompts +     │      ▲ weight sync (0.5 s)                │ token ids + per-token log-probs            │
  answer keys)  │      │                                    ▼                                            │
                │  POLICY: our Transformer (PyTorch) ◄── REWARDS: Python checkers (sandbox, maths,       │
                │  trained with the GRPO loss            IFEval, tool calls with real execution)         │
                │      ▲                                                                                 │
                │  REFERENCE: frozen copy of the SFT model (for the KL penalty)                          │
                └────────────────────────────────────────────────────────────────────────────────────────┘
```

| Part | What it is | Code | Status (27 Nov) |
|---|---|---|---|
| **Policy** | The model being trained: our own `Transformer`, loaded from the SFT winner (run 01 or run 02b) | `src/model/` | ✅ exists |
| **Reference model** | A frozen copy of the same SFT model. It never trains; it measures how far the policy drifts (§4, KL penalty) | `src/model/` | ✅ a second load |
| **Rollout engine** | **vLLM** running our model through our plugin; samples the answers | `vllm_plugin/`, `src/rl/vllm_engine.py` | ✅ built and verified |
| **Weight sync** | After each training step, copies the policy's new weights into vLLM | `VLLMEngine.sync_weights` | ✅ 0.47 s; identical to a fresh load |
| **Rewards** | Python functions that score answers against answer keys (§5) | reuse `scripts/sandbox_exec.py`, `scripts/eval_chat.py`, `scripts/real_tools.py`; optionally Prime Intellect `verifiers` (MIT) | 🟡 code exists, needs wrapping |
| **GRPO trainer** | The loop (§3) and the loss (§4) | to write: about 400 lines reusing `scripts/train.py` pieces | ❌ |
| **Task data** | Prompts plus answer keys or tests (§6) | generators and dataset loaders | ❌ |

### 2.1 Why vLLM, not our own batched generator

RL time is mostly **generation**: every step needs about 512 fresh answers.

| | Our batched generator (`src/eval/batched_gen.py`) | vLLM with our plugin |
|---|---|---|
| How it works | Plain PyTorch. Pads all prompts to one length, runs them together, Python code between every token | A dedicated inference engine (techniques below) |
| Speed | **224 tok/s** on the H100 | **15–17k tok/s on the A10**, the smaller GPU |
| One RL step's generation | about 11 minutes | about 10–25 seconds |

vLLM's techniques:
- **paged attention:** the KV cache is stored in small blocks, so no memory is wasted on padding;
- **continuous batching:** a finished answer frees its slot immediately for the next one;
- **CUDA graphs:** the GPU replays recorded kernel sequences, with no Python between tokens;
- **fused MoE kernels:** all 64 experts are computed in one kernel launch.

vLLM can't run our model by itself, because it's a custom architecture. The **plugin** (`vllm_plugin/bubblspace_vllm/`) re-expresses our layers with vLLM's kernels:
- the attention output gate (sigmoid before `o_proj`);
- the zero-centred norm, which equals vLLM's `GemmaRMSNorm`;
- our 64-expert MoE, mapped onto vLLM's fused MoE.

It reads our own `model.safetensors`. Check against our PyTorch model (`scripts/vllm_parity.py`): **99.0%** of next-token choices agree, with a mean log-prob difference of 0.033 (bf16 rounding).

The batched generator stays as a simple, readable reference to test against.

### 2.2 Weight sync and sharing one GPU

The policy and vLLM live in **one Python process on one GPU**. vLLM's engine runs in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), so a weight sync is a direct GPU-to-GPU copy of about 1.8 GB, with no serialization.

- `sync_weights(policy)` goes through vLLM's own `reload_weights()`. That calls our plugin's loader for every tensor, then reruns any weight processing the kernels need.
- Checked with `scripts/vllm_sync_test.py`: SFT step 1,000 weights pushed into an engine running step 1,500 give **exactly** the same log-probs and texts as a fresh step 1,000 engine (difference 0.0).
- `sleep()` frees vLLM's KV cache and moves its weights to CPU while the policy trains (about 8 GB freed on the A10). `wake()` brings it back. Sleep **level 1** is used: level 2 also discards non-weight buffers such as the RoPE table, which a weight sync doesn't rewrite. We tested this: outputs broke after level 2.

Two bugs were found and fixed while building this:
1. The plugin first copied tensors in directly. vLLM's reload only accepts weights through each parameter's own loader, so most updates were silently lost. Every tensor now goes through vLLM's loaders, including each of the 64 experts per layer.
2. The level-2 sleep problem above.

---

## 3. One RL step, end to end

Example settings: 64 prompts per step, 8 answers each, up to 1,024 new tokens per answer.

| # | What happens | Where | Time (H100, estimate) |
|---|---|---|---|
| 1 | Pick 64 prompts from the task mix, e.g. "I reached the hall at 2:00, 30 minutes late. When did the show start?" (answer key 1:30) | data loader | — |
| 2 | Wake vLLM; copy the policy's current weights into it | `wake()`, `sync_weights()` | 0.5 s |
| 3 | Generate 8 answers per prompt at temperature 1.0 so they differ: 512 answers, each with the log-probability of every sampled token | `VLLMEngine.generate(n=8)` | 10–25 s |
| 4 | Put vLLM to sleep so training gets the memory | `sleep()` | <1 s |
| 5 | Score every answer with its checker (§5) | reward functions | about 5 s (code runs in parallel sandboxes) |
| 6 | Compute each answer's advantage within its group of 8 (§4) | trainer | — |
| 7 | Recompute the token log-probs with the PyTorch policy (π_old) and the reference model | trainer | — |
| 8 | GRPO loss on the answer tokens, backward pass, gradient clip, optimizer step | trainer | about 10 s |
| 9 | Log reward per task, KL, answer length, entropy; every ~50 steps run the held-out evals and save a checkpoint | trainer | — |

Repeat for about 500 steps.

---

## 4. How GRPO works

GRPO = **Group Relative Policy Optimization**.

### 4.1 Advantage: better or worse than its siblings

For one prompt, the 8 answers come back with rewards, e.g. 3 correct and 5 wrong:

```
rewards:    1  0  0  1  0  0  1  0          group mean = 0.375, std ≈ 0.48
advantage:  (reward − mean) / std   →   correct answers ≈ +1.29,  wrong answers ≈ −0.77
```

Each answer is judged **against the other answers to the same question**. That's the "group relative" part, and the key idea. PPO, the older method, trains a separate *value model* to estimate how hard each question is. GRPO uses the group's average instead, which saves a whole extra model's memory and compute.

Some recipes skip dividing by the std ("Dr. GRPO") to avoid over-weighting very easy or very hard groups; we'll compare in the smoke test.

### 4.2 The update: push tokens by their answer's advantage

Every token of a correct answer has its probability nudged up in proportion to +1.29. Every token of a wrong answer is nudged down by −0.77. Over many steps, choices that lead to correct results ("late by → subtract", "weather → `get_weather`") become likely and the others fade.

### 4.3 Two safety rails

**Clipping: no single update is too big.** For each token, compare its probability under the updated policy with its probability when the answer was sampled (π_old):

```
ratio = π_new(token) / π_old(token)
token loss = −min( ratio × A ,  clip(ratio, 1 − ε, 1 + ε) × A )        ε = 0.2
```

Once an update has raised a good token's probability by more than 20% (or lowered a bad one by more than 20%), the clip stops pushing further in this step. That keeps training stable. It matters most when one batch of answers is used for several updates.

**KL penalty: stay close to the SFT model.** For each token, add `β × KL(policy ‖ reference)`, estimated from the two log-probs. Without it, RL can find odd shortcuts that score well but break general behaviour, such as one rigid answer format for everything. β is small (0.01–0.04; some recipes use 0).

**Total loss** = clipped policy term + β × KL, averaged over all answer tokens. Prompt tokens get no loss, as in SFT's assistant-only loss.

### 4.4 vLLM numbers vs PyTorch numbers

vLLM's log-probs differ slightly from PyTorch's (mean 0.03, bf16 kernels). If vLLM's values were used as π_old, the ratio would be off from 1 before any update. So π_old is **recomputed with the PyTorch policy** right after generation; vLLM is only used to sample. (The alternative is a truncated importance-sampling correction.)

---

## 5. Who scores the answers: the checkers

**The checker is ordinary Python code, not another LLM.** Every RL task comes with a known answer or a test, and a small function compares the model's output against it. It's deterministic: the same output always gets the same score.

| Task | Where the right answer comes from | How the checker scores it |
|---|---|---|
| **Maths** (GSM8K, MATH) | the dataset includes the **gold final answer** (e.g. `#### 19`) | extract the last number or `\boxed{}` from the reply (same code as our GSM8K eval); 1 if equal |
| **Time and quantity word problems** | **our generator computes it** (§5.1) | find the time in the reply ("1:30", "1.30", "half past one"); compare |
| **Code** (MBPP, SYNTHETIC-1, APPS) | the dataset includes **unit tests** | extract the code, run it with the tests in the docker sandbox (no network, 1 CPU, 512 MB, 10 s); all pass = 1 |
| **Library API tasks** | **our own tests**, against a fixed interface ("write `generate(prompt) -> str` using `transformers` and Qwen3-0.6B") | run in a sandbox image with the libraries installed and tiny random models cached offline under the real names; test the behaviour (returns text, model is a Qwen3 causal LM). Syntax and name checks alone earn nothing |
| **Instruction following** | the prompt states **checkable constraints** ("exactly 3 bullets", "no commas", "valid JSON") | one function per constraint (count bullets, search for commas, `json.loads`) |
| **Tool calls** | held-out ToolMind / Glaive / Hermes turns include the **gold call**; real tools can be **run** | parse `<tool_call>`, compare name and arguments; for real tools, run them and check the final answer contains the real result |
| **Who did what** | our generated sentences include the **correct name** | exact match ("Anita") |
| **Long documents** | questions built from the document with a **known short answer** | exact or normalized match |
| **Refusal balance** | prompts labelled harmless / harmful / live data | harmless: correct answer and no refusal; harmful: refusal; live data: says it has no live access |

Example checkers:

```python
def reward_math(reply, gold):                      # gold = "19"
    pred = last_number_or_boxed(reply)             # same extraction as scripts/eval_chat.py gsm8k_pred
    return 1.0 if pred is not None and float(pred) == float(gold) else 0.0

def reward_code(reply, tests):
    program = extract_code_block(reply) + "\n" + tests
    return 1.0 if run_in_sandbox(program).exit_code == 0 else 0.0     # scripts/sandbox_exec.py
```

Some tasks give **partial credit**. A tool call with valid JSON but the wrong tool can get a little, so the model learns the format before the choice. Every task gets nothing if the answer is cut off at the length limit or never ends its turn.

### 5.1 Generated tasks: correct by construction

For time problems (and our binding questions) there's no external dataset. A Python generator picks the numbers first, computes the answer with ordinary arithmetic, and only then writes the question text around those numbers:

```python
def make_time_problem(rng):
    start_h, start_m = rng.randint(1, 12), rng.choice([0, 15, 30, 45])   # 1. pick numbers
    late_min = rng.choice([10, 15, 20, 30, 45, 60, 90])
    total = start_h * 60 + start_m + late_min                             # 2. compute the answer
    arrive = f"{(total // 60 - 1) % 12 + 1}:{total % 60:02d}"             #    (12-hour wraparound)
    question = rng.choice([                                               # 3. write the text around them
        f"I reached the hall at {arrive}, {late_min} minutes late. When did the show start?",
        f"I walked in {late_min} minutes after the talk began, at {arrive}. What time did it begin?",
        f"I was {late_min} minutes late and arrived at {arrive}. What was the start time?",
    ])
    return question, f"{start_h}:{start_m:02d}"                           # answer key for the checker
```

Safeguards:
- **Unit tests on the generator**, plus a hand spot-check of a few hundred samples. A generator bug would give wrong answer keys.
- **Many phrasings in both directions** (add *and* subtract), minutes, wraparound past 12, and distractor numbers, so no single surface pattern pays off. Optionally a larger LLM paraphrases questions, while code checks the generator's numbers survived the rewrite.
- **Hand-test words** ("movie") and hand-test sentences stay out.

### 5.2 Is this supervised data?

Half yes. The datasets contain the right answers, like supervised data, but **the model never sees them**:
- In SFT, the full solution is shown and copied token by token.
- In RL, the model sees only the question and writes its own answers (8 tries). The answer key is used only by the checker, and training pushes the model toward its own correct tries.

So the model has to find a working way to the answer itself. That's why RL can fix what SFT couldn't.

### 5.3 Alternatives we're not using

| Approach | How it scores | Why not in Phase 4 |
|---|---|---|
| **Reward model** (classic RLHF) | a neural network trained on human ratings ("A is better than B") | needs a large preference dataset and another model; the policy learns to fool it |
| **LLM as a judge** | another LLM grades each answer | slow and costly (512 answers per step), inconsistent; a small policy learns to please the judge rather than be right |
| **Verifiable rewards (ours)** | code checks against a known answer | fast, free, exact, hard to fool, **but only where an answer can be checked** |

The limit: open-ended qualities (essay quality, helpfulness, tone) have no checkable answer, so RLVR doesn't train them. That would be a later preference-tuning step (e.g. DPO on better/worse answer pairs). OLMo-2 Instruct did both: preference tuning for general chat, then RLVR for maths, code and instructions.

---

## 6. Task mix and difficulty

| Area | Targets (measured on the SFT model, 27 Nov) |
|---|---|
| Code (sandbox tests) | correctness; rambling after the code |
| Library API tasks (behaviour tests in a sandbox image with the libraries and tiny offline models) | "code to run Qwen3 with transformers" → BERT, a classifier class, no generation |
| Maths (exact number) | GSM8K 9.2% |
| Time and quantity word problems (exact time) | wrong operation ("2 + 30 = 32 minutes"), units, 12-hour wraparound |
| Instruction following (constraint checkers) | IFEval 24.8% prompt-level |
| Tool calls with real execution | wrong tool; example values copied from descriptions; results garbled when restated; English inside arguments; repeating a failing call; calling when no tool fits |
| Who did what (exact name) | gave / passive sentences |
| Long-document questions (exact answer) | details deep in 2k–15k-token texts (the "Jamis" failure) |
| Refusal balance | answer harmless, say "no live data" when true, refuse clearly harmful |

**Difficulty filter.** If all 8 answers to a prompt are right, or all wrong, every advantage is 0 and that prompt teaches nothing. Before RL, the SFT model answers every candidate prompt 8 times (fast with vLLM), and we keep prompts it solves between 1 and 7 times out of 8. During the run, we re-filter as the model improves.

**Mixing.** Every batch mixes all areas, so improving one doesn't quietly break another.

---

## 7. Starting settings (to be tuned in the smoke test)

| Setting | Start value | Why |
|---|---|---|
| Prompts per step × answers per prompt | 64 × 8 = 512 | enough answers per group for a stable baseline |
| Sampling | temperature 1.0, top-p 1.0 | answers must differ for the group comparison to work |
| Max answer length | 1,024 tokens (no reward if cut off) | `<think>` blocks before tool calls run 400–600 tokens |
| Learning rate | about 1e-6, constant after a short warmup | RL is fragile; about 60× smaller than SFT's peak (6e-5) |
| Clip ε | 0.2 | standard |
| KL β | 0.01–0.04 (compare 0) | keeps behaviour close to SFT |
| Updates per batch | 1 at first (then try 2) | with 1, π_old = π_new at the start of the update |
| Steps | about 500 | about 90M generated tokens |

---

## 8. What can go wrong

| Problem | What it looks like | Guard |
|---|---|---|
| **No learning signal** | many all-right or all-wrong groups; reward flat | difficulty filter (§6) |
| **Reward hacking** | reward rises, real quality doesn't | check real results only (e.g. the SFT model writes fake ```` ```output ```` blocks; the reward ignores them and checks the final answer); KL penalty; read samples every ~50 steps |
| **Length creep** | answers and `<think>` blocks keep growing | length cap with no reward when cut off; track mean length |
| **Forgetting** | maths improves, chat or code gets worse | mixed batches; full eval suite at checkpoints (refusal check, BFCL, IFEval, GSM8K, HumanEval/MBPP, hand tests) |
| **Collapse** | entropy drops to near 0; every answer identical | small LR, KL penalty, monitor entropy |
| **vLLM / PyTorch mismatch** | ratios far from 1 at the start of an update | recompute π_old in PyTorch (§4.4) |

---

## 9. Where it runs and how long it takes

- **fire01 (one H100), one process:** the policy, the reference model and vLLM share the GPU, with vLLM asleep during training.
- **Setup on fire01:** the separate Python 3.12 environment with vLLM 0.33.0 and the plugin, the same as on the A10 (`~/vllm_env`). The training environment isn't touched.
- **Per step:** generation 10–25 s, training about 10 s, rewards about 5 s. **500 steps ≈ 5–7 hours** including evals.
- **Build order:**
  1. reward wrappers and task data, with the difficulty filter run by the SFT model;
  2. the GRPO trainer;
  3. a 20-step smoke test, then a report;
  4. go/no-go;
  5. the full run, with checkpoints uploaded to Hugging Face like every other phase.

  Prep is about 1–2 days.

**Why our own trainer instead of TRL or prime-rl:** those trainers expect Hugging Face `transformers` models, and ours is custom. A small GRPO loop around the verified vLLM engine is less work than porting the model to `transformers`. Prime Intellect's `verifiers` can still provide task environments and checkers, since that part doesn't depend on the model.

**Which SFT model RL starts from:** the winner of run 01 vs run 02b (doc/issue-02-phase3-over-refusal.md), decided by the chat evals and the refusal check.

---

## Glossary

| Term | Meaning |
|---|---|
| **Policy** | the model being trained by RL |
| **Rollout** | one sampled answer (here: from vLLM) |
| **Reward** | the checker's score for one rollout (usually 0 or 1) |
| **Advantage** | reward minus the group's average (optionally divided by the group's std): how much better than its siblings |
| **π_old / π_new** | the policy's token probabilities when the answer was sampled / after the update |
| **Ratio, clipping** | π_new / π_old, limited to [0.8, 1.2] in the loss so single updates stay small |
| **Reference model, KL** | the frozen SFT model; KL measures how far the policy's token distribution has moved from it |
| **RLVR** | RL with verifiable rewards: scores come from code checking known answers, not from a learned reward model |
| **GRPO** | group relative policy optimization: PPO-style updates with the group average as the baseline, so no value model |
