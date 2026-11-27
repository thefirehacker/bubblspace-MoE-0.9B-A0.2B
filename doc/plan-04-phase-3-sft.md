# Phase 3 (SFT) plan — draft

Status: **data mix v1 drafted 27 Nov 2026; prep starting.** Path: **SFT directly** after 2.5b, with no separate mid-training phase. The 2.5a decay (≈3B tokens of Q&A, agentic, code and maths at a falling LR) and 2.5b already play the role of OLMo-2's stage 2. NVIDIA's gated SFT sets are dropped (access still pending). OpenAthena references are dropped. **fire01 stays running** (H100s are hard to get back in its region). Data prep runs on the A10, and finished shards are copied to fire01 for training.

Order: `phase2-step22888` → **Phase 2.5a** (knowledge + relations) → **Phase 2.5b** (context 16384), both in [plan-03](plan-03-phase-2.5-knowledge-context.md) → **Phase 3 SFT at 16384** (this plan) → Phase 4 RL. SFT starts from the 2.5b final checkpoint.

## 1. What Phase 3 has to achieve

Phase 2 gave the model code and tool-use knowledge. It still behaves like a base model:
- it doesn't reliably stop at the end of an answer (`<|im_end|>`)
- it drifts into repetition
- it has to be prompted the way pretraining text looks

Phase 3 turns it into an assistant that answers inside the chat template, writes code when asked, and emits well-formed `<tool_call>` blocks. Phase 4 (RL) then improves *correctness* with sandbox rewards. SFT only has to get the format and behaviour right and keep the knowledge.

**Success criteria.** All are measured against the Phase 2.5b final checkpoint (with step 22888 as a reference), with the same deterministic eval suite plus two new evals.

| Eval | Base (22888) | Target after SFT |
|---|---|---|
| HumanEval / MBPP pass@1 (sandbox, chat-prompted) | 14.6% / 18.2% (step 22888) | ≥ base, completion-prompted; no format failures |
| BFCL simple / multiple / parallel | from final run | clearly up (format errors → ≈0) |
| **New:** IFEval-style instruction following | — | baseline + report |
| **New:** stop rate (ends with `<|im_end|>` within budget) | low | ≥ 95% |
| English 6-task average | ≈46.6 | no drop > 1 point |
| Probe v4 | from final run | no drop |
| Free-generation spot checks (chat REPL) | — | repetition-free, on-task |

## 2. Context: 16384 tokens (after Phase 2.5b)

Phases 1, 2 and 2.5a train at `seq_len 2048`. **Phase 2.5b ([plan-03](plan-03-phase-2.5-knowledge-context.md)) extends the context to 16384** (RoPE theta 1e6, ≈1.5B tokens), and SFT runs at **16384** from the 2.5b final checkpoint. The 16k smoke test passed (≈72k tok/s, 73 GB peak). **8192** is the fallback if 2.5b's long-context evals (needle retrieval, multi-hop) fail at 16k.

Many modern agent and reasoning traces are 8k–131k tokens:
- AgentTrove
- the penfever OT-Agent traces (32k / 131k)
- SWE-rebench
- Superior-Reasoning

At 16k far more of these fit whole than at 2048, but the 32k–131k traces still don't. Truncating a trajectory removes its outcome, which teaches the wrong thing. **So every candidate dataset still gets a length scan first,** and we keep only examples that fit whole in 16384 tokens (with our tokenizer and template).

## 3. Data shortlist (Marin token-count list)

Sizes are Marin's full-dataset token counts. **We'd use a small, filtered slice of each.** Formats and license terms below are from the CSV and still have to be confirmed by inspecting each dataset.

| Role | Dataset | Full size | License | Notes |
|---|---|---|---|---|
| Code instruction | `nvidia/Nemotron-Pretraining-SFT-v1` / sft_code | 56.7B | NVIDIA Data Agreement | high quality; **license needs your OK** |
| Code + math, verified | `PrimeIntellect/SYNTHETIC-1` | 7.3B | apache-2.0 | verifier-checked answers; also a Phase 4 task source |
| General chat / QA | `nvidia/Nemotron-Pretraining-SFT-v1` / sft_general | 85.2B | NVIDIA Data Agreement | same license question |
| Math with tools (TIR) | `AI-MO/NuminaMath-TIR` | 0.08B | apache-2.0 | short, code-interpreter style |
| Function calling | `AmazonScience/massive` (rendered as OpenAI tool calls) | 11.4B | cc-by-4.0 | simple intent/slot calls; good for format, low diversity → small slice |
| Terminal agent | `nvidia/Nemotron-Terminal-Corpus` | 6.1B | cc-by-4.0 | keep only traces ≤ 16384 |
| Terminal agent | `open-thoughts/AgentTrove` | 9.0B | apache-2.0 | mostly long; keep traces ≤ 16384 |
| Verified agent traces | penfever OT-Agent traces (qwen3.5-122b-32k etc.) | ≈4.4B total | OpenMDW | verifier-valid; mostly long; keep traces ≤ 16384 |
| Reasoning | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` | 7.1B | cc-by-4.0 | long CoT; length scan decides how many fit in 16k. Decide on `<think>` use |
| Reasoning / chat | `andyrdt/gpt-oss-20b-rollouts` | 3.2B | mit | inspect format |

**Excluded:**
- Raw code and web pools (stack-v3, nemotron_cc_*, starcoder2): pretraining text, not conversations.
- `marin-community/identity-data`: Marin's identity. We write our own small BubblSpace identity set (≈200 conversations).
- Legal, bio and DNA sets: off-target.
- safety_pt (refuseweb / safeweb): pretraining-style rephrasings. Revisit if we want refusal behaviour.

Phase 2 already used ToolMind, Glaive FC v2, Hermes FC, OpenThoughts and Bespoke-Stratos as pretraining text. **For SFT, held-out rows of the same sources are fine. Re-using the identical rows mostly re-teaches what the model already saw.**

**Target size:** ≈0.6–1.0B tokens, 1–2 epochs. At 16k throughput (≈72k tokens/s on the H100, from the 2.5b smoke test), that's **≈2.5–4 h** of training.

| Starting mix (to tune after length scans) | Share |
|---|---|
| Code instruction + verified code | 40% |
| Tool calling + short agent traces | 25% |
| General chat / instruction following | 25% |
| Math / reasoning | 8% |
| Identity | ≈0.1% |
| English replay (FineWeb-Edu, plain LM loss) to limit forgetting | ≈2% |

### 3.1 Data mix v1 (27 Nov), without NVIDIA's gated sets

Target ≈0.5–1B tokens (1–2 epochs over the curated pool), 16k context, assistant-only loss except the replay rows. The anchor is [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225), the SFT data behind OLMo-2-0425-1B-Instruct (16/16 on our calculation prompts and 14/18 relational, [handtest-01](handtest-01-relational-vs-olmo.md)). Rows carry a source label, so subsets with unwanted licence terms can be dropped.

| Share | Group | Sources | Licence |
|---|---|---|---|
| 31% | Code instruction | `PrimeIntellect/SYNTHETIC-1` (code, verified) + Tülu 3 code subsets | Apache-2.0 / per subset |
| 24% | General chat + instruction following | Tülu 3 OLMo-2 mix (general, instruction-following, safety subsets) | ODC-BY, filtered per subset |
| 22% | Tool calling + agent traces | held-out rows of Hermes FC / Glaive FC v2 / ToolMind, chosen for **multiple-function selection, parallel calls and multi-turn `<tool_response>`** (BFCL multiple was 6% and parallel 0% at 2.5a step 24000) + `AmazonScience/massive` (small) + `Nemotron-Terminal-Corpus` (CC-BY-4.0, ungated; only the ≈47% of traces ≤16k). `AgentTrove` dropped (27 Nov: GPT-5 teacher outputs, OpenAI terms). OT-Agent traces (≤16k) | Apache-2.0 / CC-BY-4.0 / OpenMDW |
| 10% | Maths, step by step | Tülu 3 maths subsets + `NuminaMath-TIR` + SYNTHETIC-1 maths + GSM8K train + **set C** | Apache-2.0 / MIT / ours |
| 3% | Binding and relational Q&A | **set D**: passives, "respectively", exclusion, state tracking | ours |
| 3% | Direct-answer facts | **set B**: `fact_seeking` quizzes rewritten as question → plain answer (against the quiz habit) | CC-BY-4.0 source |
| 3% | English replay (plain LM loss) | `en_wiki_full` + `fwe_edu4` (already tokenized on fire01) | — |
| 3% | Code replay (plain LM loss) | `code_python_edu_fresh` + `code_starcoder_python_fresh` (already on fire01) | — |
| ≈0.1% | Identity | ≈200 BubblSpace conversations (**set I**) | ours |

Reasoning style (default, pending your confirmation): **short worked steps inside the answer, no long `<think>` traces.** That's how OLMo-2 1B Instruct solved two-step problems ("3 − 1 = 2, 2 − 1 = 1"). `<think>` is left for Phase 4.

### 3.2 Prep order (on the A10, using all its cores. Moves to fire01 once fire01 is free, after 2.5b, or sooner if the A10 hits a compute limit. fire01 is never loaded while it trains. Finished shards are copied to `/lambda/nfs/h100-f1/data/phase3/` before the smoke run)

1. HF security status + 50-row inspection per external dataset.
2. Length scan at 16k with our tokenizer and template.
3. Generators for sets B, C, D, I (rule-based, CPU).
4. Render everything into our chat template, filter, dedup, decontaminate (HumanEval, MBPP, BFCL, IFEval).
5. Write token shards + loss-mask shards to `/lambda/nfs/h100-f1/data/phase3/`.
6. Training code: mask shards in `src/data/dataset.py`, mask unit test, `configs/phase3.yaml`.
7. Evals: chat-prompted HumanEval/MBPP, IFEval, stop rate, multi-turn tool use.
8. Smoke run on fire01 after 2.5b (≈200 steps) → report → **your go** → SFT (≈3–6 h).

**Timeline:** about 1 day to the smoke test. Most of the machinery exists: `scripts/prepare_agentic.py` already renders conversations and tool calls into our template and packs whole examples into blocks; `kind: block` pools already mask padding; `scripts/decontaminate.py` and `scripts/gen_contrastive_v4.py` are reused. The evals in step 7 don't block the start. They're built while SFT trains, since they're only needed to judge the result.

### 3.2.1 Prep status (night of 26–27 Nov)

| Pool (`/lambda/nfs/h100-f1/data/phase3/`, built on the A10) | Episodes | Tokens | Decon removed |
|---|---|---|---|
| `tulu3` (Aya removed) | 754k | 0.60B | 0.37% (HumanEval 1468, BFCL 441, CRUXEval 337, …) |
| `agentic_toolmind` (APIGen-MT removed: CC-BY-NC) | 325k | 0.82B | 0.26% (BFCL 602, …) |
| `agentic_glaive` (parser fixed: two- or three-newline turns) | 104k | 0.06B | 0.42% (BFCL 187, HumanEval 164) |
| `agentic_hermes` | 11k | 0.01B | 0.65% |
| `terminal` (≤16k, capped at 20k) | 20k | 0.19B | 3.5% (HumanEval 444, MBPP 100, BFCL 98) |
| `synthetic1` (gold solutions; verified code without reasoning traces) | 122k | 0.08B | 0.33% |
| `gsm8k` (train) | 7.5k | — | 0.13% (GSM8K test 9) |
| `own_facts` / `own_math` / `own_binding` / `own_identity` | 150k / 43k / 88k / 26 | 0.016B | 14 / 0 / 15 (hand-test wording) / 0 |

Tool-call coverage (ToolMind): 86% offer several tools, 22% parallel calls, 40% multi-step chains, 13% correctly make no call. Hermes adds 11% parallel; Glaive is single-call only. AgentTrove dropped (GPT-5 teacher outputs). `AmazonScience/massive` skipped (needs a loading script). NuminaMath-TIR unreachable (Tülu 3 covers maths).

Code: `scripts/phase3_build_sft.py`, `phase3_gen_own_sets.py`, `phase3_decon.py` (extended index: BFCL, IFEval, GSM8K test, hand tests), `phase3_toolcall_stats.py`; mask support in `src/data/dataset.py` + `tests/test_sft_mask.py` (13/13 tests pass); `configs/phase3.yaml` (1,500 steps, tool + agent ≈38%, replay 3% + 3%); `configs/phase3_smoke.yaml` (200 steps); the chat script's `/chat` now uses the SFT wire format. The smoke test runs automatically after 2.5b (`scripts/phase3_smoke_after_25b.sh`). The full SFT waits for the user's go.

### 3.3 Checkpoint uploads (standing, every stage)

When a stage ends, its final checkpoint is uploaded as a **full training state** to [`TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states`](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states):
- format: safetensors + JSON via `scripts/restore_checkpoint.py export --verify` (bit-exact)
- one branch per checkpoint, OLMo-style: `<phase>-step<N>-tokens<X>B`, where X = global step × 524,288

| Branch | Total tokens | Source |
|---|---|---|
| `phase1-step19073-tokens10B` | 10.0B | already on `main` (folder): server-side copy |
| `phase2-step19000-tokens20B` | 20.0B | already on `main`: server-side copy |
| `phase2-step22888-tokens22B` | 22.0B | already on `main`: server-side copy |
| `phase25a-step20000-tokens32B` | 32.5B | A10 `keep/` |
| `phase25a-step28000-tokens37B` / `step29000` | 36.7 / 37.2B | A10 / fire01 (once free) |
| `phase25a-step29723-tokens38B` (final) | 37.6B | relayed to the A10 |
| `phase25b-step2000-tokens39B`, `phase25b-step2861-tokens39B` (final) | 38.6 / 39.1B | relayed to the A10 |
| Phase 3 final, Phase 4 final, … | — | same procedure |

Each branch README records the phase, step, tokens, W&B run ID and the restore command.

## 4. Data pipeline (per dataset)

1. **HF security scan** (malware / pickle) before any download, as in Phase 2.
2. **Inspect 50 rows:** format, roles, tool schema, whether the outcome is included.
3. **Length scan** on ≈5k streamed rows with our tokenizer and template → fraction ≤ 16384 (and ≤ 8192, for the fallback). Drop a dataset if too little fits.
4. **Render to one chat template:** `<|im_start|>system/user/assistant/tool … <|im_end|>`, tools in `<tools>`, calls in `<tool_call>`, results in `<tool_response>`.
5. **Filter:**
   - drop proprietary-teacher rows where flagged
   - drop rows with empty or broken assistant turns
   - exact and near-dedup
6. **Decontaminate** against HumanEval, MBPP, BFCL and IFEval. Reuse the 20-token n-gram logic of `scripts/contamination_check_code.py` on rendered text. Drop hits.
7. **Write shards:** `uint16` token shards plus a parallel `uint8` loss-mask shard, in 16384-token blocks packed with whole examples only.

## 5. Training code changes

- **Assistant-only loss.** Loss on assistant tokens (including `<|im_end|>`, `<tool_call>` blocks and the stop token). System, user and tool-response tokens are masked to `IGNORE_INDEX`. `src/data/dataset.py` already has `kind: block` with pad masking, so this extends it with the mask shard.
- **Packing.** Whole examples only, never split across blocks. A per-example attention reset already exists (`build_doc_block_mask` with flex_attention, keyed on `<|endoftext|>`), so packed examples can't attend to each other.
- **Config `configs/phase3.yaml`:**
  - `--init_from` the Phase 2.5b final checkpoint (`step_22888.pt` was the plan before Phase 2.5 was added)
  - `seq_len 16384`, `rope_theta 1e6`, `micro_batch 2` (as in `configs/phase25b.yaml`)
  - fresh schedule: warmup 100 steps, peak LR ≈6e-5 (¼ of the Phase 2 plateau), linear decay to 0
  - router aux loss kept; routing watched for collapse on the narrower data
- **Tests:** a unit test that the mask covers exactly the assistant spans, plus a smoke run (200 steps) that overfits a tiny set.

## 6. Evaluation additions

- **Chat-prompted HumanEval/MBPP:** prompt in the template, extract the code block, run in the existing Docker sandbox.
- **IFEval** (instruction following), with deterministic scoring.
- **Stop-rate and repetition metrics** from fixed prompts.
- **Same eval watcher** on the A10 for every SFT checkpoint.

## 7. Phase 4 (RL) preview

The full plan is now [plan-06-phase-4-rl.md](plan-06-phase-4-rl.md) (27 Nov): task data, rewards, engineering status, time estimate.

GRPO-style RL with **sandbox-verified rewards** on code. Task sources:
- MBPP train
- SYNTHETIC-1 verified problems
- short penfever verifier tasks

Possibly also tool-call correctness rewards (BFCL-style AST match).

This needs fast sampling (many completions per prompt), so the **batched generator** stops being optional here. It's the main engineering item to build during or after Phase 3.

## 8. Open decisions (yours)

1. **NVIDIA Nemotron SFT sets:** dropped (27 Nov). Gated access is still pending, and Tülu 3 plus SYNTHETIC-1 cover their roles.
2. **Context:** decided 23 Nov: add Phase 2.5 before SFT (2.5b extends to 16384, fallback 8192).
3. **Reasoning style:** default short worked steps, no `<think>` (awaiting your confirmation).
4. **OpenAthena references:** dropped (27 Nov, your decision).
5. **fire01:** stays running (27 Nov). H100s are hard to get back in that region, so prep runs on it.
6. **Path:** SFT directly, with no separate mid-training phase (27 Nov).
7. **Our sets B, C, D, I:** default build all four (awaiting your confirmation).
