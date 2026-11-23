# Phase 2.5 (context extension) + Phase 3 (SFT) plan — draft

Status: **draft, data shortlist open** (23 Nov 2026). Marin refs reviewed; OpenAthena refs still to come.

Order: `phase2-step22888` (Phase 2 final) → **Phase 2.5**: extend context 2048 → 8192 → **Phase 3**: SFT at 8192 → Phase 4: RL.

## 0. Phase 2.5: context extension to 8192 tokens

**Why.** The model was trained at 2048 tokens. Most agent and tool-use traces, multi-file code and long chats don't fit in that. SFT and RL at 2048 would have to throw them away (see §2). Extending first lets Phase 3 and 4 train on whole episodes.

**How it works.**
1. **Stretch the position signal.** Raise `rope_theta` from 10,000 to ~500,000 ("ABF"). RoPE has no learned weights, so this is a config change. It slows the rotations so positions up to 8192 fall in angle ranges the model mostly recognises.
2. **Teach long-range use.** Continue training at `seq_len 8192` on documents that are *naturally* long, where later text depends on much earlier text. Packing short documents into 8k windows doesn't teach this. Per-document attention masking (`build_doc_block_mask`, already in `src/model/attention.py`) keeps packed short documents separate.
3. **Keep short skills.** Most of the mix stays the normal short Phase 2 data.

This follows the published recipes: ProLong ("How to Train Long-Context LMs (Effectively)", Gao et al. 2024) and the long-context stages of Llama 3 and Qwen.

**Long data already on disk.** Measured on 23 Nov with `scripts/measure_doc_lengths.py`: first 300M tokens per pool, scaled to full pool size. Raw output is in `benchmark/phase2/doc_lengths.json`.

| Pool | Pool size | Tokens in docs > 4k | Tokens in docs > 8k |
|---|---|---|---|
| FineWeb-Edu (Phase 1) | 10.0B | 2.46B | 1.36B |
| StarCoder Python | 1.25B | 0.57B | 0.35B |
| StarCoder JavaScript | 1.00B | 0.45B | 0.34B |
| StarCoder Go | 0.75B | 0.42B | 0.32B |
| StarCoder TypeScript | 1.25B | 0.40B | 0.27B |
| FineMath | 1.00B | 0.37B | 0.22B |
| Wikipedia | 1.50B | 0.38B | 0.18B |
| Python-edu | 1.50B | 0.18B | 0.07B |
| Cosmopedia, all agentic pools | — | 0 | 0 |
| **Total** | | **~5.2B** | **~3.1B** |

- **Enough long text already exists**: about 3.1B tokens in documents over 8k, against a need of ~0.4–0.6B. No new download is needed for prose or single-file code. It has all been seen in Phase 1 or 2, but one more pass over a small part of it is not a concern.
- **No long agent data yet.** The Phase 2 agentic pools were capped at ~3k tokens during prep. Long tool-use traces have to come from new sources: Nemotron-Terminal, AgentTrove, penfever traces.
- **Optional additions** to check: repo-level code (files of one repo concatenated, real cross-file dependencies), and books/PDFs such as `princeton-nlp/prolong-data-64K`, finepdfs or institutional_books. Each needs a size/license check and the malware scan.

**Mix and budget** (~1.2B tokens, ~5–7 h on the H100 at an expected 60–75% of Phase 2 throughput):

| Share | Content |
|---|---|
| ~35% | long documents (4k–8k+): FineWeb-Edu, StarCoder files, FineMath, Wikipedia, plus long agent traces if they pass the scan |
| ~65% | the Stage 2B short mix (English / code / agentic), with doc masking |

**Settings:**
- `--init_from step_22888.pt`
- `rope_theta 500000`, `seq_len 8192`, `micro_batch 4` (the same tokens per micro-batch as 16 × 2048)
- LR warmup 200 steps to ~1e-4, then linear decay to ~1e-5

**Code changes:**
- `rope_theta` and `seq_len` read per phase from config. Check that no RoPE cache is stored in checkpoints.
- re-prepare `kind: block` shards at 8192
- raise length limits in the eval and generation scripts

**Evals:**
- passkey / needle retrieval at 2k, 4k and 8k
- perplexity by position on held-out long documents (it should keep falling past 2048)
- the full existing suite (English 6, probes, CRUXEval/BFCL, HumanEval/MBPP): no drop > 1 point

**Go/no-go:**
- If needle retrieval at 8k works and short evals hold, Phase 3 runs at 8192.
- If short evals drop, add more short data or lower the LR, and re-run.
- If 8k fails outright, fall back to 4096.

## 1. What Phase 3 has to achieve

Phase 2 gave the model code and tool-use knowledge. It still behaves like a base model:
- it doesn't reliably stop at the end of an answer (`<|im_end|>`)
- it drifts into repetition
- it has to be prompted the way pretraining text looks

Phase 3 turns it into an assistant that answers inside the chat template, writes code when asked, and emits well-formed `<tool_call>` blocks. Phase 4 (RL) then improves *correctness* with sandbox rewards. SFT only has to get the format and behaviour right and keep the knowledge.

**Success criteria.** All are measured against step 22888, with the same deterministic eval suite plus two new evals.

| Eval | Base (22888) | Target after SFT |
|---|---|---|
| HumanEval / MBPP pass@1 (sandbox, chat-prompted) | from final run | ≥ base, completion-prompted; no format failures |
| BFCL simple / multiple / parallel | from final run | clearly up (format errors → ~0) |
| **New:** IFEval-style instruction following | — | baseline + report |
| **New:** stop rate (ends with `<|im_end|>` within budget) | low | ≥ 95% |
| English 6-task average | ~46.6 | no drop > 1 point |
| Probe v4 | from final run | no drop |
| Free-generation spot checks (chat REPL) | — | repetition-free, on-task |

## 2. Hard constraint: 2048-token context

The model was trained at `seq_len 2048`. Most modern agent and reasoning traces are 8k–131k tokens:
- AgentTrove
- the penfever OT-Agent traces (32k / 131k)
- SWE-rebench
- Superior-Reasoning

Truncating a trajectory removes its outcome, which teaches the wrong thing. **So every candidate dataset gets a length scan first.** We keep only examples that fit whole in 2048 tokens (with our tokenizer and template).

**Phase 2.5 (§0) raises this to 8192.** The length scan then measures the fraction that fits in 8192, and 2048 stays the fallback if Phase 2.5 fails.

## 3. Data shortlist (Marin token-count list)

Sizes are Marin's full-dataset token counts. **We'd use a small, filtered slice of each.** Formats and license terms below are from the CSV and still have to be confirmed by inspecting each dataset.

| Role | Dataset | Full size | License | Notes |
|---|---|---|---|---|
| Code instruction | `nvidia/Nemotron-Pretraining-SFT-v1` / sft_code | 56.7B | NVIDIA Data Agreement | high quality; **license needs your OK** |
| Code + math, verified | `PrimeIntellect/SYNTHETIC-1` | 7.3B | apache-2.0 | verifier-checked answers; also a Phase 4 task source |
| General chat / QA | `nvidia/Nemotron-Pretraining-SFT-v1` / sft_general | 85.2B | NVIDIA Data Agreement | same license question |
| Math with tools (TIR) | `AI-MO/NuminaMath-TIR` | 0.08B | apache-2.0 | short, code-interpreter style |
| Function calling | `AmazonScience/massive` (rendered as OpenAI tool calls) | 11.4B | cc-by-4.0 | simple intent/slot calls; good for format, low diversity → small slice |
| Terminal agent | `nvidia/Nemotron-Terminal-Corpus` | 6.1B | cc-by-4.0 | keep only traces ≤ 2048 |
| Terminal agent | `open-thoughts/AgentTrove` | 9.0B | apache-2.0 | mostly long; short ones only |
| Verified agent traces | penfever OT-Agent traces (qwen3.5-122b-32k etc.) | ~4.4B total | OpenMDW | verifier-valid; mostly long; short ones only |
| Reasoning | `Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b` | 7.1B | cc-by-4.0 | long CoT; probably few fit. Decide on `<think>` use |
| Reasoning / chat | `andyrdt/gpt-oss-20b-rollouts` | 3.2B | mit | inspect format |

**Excluded:**
- Raw code and web pools (stack-v3, nemotron_cc_*, starcoder2): pretraining text, not conversations.
- `marin-community/identity-data`: Marin's identity. We write our own small BubblSpace identity set (~200 conversations).
- Legal, bio and DNA sets: off-target.
- safety_pt (refuseweb / safeweb): pretraining-style rephrasings. Revisit if we want refusal behaviour.

Phase 2 already used ToolMind, Glaive FC v2, Hermes FC, OpenThoughts and Bespoke-Stratos as pretraining text. **For SFT, held-out rows of the same sources are fine. Re-using the identical rows mostly re-teaches what the model already saw.**

**Target size:** ~0.6–1.0B tokens, 1–2 epochs. At Phase 2 throughput (~84k tokens/s on the H100), that's **~2–4 h** of training.

| Starting mix (to tune after length scans) | Share |
|---|---|
| Code instruction + verified code | 40% |
| Tool calling + short agent traces | 25% |
| General chat / instruction following | 25% |
| Math / reasoning | 8% |
| Identity | ~0.1% |
| English replay (FineWeb-Edu, plain LM loss) to limit forgetting | ~2% |

## 4. Data pipeline (per dataset)

1. **HF security scan** (malware / pickle) before any download, as in Phase 2.
2. **Inspect 50 rows:** format, roles, tool schema, whether the outcome is included.
3. **Length scan** on ~5k streamed rows with our tokenizer and template → fraction ≤ 2048. Drop a dataset if too little fits.
4. **Render to one chat template:** `<|im_start|>system/user/assistant/tool … <|im_end|>`, tools in `<tools>`, calls in `<tool_call>`, results in `<tool_response>`.
5. **Filter:**
   - drop proprietary-teacher rows where flagged
   - drop rows with empty or broken assistant turns
   - exact and near-dedup
6. **Decontaminate** against HumanEval, MBPP, BFCL and IFEval. Reuse the 20-token n-gram logic of `scripts/contamination_check_code.py` on rendered text. Drop hits.
7. **Write shards:** `uint16` token shards plus a parallel `uint8` loss-mask shard, in 2048-token blocks packed with whole examples only.

## 5. Training code changes

- **Assistant-only loss.** Loss on assistant tokens (including `<|im_end|>`, `<tool_call>` blocks and the stop token). System, user and tool-response tokens are masked to `IGNORE_INDEX`. `src/data/dataset.py` already has `kind: block` with pad masking, so this extends it with the mask shard.
- **Packing.** Whole examples only, never split across blocks. A per-example attention reset already exists (`build_doc_block_mask` with flex_attention, keyed on `<|endoftext|>`), so packed examples can't attend to each other.
- **Config `configs/phase3.yaml`:**
  - `--init_from step_22888.pt`
  - fresh schedule: warmup 100 steps, peak LR ~6e-5 (¼ of the Phase 2 plateau), linear decay to 0
  - router aux loss kept; routing watched for collapse on the narrower data
- **Tests:** a unit test that the mask covers exactly the assistant spans, plus a smoke run (200 steps) that overfits a tiny set.

## 6. Evaluation additions

- **Chat-prompted HumanEval/MBPP:** prompt in the template, extract the code block, run in the existing Docker sandbox.
- **IFEval** (instruction following), with deterministic scoring.
- **Stop-rate and repetition metrics** from fixed prompts.
- **Same eval watcher** on the A10 for every SFT checkpoint.

## 7. Phase 4 (RL) preview

GRPO-style RL with **sandbox-verified rewards** on code. Task sources:
- MBPP train
- SYNTHETIC-1 verified problems
- short penfever verifier tasks

Possibly also tool-call correctness rewards (BFCL-style AST match).

This needs fast sampling (many completions per prompt), so the **batched generator** stops being optional here. It's the main engineering item to build during or after Phase 3.

## 8. Open decisions (yours)

1. **NVIDIA Data Agreement license** for the Nemotron SFT sets: OK to use, or apache/cc-by sources only?
2. **Context:** decided 23 Nov: add Phase 2.5 (8192) before SFT.
3. **Reasoning style:** train `<think>…</think>` traces, or plain answers only?
4. **OpenAthena references:** pending. The shortlist gets finalized after those.
5. **fire01:** keep running for prep, or stop until the data is ready.
