# Phase 2.5 plan: knowledge and relations (2.5a) + context extension to 8k (2.5b)

Status: **data prep running** (started 23 Nov 2026). Branch `2.5.0_Phase2_a_b_FactualData_ContextLength8k`.

```
phase2-step22888 (Phase 2 final)
   │ --init_from
   ▼
2.5a: knowledge + semantic relations · ctx 2048 · 10–20B tokens (open-ended plateau, decided by evals)
   │ --init_from (weights + optimizer state)
   ▼
2.5b: context extension · ctx 8192 · ~1.2B tokens
   │
   ▼
Phase 3 SFT at 8192 (plan-04) → Phase 4 RL
```

These are two launches, not one run. 2.5b changes sequence length and RoPE base, which `train.py` stages can't switch mid-run. Weights and optimizer state carry over, as they did from Phase 1 to Phase 2. A chain script starts 2.5b as soon as 2.5a's final checkpoint is written. 2.5a's evals run on the A10 in parallel, and 2.5b is stopped if they show a problem.

## 1. Why

**Facts.** In the chat REPL, step 22888 produces confident factual errors (Tagore's Nobel "in 1923", Everest "in Uttar Pradesh"). The closed-book baseline shows the picture is mixed:
- **Simple-facts probe:** 82/95 (86%) when asked in a clean Q/A format. Capitals 25/25, science 13/20.
- **Harder trivia:** TriviaQA and NQ-open are low (8% / 4% on a 50-question smoke test; full runs pending).

The errors are mostly **confusions between related concepts**:
- oxygen vs CO₂, oxygen vs nitrogen
- nucleus vs mitochondria
- Venus vs Mars
- Galileo vs Newton
- Ashoka vs Chandragupta

Part of what looked like missing knowledge in the REPL is base-model rambling, which SFT fixes. The rest is too little exposure: ~22B tokens seen in total, mostly general web text.

**What the research says** (Allen-Zhu & Li, *Physics of Language Models* 3.1/3.3):
- Models store roughly 2 bits of knowledge per parameter. With 915M total parameters, size isn't the bottleneck.
- A fact needs many exposures (on the order of hundreds) before it is stored reliably.
- The same fact **in varied phrasings, including Q&A**, is what makes it retrievable by a question rather than only memorized as a sentence.

**Semantic relations.** In the Phase 2 anneal, contrastive corpus v2 lifted probe v4 (same relation types, new frames/names) from 17 to 82/112. Probe v3 (new sentence structures) stayed flat at 4 → 6/70. Binding carried over to new words and frames but **not to new structures**. More of the same templates won't fix that. Structural variety and naturally written relational text are what's missing.

**Context.** Agent traces, multi-file code and long chats don't fit in 2048 tokens. Measured doc-length data is in §3.

## 2. Phase 2.5a: knowledge + relations (context 2048)

### 2.1 Budget: prepare for 20B, train at least 10B, decay when the evals flatten

| | |
|---|---|
| Throughput | ~84k tokens/s (Phase 2 measured) → 1B tokens ≈ 3.3 h, ~1,907 steps of 524,288 tokens |
| Minimum | 10B tokens (19,073 steps, ~33 h) |
| Maximum | 20B tokens (38,147 steps, ~66 h) |
| Schedule | WSD: re-warm 300 steps → plateau at 2.0e-4 → linear decay over the final ~15% to 6e-6 |
| Decision rule | eval every 2,000 steps (~1B tokens). While TriviaQA/NQ, facts or probes v3/v5 improve clearly per 1B tokens, extend the plateau (up to 20B). When they flatten, set the decay start, which is an exact resume because the plateau LR is constant. If code/agentic evals drop, raise replay or stop. |

Data is prepared for the 20B maximum up front, so extending is never blocked by missing data.

### 2.2 Mix

| Share | Source | Role | License | Prep for 20B |
|---|---|---|---|---|
| 25% | **Wikipedia** full English (`wikimedia/wikipedia` 20231101.en) | core facts, more passes | CC BY-SA 3.0 / GFDL | all of it, ~5–8B unique |
| 20% | **`fact_seeking`** (`nvidia/Nemotron-Pretraining-Specialized-v1.2`) | each fact asked several ways (MCQ written by Qwen3-30B-A3B from encyclopedic docs) | CC BY 4.0 | 6.5B (8 scan-safe files) |
| 10% | **FineWeb-Edu, `int_score ≥ 4`** (`sample-100BT`) | high-quality explanatory prose | ODC-BY | 2.5B |
| 10% | **`diverse_qa`** (`nvidia/Nemotron-CC-v2`): Q&A rewrites of web text. **Fallback:** finepdfs (ODC-BY) or more Wikipedia | rephrasing / Q&A | NVIDIA Data Agreement (access request pending) | 2.5B |
| 10% | **Relational data**, see 2.3 | semantic relations and binding | various | ~1.5B unique |
| 25% | **Replay** of the Stage 2B pools: code 15%, tool/agentic 5%, math 5% | keep Phase 2 skills | already on disk | — |

`wiki_rewrite` (in Marin's list) is **not** in NVIDIA's public repo: Specialized-v1 now holds only InfiniByte-Reasoning and Math-Textbooks. It's dropped.

### 2.3 Relational data (10%)

Each source is capped at **≤ 4 passes over its unique tokens**. Contrastive v1 got memorized as a template when it was repeated too often.

| Source | What | Unique (target) |
|---|---|---|
| **Contrastive corpus v3** (`scripts/gen_contrastive_v3.py`, new) | v2's generator extended to new structures: reversed questions, passive voice, nested clauses, negation, comparatives, temporal order, part–whole, cause–effect. **Some structures are held out entirely** and go into probe v5. | ~0.3B |
| **LLM-written relational passages** | Qwen3-30B-A3B (Apache-2.0) via vLLM on the H100 during data prep. Naturally phrased minimal-pair passages and explanations over many relation types, with diverse topics and structures. The probe v3/v4/v5 frames, names and pairs are banned from the prompts. | ~0.5B (a few to ~10 h of H100) |
| **Human-written relational reasoning** | e-SNLI (entailment with explanations), MNLI, CommonsenseQA train, ATOMIC-2020 causal commonsense, each rendered as text. ANLI is skipped (non-commercial license). | ~0.2–0.3B |
| **Verbalized knowledge graphs** | ConceptNet (is-a, part-of, used-for, causes, capable-of) and Wikidata triples, each written in several phrasings ("Plants take in carbon dioxide and release oxygen.") | ~0.5B |

### 2.4 Pipeline per source

1. **HF security scan.** Only files with status `safe` are used. Wikipedia: 41/41 safe. `fact_seeking`: 41 safe, 5 queued, so we use 8 safe files.
2. **Tokenize** with `src/data/prepare.py`. It now supports `--data_files` (scan-safe file lists), `--min_int_score`, and batched tokenization.
3. **Decontaminate** against TriviaQA, NQ-open, ARC, HellaSwag, PIQA, WinoGrande, the facts probe and probes v2–v5. Use 20-token n-gram matching (as in `scripts/contamination_check_code.py`), dropping matching documents. Wikipedia naturally contains the answers to trivia questions. That's legitimate knowledge, so only question-text overlap is removed.
4. **Sanity check:** decode samples and measure the document-length distribution.

### 2.5 Data prep status

| Pool | Path (`/lambda/nfs/h100-f1/data/phase25/`) | Target | Status |
|---|---|---|---|
| Wikipedia full | `en_wiki_full` | all (cap 8B) | running |
| fact_seeking | `fact_seeking` | 6.5B | running |
| FineWeb-Edu ≥ 4 | `fwe_edu4` | 2.5B | running (14% of docs pass the filter) |
| diverse_qa / fallback | — | 2.5B | waiting for NVIDIA access |
| contrastive v3 | — | ~0.3B | generator to write |
| LLM relational passages | — | ~0.5B | vLLM job to set up (H100 is free now) |
| NLI / CSQA / ATOMIC | — | ~0.25B | to download + render |
| ConceptNet / Wikidata text | — | ~0.5B | to download + render |
| decontamination | — | — | after tokenization |

### 2.6 Evals (A10, every 2,000 steps)

- **Facts:** simple-facts probe (`scripts/eval_facts.py`), TriviaQA (5-shot, first 2,000), NQ-open (5-shot, all 3,610). Baselines: step 22888 and Phase 1 final, both running.
- **Relations:** probes v2, v3 (kept untouched as an outside test), v4, **v5** (new, from the held-out structures), plus a **relation-confusion probe** built from the facts-probe errors (input/output, part/whole, cause/effect).
- **Retention:** English 6 (no drop > 1 point), HumanEval/MBPP (sandbox), CRUXEval/BFCL at milestones, routing balance.
- **Free generation:** REPL spot checks on facts and relations (the WSP lesson: a scored probe alone isn't enough).

Expected: facts and probes v4/v5 rise. **v3 is the honest test and isn't promised.** If it stays flat, structural generalization is the open problem, and that gets reported as such.

## 3. Phase 2.5b: context extension to 8192

**How it works.**
1. **Stretch the position signal.** Raise `rope_theta` 10,000 → ~500,000 (ABF). RoPE has no learned weights. This slows the rotations so positions up to 8192 fall in angle ranges the model mostly recognises.
2. **Teach long-range use.** Continue at `seq_len 8192` on naturally long documents, where later text depends on much earlier text. Packed short documents don't teach this. Per-document attention masking (`build_doc_block_mask`, already in `src/model/attention.py`) keeps packed documents separate.
3. **Keep short skills** with mostly short data.

This follows ProLong ("How to Train Long-Context LMs (Effectively)", Gao et al. 2024) and the long-context stages of Llama 3 and Qwen.

**Long data on disk.** Measured 23 Nov with `scripts/measure_doc_lengths.py`, first 300M tokens per pool, scaled to full size. Raw data: `benchmark/phase2/doc_lengths.json`.

| Pool | Pool size | Tokens in docs > 4k | Tokens in docs > 8k |
|---|---|---|---|
| FineWeb-Edu (Phase 1) | 10.0B | 2.46B | 1.36B |
| StarCoder Python | 1.25B | 0.57B | 0.35B |
| StarCoder JavaScript | 1.00B | 0.45B | 0.34B |
| StarCoder Go | 0.75B | 0.42B | 0.32B |
| StarCoder TypeScript | 1.25B | 0.40B | 0.27B |
| FineMath | 1.00B | 0.37B | 0.22B |
| Wikipedia (Phase 2 pool) | 1.50B | 0.38B | 0.18B |
| Python-edu | 1.50B | 0.18B | 0.07B |
| Cosmopedia, all agentic pools | — | 0 | 0 |
| **Total** | | **~5.2B** | **~3.1B** |

Full Wikipedia (2.5a) adds more long articles. The Phase 2 agentic pools were capped at ~3k tokens during prep, so long tool-use traces must come from new sources (Nemotron-Terminal, AgentTrove, penfever traces), after the length and security scans.

**Mix** (~1.2B tokens, ~5–7 h at an expected 60–75% of Phase 2 throughput):

| Share | Content |
|---|---|
| ~35% | long documents (4k–8k+): FineWeb-Edu, full Wikipedia, StarCoder files, FineMath, plus long agent traces if they pass the scans |
| ~65% | short data taken from the 2.5a mix (doubles as a replay of the new knowledge), with doc masking |

**Settings:**
- `--init_from <2.5a final>`
- `rope_theta 500000`, `seq_len 8192`, `micro_batch 4` (the same tokens per micro-batch as 16 × 2048)
- warmup 200 steps to ~1e-4, linear decay to ~1e-5

**Code changes:**
- `rope_theta` and `seq_len` per phase from config (check that no RoPE cache is stored in checkpoints)
- re-prepare `kind: block` shards at 8192
- raise length limits in the eval and generation scripts

**Evals:**
- passkey / needle retrieval at 2k, 4k and 8k
- loss by position on held-out long documents (should keep falling past 2048)
- the full 2.5a suite: no drop > 1 point

**Go/no-go:**
- Needle retrieval works at 8k and short evals hold → Phase 3 runs at 8192.
- Short evals drop → more short data or a lower LR, then re-run.
- 8k fails → fall back to 4096.

## 4. Timeline (estimate)

| Step | Time |
|---|---|
| Data prep: tokenize, generate relational data, decontaminate | ~1 day (parallel, CPU and the idle H100) |
| 2.5a | 33–66 h depending on the decision rule |
| 2.5b | 5–7 h (starts automatically) |
| Evals | parallel on the A10 |

## 5. Open items

1. NVIDIA access for Nemotron-CC-v2 (`diverse_qa`). If it doesn't arrive before prep finishes, the fallback is used.
2. Long agent-trace sources for 2.5b (length and security scans).
3. OpenAthena references (for Phase 3 data; may also add 2.5a sources).
