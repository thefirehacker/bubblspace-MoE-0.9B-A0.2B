# Phase 2.5 plan: knowledge and relations (2.5a) + context extension to 16k (2.5b)

Status: **data prep running** (started 23 Nov 2026). Branch `2.5.0_Phase2_a_b_FactualData_ContextLength8k`.

```
phase2-step22888 (Phase 2 final)
   │ --init_from
   ▼
2.5a: knowledge + semantic relations + code · ctx 2048 · 20B tokens
   │ --init_from (weights + optimizer state)
   ▼
2.5b: context extension · ctx 16384 · ~1.5B tokens (fallback 8192)
   │
   ▼
Phase 3 SFT at 16384 (plan-04) → Phase 4 RL
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

## 2. Phase 2.5a: knowledge + relations + code (context 2048, 20B tokens)

### 2.1 Budget: 20B tokens (decided 23 Nov)

| | |
|---|---|
| Throughput | ~84k tokens/s (Phase 2 measured) → 1B tokens ≈ 3.3 h, ~1,907 steps of 524,288 tokens |
| Length | **20B tokens = 38,147 steps, ~66 h** |
| Schedule | WSD: re-warm 300 steps → plateau at 2.0e-4 to step ~32,400 → linear decay over the final ~15% (~5,700 steps) to 6e-6 |
| Guard rule | fast-tier eval every 2,000 steps (~1B tokens). If facts, probes, code or English drop clearly, pause and adjust the mix (an exact resume, as in the Phase 2 Stage 2B switch). If everything has flattened well before 20B, the decay can start earlier. |

### 2.2 Mix

**By category (20B):**

| Category | Billion tokens | Share |
|---|---|---|
| Knowledge (English) | **11.2B** | 56% |
| Code | **5.0B** | 25% |
| Relations (English) | **1.8B** | 9% |
| Agentic / tool | **1.0B** | 5% |
| Math | **1.0B** | 5% |
| **Total** | **20.0B** | 100% |

About 65% of 2.5a is English text (knowledge + relations), so no separate "general English" group is needed.

**By source:**

| Group | Source | Share | Tokens trained | Unique prepared | Passes | License |
|---|---|---|---|---|---|---|
| Knowledge | **Wikipedia** full English (`wikimedia/wikipedia` 20231101.en) | 22% | 4.4B | ~5–6B (all) | ~0.8 | CC BY-SA 3.0 / GFDL |
| | **`fact_seeking`** (`nvidia/Nemotron-Pretraining-Specialized-v1.2`): MCQ about encyclopedic facts, written by Qwen3-30B-A3B, several questions per fact | 18% | 3.6B | 6.5B | 0.55 | CC BY 4.0 |
| | **FineWeb-Edu, `int_score ≥ 4`** (`sample-100BT`) | 8% | 1.6B | 2.5B | 0.64 | ODC-BY |
| | **SYNTH** (`PleIAs/SYNTH`, English rows only, query + answer; reasoning traces dropped) | 8% | 1.6B | 2.5B | 0.64 | CC BY 4.0 |
| Relations | contrastive corpus v3 | 2.5% | 0.5B | 0.3B | 1.7 | ours |
| | LLM-written relational passages | 3% | 0.6B | 0.5B | 1.2 | Qwen3 (Apache-2.0) output |
| | NLI / CommonsenseQA / ATOMIC as text | 1.5% | 0.3B | 0.25B | 1.2 | mostly CC BY / CC BY-SA |
| | ConceptNet / Wikidata as text | 2% | 0.4B | 0.5B | 0.8 | CC BY-SA / CC0 |
| Code | **Fresh Python-edu** (`Avelina/python-edu-cleaned`, skipping the 2,936,148 docs Phase 2 used) | 10% | 2.0B | up to 2.5B | ~0.8 | per-file permissive |
| | **Fresh StarCoderData Python** (`bigcode/starcoderdata` python, skipping 881,190 used docs) | 5% | 1.0B | 1.5B | 0.67 | per-file permissive |
| | **Fresh StarCoderData JS/TS** (skipping 939,475 / 1,616,913 used docs) | 1.5% | 0.3B | 0.5B | 0.6 | per-file permissive |
| | Phase 2 code pools (Python-edu, StarCoder Python/JS/TS/Go/SQL-shell-config; seen once) | 8.5% | 1.7B | 6.5B | 0.26 | — |
| Agentic | Phase 2 tool/agentic pools (ToolMind, Glaive FC v2, Hermes, OpenThoughts, Bespoke-Stratos) | 5% | 1.0B | 0.38B | 2.6 | — |
| Math | FineMath (Phase 2 pool) | 5% | 1.0B | 1.0B | 1.0 | — |

**Code: 5.0B, ~70% Python, ~3.3B never seen before.**
- With Phase 2's ~5.3B, the model will have seen ~10B code tokens.
- Python is weighted up because HumanEval/MBPP are Python.
- `nvidia/Nemotron-Pretraining-Code-v3` was checked and rejected: it holds only metadata (repo, path, commit), and the code would have to be fetched from GitHub.

**Agentic stays small** (2.6 passes, under the cap). Fresh tool-use data belongs in Phase 3 SFT.

**SYNTH replaces `diverse_qa`** (decided 23 Nov). SYNTH:
- 79.6M samples (~75B tokens), CC BY 4.0
- built by amplifying ~58.7k Wikipedia "Vital articles" ≥100× into varied queries with grounded answers (fine-tuned Qwen3-8B with the seed section in context), including negative (false-premise) queries

That's "core facts asked many ways", exactly the facts the model gets wrong. Things to keep in mind:
- **Narrow knowledge base** (58.7k articles), so it helps common facts more than long-tail trivia.
- **Exam-style format**, which flatters MMLU-like benchmarks.
- **Idiosyncratic reasoning-trace notation**, which is why we drop the traces.
- **Western-centric selection.**

Its files have no HF scan status, so they are scanned with ClamAV. If NVIDIA access for `diverse_qa` arrives later, SYNTH stays as is (same role, cleaner license), or the slot is split 0.8B / 0.8B.

**Indian languages are not in 2.5a.** They get a separate stage later (§6).

`wiki_rewrite` (in Marin's list) is **not** in NVIDIA's public repo: Specialized-v1 now holds only InfiniByte-Reasoning and Math-Textbooks. It's dropped.

### 2.3 Relational data (9%, 1.8B tokens)

Each source is capped at **≤ 4 passes over its unique tokens**. Contrastive v1 got memorized as a template when it was repeated too often.

| Source | What | Unique (target) |
|---|---|---|
| **Contrastive corpus v3** (`scripts/gen_contrastive_v3.py`, new) | v2's generator extended to new structures: reversed questions, passive voice, nested clauses, negation, comparatives, temporal order, part–whole, cause–effect. **Some structures are held out entirely** and go into probe v5. | ~0.3B |
| **LLM-written relational passages** (`scripts/gen_relational_llm.py`) | Qwen3-30B-A3B-Instruct-2507 via vLLM on the H100. 16 relation types × 8 formats × 4 reading levels. **70% grounded** on a real 120–350-word Wikipedia passage (only facts from the passage plus everyday knowledge), **30% everyday/fictional scenes** (correctness = internal consistency). Every output is fact-checked by the same model (short critique + `VERDICT: PASS/FAIL`, against the source when grounded), and outputs copying a 15-word stretch of the source are dropped. A probe-overlap filter keeps probes v2–v5 and the facts probe clean. To be published as `TheFireHacker/bubblspace-relational-passages-Qwen3-30B` (CC BY-SA 4.0, because the grounded part derives from Wikipedia). | ~0.25B kept (~15–18 h of H100) |
| **Human-written relational reasoning** | e-SNLI (entailment with explanations), MNLI, CommonsenseQA train, ATOMIC-2020 causal commonsense, each rendered as text. ANLI is skipped (non-commercial license). | ~0.2–0.3B |
| **Verbalized knowledge graphs** | ConceptNet (is-a, part-of, used-for, causes, capable-of) and Wikidata triples, each written in several phrasings ("Plants take in carbon dioxide and release oxygen.") | ~0.5B |

### 2.4 Pipeline per source

1. **Security scan.** Only files with HF status `safe` are used: Wikipedia 41/41, `fact_seeking` 8 safe files, Python-edu-cleaned 29/29. StarCoderData Python/JS and SYNTH files have **no HF scan result** (status `none`), so they are downloaded to fire01 and scanned with ClamAV first (StarCoderData: clean). Only the scanned local copies are tokenized.
2. **Tokenize** with `src/data/prepare.py`. It now supports:
   - `--data_files`: scan-safe file lists
   - `--min_int_score`: FineWeb-Edu quality filter
   - `--skip_docs`: start after the documents an earlier pool used
   - batched tokenization
3. **Decontaminate** against TriviaQA, NQ-open, ARC, HellaSwag, PIQA, WinoGrande, the facts probe and probes v2–v5. Use 20-token n-gram matching (as in `scripts/contamination_check_code.py`), dropping matching documents. Wikipedia naturally contains the answers to trivia questions. That's legitimate knowledge, so only question-text overlap is removed.
4. **Sanity check:** decode samples and measure the document-length distribution.

### 2.5 Data prep status

| Pool | Path (`/lambda/nfs/h100-f1/data/phase25/`) | Target | Status |
|---|---|---|---|
| Wikipedia full | `en_wiki_full` | all | ✅ 5.06B (6.41M articles) |
| fact_seeking | `fact_seeking` | 6.5B | running |
| FineWeb-Edu ≥ 4 | `fwe_edu4` | 2.5B | ✅ 2.50B (14% of docs pass the filter) |
| fresh Python-edu | `code_python_edu_fresh` | 2.5B | ✅ 2.42B (dataset exhausted) |
| fresh StarCoderData Python | `code_starcoder_python_fresh` | 1.5B | ✅ 1.50B (ClamAV clean) |
| fresh StarCoderData JS/TS | `code_starcoder_js_fresh`, `code_starcoder_ts_fresh` | 0.5B | ✅ 0.25B + 0.25B (ClamAV clean) |
| SYNTH (English, query + answer) | `synth_en` | 2.5B | downloading → ClamAV → tokenize |
| contrastive v3 | — | ~0.3B | generator to write |
| LLM relational passages | `raw/relational_llm` | ~0.25B kept | grounded test 3 running; full run after review |
| NLI / CSQA / ATOMIC | — | ~0.25B | to download + render |
| ConceptNet / Wikidata text | — | ~0.5B | to download + render |
| decontamination | — | — | after tokenization |

### 2.6 Evals (A10)

**Two tiers** keep evals ahead of training (a checkpoint every ~3.5 h):

| Tier | When | What | Time |
|---|---|---|---|
| **Fast** | every 2,000 steps (~1B tokens) | all probes, facts probe, TriviaQA (first 1,000), NQ-open (first 1,000), English 6 | ~1.5 h |
| **Full** | every ~4B tokens + final | full TriviaQA (2,000) / NQ-open (3,610), HumanEval/MBPP (sandbox), CRUXEval/BFCL, routing | ~3 h |

The extend-or-decay rule uses the fast tier. Probes are cheap (~1–2 min each), so adding more is fine.

**Prerequisite: KV-cache generation in the lm-eval adapter.** `generate_until` in `src/eval/lm_eval_adapter.py` re-runs the full prefix for every token. Porting the KV-cache loop from `scripts/eval_code_gen.py` should cut TriviaQA/NQ/facts time ~5–10×. Greedy output is deterministic, so it is verified by identical scores on step 22888.

**What's measured:**

- **Facts:** simple-facts probe (`scripts/eval_facts.py`), TriviaQA (5-shot, first 2,000), NQ-open (5-shot, all 3,610). Baselines: step 22888 and Phase 1 final, both running.
- **Relations:** probes v2, v3 (kept untouched as an outside test), v4, **v5** (new, from the held-out structures), plus a **relation-confusion probe** built from the facts-probe errors (input/output, part/whole, cause/effect).
- **Retention:** English 6 (no drop > 1 point), HumanEval/MBPP (sandbox), CRUXEval/BFCL at milestones, routing balance.
- **Free generation:** REPL spot checks on facts and relations (the WSP lesson: a scored probe alone isn't enough).

Expected: facts and probes v4/v5 rise. **v3 is the honest test and isn't promised.** If it stays flat, structural generalization is the open problem, and that gets reported as such.

## 3. Phase 2.5b: context extension to 16384

**How it works.**
1. **Stretch the position signal.** Raise `rope_theta` 10,000 → ~1,000,000 (ABF). RoPE has no learned weights. This slows the rotations so positions up to 16384 fall in angle ranges the model mostly recognises.
2. **Teach long-range use.** Continue at `seq_len 16384` on naturally long documents, where later text depends on much earlier text. Packed short documents don't teach this. Per-document attention masking (`build_doc_block_mask`, already in `src/model/attention.py`) keeps packed documents separate.
3. **Keep short skills** with mostly short data.

This follows ProLong ("How to Train Long-Context LMs (Effectively)", Gao et al. 2024) and the long-context stages of Llama 3 and Qwen. ProLong trained directly at the target length with mixed document lengths, so there is no 8k step first.

**Why 16k, not 8k.** 16k fits roughly twice as many agent/terminal/SWE traces whole (many are 8–32k), plus multi-file code and longer chats, for ~3 h more training. SFT and RL can still cap episode length lower where that's cheaper.

**Long data on disk.** Measured 23 Nov with `scripts/measure_doc_lengths.py`, first 300M tokens per pool, scaled to full size. Raw data: `benchmark/phase2/doc_lengths.json`.

| Pool | Pool size | Tokens in docs > 4k | > 8k | > 16k |
|---|---|---|---|---|
| FineWeb-Edu (Phase 1) | 10.0B | 2.46B | 1.36B | 0.53B |
| StarCoder Python | 1.25B | 0.57B | 0.35B | 0.20B |
| StarCoder JavaScript | 1.00B | 0.45B | 0.34B | 0.25B |
| StarCoder Go | 0.75B | 0.42B | 0.32B | 0.25B |
| StarCoder TypeScript | 1.25B | 0.40B | 0.27B | 0.19B |
| FineMath | 1.00B | 0.37B | 0.22B | 0.11B |
| Wikipedia (Phase 2 pool) | 1.50B | 0.38B | 0.18B | 0.06B |
| Python-edu | 1.50B | 0.18B | 0.07B | 0.03B |
| Cosmopedia, all agentic pools | — | 0 | 0 | 0 |
| **Total** | | **~5.2B** | **~3.1B** | **~1.6B** |

The need is ~0.5B long tokens, so ~1.6B in documents over 16k is enough without new downloads.

Full Wikipedia (2.5a) adds more long articles. The Phase 2 agentic pools were capped at ~3k tokens during prep, so long tool-use traces must come from new sources (Nemotron-Terminal, AgentTrove, penfever traces), after the length and security scans.

**Mix** (~1.5B tokens, ~9–10 h at an expected ~50% of Phase 2 throughput; attention cost per token grows with context, and doc masking makes the short share cheaper):

| Share | Content |
|---|---|
| ~35% | long documents (8k–16k+, some 4k–8k): FineWeb-Edu, full Wikipedia, StarCoder files, FineMath, plus long agent traces if they pass the scans |
| ~65% | short data taken from the 2.5a mix (doubles as a replay of the new knowledge), with doc masking |

**Settings:**
- `--init_from <2.5a final>`
- `rope_theta 1000000`, `seq_len 16384`, `micro_batch 2` (the same tokens per micro-batch as 16 × 2048)
- warmup 200 steps to ~1e-4, linear decay to ~1e-5

**Code changes:**
- `rope_theta` and `seq_len` per phase from config (check that no RoPE cache is stored in checkpoints)
- re-prepare `kind: block` shards at 16384
- raise length limits in the eval and generation scripts

**Evals:**
- passkey / needle retrieval at 2k, 4k, 8k and 16k
- **multi-hop retrieval** at the same lengths (two or more facts spread far apart, combined in one answer). Needle tests alone would overstate how well the model *uses* long context.
- loss by position on held-out long documents (should keep falling past 2048)
- the full 2.5a suite: no drop > 1 point

**Go/no-go:**
- Retrieval works at 16k and short evals hold → Phase 3 runs at 16384.
- Short evals drop → more short data or a lower LR, then re-run.
- 16k fails → fall back to 8192 (`rope_theta` 500k, a shorter re-run).
- Memory checked: micro-batch 2 × 16384 fits on the H100. The KV cache at 16k is ~0.5 GB per sequence (GQA, 4 KV heads), so evals and generation fit on the A10.

## 4. Timeline (estimate)

| Step | Time |
|---|---|
| Data prep: tokenize, generate relational data, decontaminate; KV-cache eval fix | ~1 day (parallel, CPU and the idle H100) |
| 2.5a | ~66 h (20B tokens) |
| 2.5b | 9–10 h (starts automatically) |
| Evals | parallel on the A10 |

## 5. Open items

1. NVIDIA access for Nemotron-CC-v2 (`diverse_qa`): optional now that SYNTH fills the slot.
2. Long agent-trace sources for 2.5b (length and security scans).
3. OpenAthena references (for Phase 3 data; may also add 2.5a sources).
4. Batched generator before Phase 4: RL rollouts at up to 16k are too slow one sequence at a time.

## 6. Later: Indian languages (separate stage, decided 23 Nov)

Hindi, Marathi, Punjabi and Bengali are **not** added to 2.5a. The SmolLM2 tokenizer splits these scripts into byte-level fragments (measured on the same sentence):

| Language | Tokens vs English | Tokens per word |
|---|---|---|
| English | 1.0× | 1.2 |
| Hindi | 5.1× | 6.2 |
| Marathi | 5.5× | 6.8 |
| Punjabi | 8.0× | 9.8 |
| Bengali | 7.7× | 12.1 |

5B tokens would hold only ~0.7–1B tokens' worth of English-equivalent content, spread over four languages.

**Planned approach:**
1. Extend the tokenizer with ~15–25k Indic tokens. English and code tokenization stays identical, so existing data and checkpoints remain valid.
2. Grow the tied embedding table (~+20M parameters). Initialize new rows from their byte-piece embeddings.
3. Run a dedicated ~10–15B-token Indic stage.
   - Candidate data: AI4Bharat Sangraha, FineWeb-2 (`hin_Deva`, `mar_Deva`, `pan_Guru`, `ben_Beng`), IndicCorp v2, `nvidia/Nemotron-Pretraining-Indic-CC-v1` (gated), `Indic-Specialized-v1`. Each needs the license check and security scan.
   - Evals: Belebele (all four), IndicXNLI, FLORES perplexity.
