# Data pipeline: every dataset, phase by phase

The single record of what data BubblSpace MoE-0.9B-A0.2B was trained on, how each pool was built, and what is planned next. **Updated whenever a pool is added or a mix changes.** Last update: 23 Nov 2026 (Phase 2.5a data prep in progress).

Related: [plan-03 (Phase 2.5a/b)](plan-03-phase-2.5-knowledge-context.md) · [plan-04 (Phase 3 SFT)](plan-04-phase-3-sft.md) · [storyboard 02.5 (Phase 1)](storyboard/02.5-Phase1.md) · [storyboard 04 (Phase 2)](storyboard/04-Phase2-Trainig.md) · [`configs/phase2.yaml`](../configs/phase2.yaml)

## 1. Conventions

| Item | Value |
|---|---|
| **Tokenizer** | `bubblspace-tokenizer`: SmolLM2's 49,152-token BPE + 12 special tokens (`<\|im_start\|>`, `<\|im_end\|>`, FIM, `<tools>`, `<tool_call>`, `<tool_response>`, `<think>` …). 49,164 real tokens. |
| **Shard format** | `uint16` token streams, `shard_XXXXX.bin`, 100M tokens per shard, documents separated by `<\|endoftext\|>` (id 0), plus a `meta.json` (source, config, filters, token count) |
| **Block pools** (`kind: block`) | agentic/reasoning episodes packed whole into 2048-token blocks, tail-padded (pad id 49155, targets masked); episodes longer than 2048 are dropped |
| **Builders** | `src/data/prepare.py` (streaming tokenizer: `--data_files`, `--min_int_score`, `--skip_docs`, `--text_template`, `--where`, batched tokenization) · `scripts/prepare_agentic.py` (chat-template rendering of tool/reasoning sets) · `scripts/gen_contrastive*.py` · `scripts/gen_relational_llm.py` |
| **Storage** | `/lambda/nfs/h100-f1/data/` (tokenized), `/lambda/nfs/h100-f1/raw/` (scanned raw downloads). The filesystem persists across instances. |
| **Security** | **Every new dataset is malware-checked before use.** Files with HF scan status `safe` are used directly. Files with no HF status are downloaded and scanned with **ClamAV** on fire01, and only the scanned local copy is tokenized. |
| **Decontamination** | Code pools were checked against HumanEval/MBPP (`scripts/contamination_check_code.py`, 20-token windows). Phase 2.5 pools get the same check against TriviaQA, NQ-open, ARC, HellaSwag, PIQA, WinoGrande, the facts probe and probes v2–v5 before training. |
| **Revisions** | The commit hashes below are the repos' `main` as checked on 23 Nov 2026. Phase 2.5 pools were downloaded that day. Phase 1/2 pools were downloaded earlier, so their hash is the best available record. |

## 2. Summary

| Phase | Tokens trained | Main data | Status |
|---|---|---|---|
| **Phase 1** (pretraining) | 10.0B | FineWeb-Edu `sample-10BT` | ✅ done (step 19,073) |
| **Phase 2** (continual: English + code + agentic + contrastive) | 12.0B | Wikipedia, Cosmopedia v2, FineMath, FineWeb-Edu, Python-edu, StarCoderData (6 languages + config/shell), 5 tool/reasoning sets, contrastive v1/v2 | ✅ done (step 22,888) |
| **Phase 2.5a** (knowledge + relations + code) | 20.0B planned | full Wikipedia, fact_seeking, FineWeb-Edu ≥ 4, SYNTH, fresh code, relational sources, replay | 🔄 data prep |
| **Phase 2.5b** (context 16k) | ~1.5B planned | long documents (8k–16k+) from existing pools + short replay from 2.5a | ⏳ planned |
| **Phase 3** (SFT) | ~0.6–1.0B planned | chat / code / tool-use conversations | ⏳ planned ([plan-04](plan-04-phase-3-sft.md)) |
| **Indic stage** | ~10–15B planned | Hindi, Marathi, Punjabi, Bengali (needs tokenizer extension) | ⏳ later (plan-03 §6) |

## 3. Phase 1: pretraining (10.0B tokens)

| Pool | Source | Config | Tokens | License | Revision |
|---|---|---|---|---|---|
| `data/` (root, 100 shards) | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | 10.000B | ODC-BY | `87f09149ef47` |

- **One pass**, warmup + cosine schedule, context 2048. English only.
- **Document lengths** (measured 23 Nov): median 628 tokens. 24.6% of tokens sit in documents over 4k, 13.7% over 8k, 5.3% over 16k. That's what 2.5b reuses.

## 4. Phase 2: continual training (12.0B tokens)

Config: [`configs/phase2.yaml`](../configs/phase2.yaml). Init from Phase 1 final. WSD schedule: re-warm 6e-5 → 2.4e-4 over 300 steps, plateau to step 19,073 (**Stage 2A**, 10.0B), then linear decay to 6e-6 by step 22,888 (**Stage 2B**, 2.0B). Doc masking on. Window-level FIM on code pools (rate 0.5, 50/50 PSM/SPM).

### 4.1 Pools

| Group | Pool | Source (config / subset) | Unique tokens | License | Revision |
|---|---|---|---|---|---|
| English | `en_wiki` | `wikimedia/wikipedia` (20231101.en), first 1.5B tokens | 1.500B | CC BY-SA 3.0 / GFDL | `b04c8d1ceb2f` |
| | `en_cosmo` | `HuggingFaceTB/smollm-corpus` (cosmopedia-v2) | 1.520B | ODC-BY | `3ba9d6057741` |
| | `en_finemath` | `HuggingFaceTB/finemath` (finemath-4plus) | 1.000B | ODC-BY | `e92b25a61673` |
| | `en_fineweb` | the Phase 1 FineWeb-Edu pool (reused) | 10.000B | ODC-BY | `87f09149ef47` |
| Code | `code_python_edu` | `Avelina/python-edu-cleaned`, first 2,936,148 docs | 1.500B | ODC-BY | `3fedf55e02e7` |
| | `code_starcoder_python` | `bigcode/starcoderdata` (python), first 881,190 docs | 1.250B | BigCode terms (gated) | `9fc30b578ced` |
| | `code_starcoder_typescript` | starcoderdata (typescript), first 1,616,913 docs | 1.250B | same | same |
| | `code_starcoder_javascript` | starcoderdata (javascript), first 939,475 docs | 1.000B | same | same |
| | `code_starcoder_go` | starcoderdata (go), first 375,717 docs | 0.750B | same | same |
| | `code_starcoder_sql_shell_cfg` | starcoderdata: sql, shell, yaml, json, dockerfile, markdown (0.122B each) | 0.730B | same | same |
| Agentic (block) | `agentic_toolmind` | `Nanbeige/ToolMind` (APIGen-MT, BUTTON, ToolACE, When2Call, Glaive, tau, xLAM queries) | 0.203B (192,448 episodes) | Apache-2.0 | `8020ed1c03c3` |
| | `agentic_glaive_fc_v2` | `glaiveai/glaive-function-calling-v2` | 0.059B (112,431) | Apache-2.0 | `e7f4b6456019` |
| | `agentic_openthoughts` | `open-thoughts/OpenThoughts-114k` | 0.014B (8,575; 105k dropped as > 2048) | Apache-2.0 | `bd093c3994fd` |
| | `agentic_hermes` | `NousResearch/hermes-function-calling-v1` | 0.007B (9,717) | Apache-2.0 | `dae3e1d28cfb` |
| | `agentic_bespoke_stratos` | `bespokelabs/Bespoke-Stratos-17k` | 0.006B (3,450; 13k dropped as > 2048) | Apache-2.0 | `9e9adba94391` |
| Contrastive | `contrastive` (v1) | ours: `scripts/gen_contrastive.py`, one sentence frame. Published: [`TheFireHacker/bubblspace-contrastive-binding`](https://huggingface.co/datasets/TheFireHacker/bubblspace-contrastive-binding) config `v1` (2,089,904 docs) | 0.062B | CC BY 4.0 | — |
| | `contrastive_v2` | ours: `scripts/gen_contrastive_v2.py`, 15 frames, pronoun/former-latter, varied Q&A, distractors, possession chains. Held-out frames/pairs/names → probe v4. Published: same repo, config `v2` (3,635,229 docs) | 0.133B | CC BY 4.0 | — |

The agentic token counts exclude padding (blocks incl. padding: 0.376B in total).

### 4.2 Mix per stage (group shares)

| Group | Stage 2A (10.0B) | Stage 2B (2.0B) | Total drawn |
|---|---|---|---|
| English | 41.5% (4.15B) | 34% (0.68B) | ~4.83B |
| Code | 49% (4.90B) | 38% (0.76B) | ~5.66B |
| Agentic | 8% (0.80B) | 22% (0.44B) | ~1.24B |
| Contrastive | 1.5% (0.15B, v1 only) | 6% (0.12B, v2 only) | ~0.27B |

- **Within-group weights:** English weights wiki 1.5 / cosmo 1.5 / finemath 1.0 / fineweb 0.8. Code is weighted by unique tokens, with `code_python_edu` ×3 in Stage 2B.
- **Stage 2B changes** were applied by resuming from step 19,000:
  - FineWeb-Edu kept at its Stage 2A weight (~5.7% of 2B) instead of 0, after a −1.8-point drop on the English gate at steps 5k/7k
  - contrastive v1 → v2, because v1's template was memorized and copied into generations (its validation loss jumped 0.59 → 1.67 after the switch)

### 4.3 Contamination findings

- **BFCL:** 11 question ids found in the agentic data are excluded from scoring.
- **HumanEval/MBPP** vs the two Python pools (`benchmark/phase2/contamination_code.json`):
  - 36/164 HumanEval and 105/500 MBPP reference solutions share at least one 20-token window with training
  - passed problems are not more overlapping than failed ones (HumanEval 14% vs 23%, MBPP 24% vs 20%), so scores aren't inflated

## 5. Phase 2.5a: knowledge + relations + code (20.0B tokens, data prep in progress)

Plan: [plan-03 §2](plan-03-phase-2.5-knowledge-context.md). Init from Phase 2 final, context 2048, 38,147 steps.

### 5.1 Mix (20B)

| Category | Billion tokens | Share |
|---|---|---|
| Knowledge (English) | 11.2B | 56% |
| Code | 5.0B | 25% |
| Relations (English) | 1.8B | 9% |
| Agentic / tool | 1.0B | 5% |
| Math | 1.0B | 5% |

### 5.2 Pools

Paths are under `/lambda/nfs/h100-f1/data/phase25/` unless noted.

| Category | Pool | Source / build | Filters | Prepared | Trained | License | Revision | Scan | Status |
|---|---|---|---|---|---|---|---|---|---|
| Knowledge | `en_wiki_full` | `wikimedia/wikipedia` 20231101.en, all 6,407,814 articles | — | 5.058B | 4.4B | CC BY-SA 3.0 / GFDL | `b04c8d1ceb2f` | HF safe 41/41 | ✅ |
| | `fact_seeking` | `nvidia/Nemotron-Pretraining-Specialized-v1.2` / Fact-Seeking, `part_000000`–`000007` (MCQ about encyclopedic facts, written by Qwen3-30B-A3B) | scan-safe files only | 5.424B | 3.6B | CC BY 4.0 | `807afc1fa65c` | HF safe (8 files) | ✅ |
| | `fwe_edu4` | `HuggingFaceFW/fineweb-edu` sample-100BT | `int_score ≥ 4` (14% of docs kept) | 2.500B | 1.6B | ODC-BY | `87f09149ef47` | HF safe | ✅ |
| | `synth_en` | `PleIAs/SYNTH` (Wikipedia vital articles amplified into queries + grounded answers) | `language=en`; text = query + answer (reasoning traces dropped) | 2.5B target | 1.6B | CC BY 4.0 | `0d6813a29666` | ClamAV clean | 🔄 tokenizing |
| Code | `code_python_edu_fresh` | `Avelina/python-edu-cleaned` | skip first 2,936,148 docs (used in Phase 2), so never-seen code | 2.422B (dataset exhausted) | 2.0B | ODC-BY | `3fedf55e02e7` | HF safe 29/29 | ✅ |
| | `code_starcoder_python_fresh` | `bigcode/starcoderdata` python (local scanned parquet) | skip first 881,190 docs | 1.500B | 1.0B | BigCode terms | `9fc30b578ced` | ClamAV clean | ✅ |
| | `code_starcoder_js_fresh` / `_ts_fresh` | starcoderdata javascript / typescript | skip 939,475 / 1,616,913 docs | 0.250B + 0.250B | 0.3B | BigCode terms | same | ClamAV clean | ✅ |
| | Phase 2 code pools (replay) | §4.1 | — | 6.48B | 1.7B | — | — | — | ✅ on disk |
| Relations | `contrastive_v3` | ours: `scripts/gen_contrastive_v3.py`: v2 shapes (39%) + new structures (reported speech, whereas/while, negation pairs, both…but only, direct comparatives, doer/receiver roles, fit counterfactuals) + new families (part–whole 64 pairs, cause–effect 45 pairs, temporal). **Held out entirely:** passive voice, relative clauses, "respectively" → probe v5 (120 cases, private until used). Published: `bubblspace-contrastive-binding` config `v3` (with `family` / `structure` columns) | 63,603 v2-style docs skipped (surface overlap with held-out structures) | 0.348B (10,094,548 docs) | 0.6B | CC BY 4.0 | — | — | ✅ |
| | `relational_llm` (raw: `raw/relational_llm`) | ours: `scripts/gen_relational_llm.py`, Qwen3-30B-A3B-Instruct-2507 (`0d7cf23991f4`), grounded on Wikipedia passages, 5 formats, fact-checked | probe-overlap, copy (15-word), fact-check verdict | ~50–70M (time-boxed) | 0.15B | CC BY-SA 4.0 | — | — | 🔄 generating; **user EDA before 2.5a**; published as `TheFireHacker/bubblspace-relational-passages-Qwen3-30B` |
| | `kg_text` | ConceptNet (is-a, part-of, used-for, causes, capable-of) + Wikidata triples, each in several phrasings | — | ~0.5B | 0.6B | CC BY-SA / CC0 | — | to check | ⏳ to build |
| | `nli_commonsense` | e-SNLI, MNLI, CommonsenseQA (train), ATOMIC-2020, rendered as text (ANLI excluded: non-commercial) | — | ~0.25B | 0.45B | mostly CC BY / CC BY-SA | — | to check | ⏳ to build |
| Agentic | Phase 2 agentic pools (replay) | §4.1 | — | 0.38B in blocks (0.29B unpadded) | 1.0B (~2.7 passes over the blocks) | Apache-2.0 | — | — | ✅ on disk |
| Math | `en_finemath` (replay) | §4.1 | — | 1.0B | 1.0B | ODC-BY | — | — | ✅ on disk |

**Considered and not used:**
- `nvidia/Nemotron-CC-v2` / `diverse_qa`: gated, access request pending. SYNTH fills the slot.
- `wiki_rewrite`: listed by Marin but not in NVIDIA's public Specialized-v1 repo any more.
- `nvidia/Nemotron-Pretraining-Code-v3`: metadata only (repo/path/commit); the code itself would need fetching from GitHub.
- Indian-language data: deferred to a separate stage with a tokenizer extension (current tokenizer: Hindi 5.1×, Marathi 5.5×, Punjabi 8.0×, Bengali 7.7× English tokens for the same sentence).

## 6. Phase 2.5b: context extension to 16,384 (planned, ~1.5B tokens)

Plan: [plan-03 §3](plan-03-phase-2.5-knowledge-context.md). Init from 2.5a final. RoPE base 10,000 → 1,000,000, seq 16,384, micro-batch 2.

| Share | Content | Source pools | Tokens in docs > 8k / > 16k |
|---|---|---|---|
| ~35% | **Long documents** (8k–16k+, some 4k–8k) | FineWeb-Edu (Phase 1) | 1.36B / 0.53B |
| | | StarCoder Python / JS / Go / TS (Phase 2) | 1.28B / 0.89B |
| | | FineMath | 0.22B / 0.11B |
| | | Wikipedia (Phase 2 pool; full Wikipedia adds more) | 0.18B / 0.06B |
| | | Python-edu | 0.07B / 0.03B |
| | | long agent traces (Nemotron-Terminal, AgentTrove, penfever): **if they pass the length and security scans** | — |
| ~65% | **Short replay** from the 2.5a mix | §5 | — |

**To build:**
- a long-document extraction script (docs > 4k/8k out of the existing shards into `phase25b/long_*` pools)
- re-prep or streaming mode for the block-format agentic pools at 16,384

Measured with `scripts/measure_doc_lengths.py` (`benchmark/phase2/doc_lengths.json`).

## 7. Evaluation data (not trained on)

| Eval | Source | Notes |
|---|---|---|
| English 6 | lm-eval: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, LAMBADA | log-likelihood |
| Closed-book facts | lm-eval TriviaQA (5-shot, first 2,000), NQ-open (5-shot, 3,610); `benchmark/data/facts-closedbook/facts-simple-v1-20261123.json` (95 questions, ours) | greedy generation |
| Binding probes | `benchmark/data/probefinding-winograndestyle/`: v2 (100), v3 (70, new structures), v4 (112, held out from contrastive v2), v5 (120, held-out structures of contrastive v3: passive, relative clause, respectively; private until used). v2–v4 published on HF with a canary GUID | log-likelihood, minimal pairs |
| Code | HumanEval (164), MBPP full test (500), Docker sandbox; CRUXEval-O (800) | greedy |
| Tool use | BFCL simple / multiple / parallel (11 contaminated ids excluded) | AST match |
| Long context (2.5b) | needle/passkey + multi-hop retrieval at 2k–16k, loss by position | to build |

## 8. Changelog

| Date | Change |
|---|---|
| 23 Nov 2026 | Contrastive v3 generated (348M tokens) and published as config `v3`; probe v5 built (private) |
| 23 Nov 2026 | Contrastive v1/v2 + binding probes v2/v3/v4 (canary GUID) published on HF (`bubblspace-contrastive-binding`, CC BY 4.0) |
| 23 Nov 2026 | File created: Phases 1–2 documented from pool metadata; Phase 2.5a pools and status; 2.5b plan; revisions recorded |
