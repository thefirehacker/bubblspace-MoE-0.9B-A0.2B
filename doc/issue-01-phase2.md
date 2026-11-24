# Issue 01 — Phase 2 Continual-Training Plan Audit & Corrections

Audit of the initial Phase 2 plan ([`doc/plan-02-phase-2-continual-training.md`](plan-02-phase-2-continual-training.md)) against the actual codebase (`src/data/prepare.py`, `src/data/fim.py`, `src/data/dataset.py`, `scripts/train.py`), live Hugging Face dataset schemas, and empirical measurements on the `step_9000.pt` checkpoint.

Includes independent verification notes and pushbacks on the proposed fixes so both sides' corrections are captured in one place.

---

## 1. Confirmed Bugs & Structural Issues in the Initial Plan

### Issue 1 — English-heavy Phase 2 ratio (71.5%) wastes GPU budget + 13-language dilution
- **Problem:** Phase 1 already trains on **10.0B tokens of 100% English** (`FineWeb-Edu`). Spending another 71.5% of Phase 2 on English (3.58B of a 5B budget) leaves the finished 15B-token model at **90.5% English / 7.3% code / 2.2% agentic** — burning ≈11 of 16 H100 hours on English replay. Continual-pretraining literature supports **≈20–30% replay** of the base distribution to prevent catastrophic forgetting.
- **Language dilution:** Splitting ≈1.1B code tokens across 13+ languages (`Python, JS, TS, JSX/TSX, HTML, CSS, Vue, Java, Go, PHP, C#, Ruby, SQL`) yields ≈85M tokens/language. At **211M active params**, focusing on **5–6 core languages** (Python, TypeScript, JavaScript, Go, SQL/Shell/Configs) gives 4–6x more signal per language.
- **MoE per-expert math:** `805M / 915M` params live in 64 experts with top-8 routing (`1/8` active per token). Under a 5B–6B Phase 2, each individual expert sees only ≈140M–400M code tokens. A **12B–15B** Phase 2 (`~38–48h` on H100 at measured ≈87k tok/s) gives **≈1.0B–1.25B code tokens per expert** (≈22B–25B cumulative).
- **Status:** Confirmed. Token budget tier (`6B` floor vs `12B–15B` sweet spot) is an explicit cost decision for the user.

### Issue 2 — `bigcode/the-stack-v2-dedup` and `python-edu` only ship Software Heritage `blob_id` pointers (no code text)
- **Problem:**
  - `bigcode/the-stack-v2-dedup` does **not** include file contents — only SWHIDs requiring a separate fetch pipeline against `s3://softwareheritage/content/{blob_id}`.
  - `HuggingFaceTB/python-edu` returns `401 / Repository Not Found`; the real subset is `HuggingFaceTB/smollm-corpus` (`name="python-edu"`), which **also only ships `blob_id` metadata** (`blob_id, repo_name, path, length_bytes, score, int_score`) and takes ≈6h of S3 `GetObject` calls + `NoSuchKey` handling.
- **Verified replacements (inline text, no S3 blob fetching):**
  - `Avelina/python-edu-cleaned` — same `python-edu` schema with a hydrated `text` column and 980 junk files removed.
  - `bigcode/starcoderdata` — inline `content` column (≈250B tokens, gated — accept Terms of Use + set `HF_TOKEN`).
  - `OpenCoder-LLM/opc-annealing-corpus` — 15.6M rows (≈26.8 GB) of algorithmic/synthetic code + Q&A with a direct `text` column.
- **Status:** Confirmed via live HF dataset-viewer API.

### Issue 3 — `src/data/prepare.py` is missing `--data_dir` support for `bigcode/starcoderdata`
- **Problem:** `bigcode/starcoderdata` selects languages via `data_dir="python"` (not `name=` / `--dataset_config`), and mixing its 4 non-code folders (`github-issues-*`, `jupyter-*`, `git-commits-*`) crashes schema loading. `prepare.py` currently passes `name=args.dataset_config` to `load_dataset()` and has no `--data_dir` option.
- **Clarification / correction:** `prepare.py` **already has `--text_field`** (default `"text"`), so reading `starcoderdata`'s `content` column is just `--text_field content` — only `--data_dir` needs to be added.
- **Status:** Confirmed (`--data_dir` missing; `--text_field` already exists).

### Issue 4 — File-level FIM (`src/data/prepare.py` + `src/data/fim.py`) breaks on files > 2048 tokens, and `fim.py` is PSM-only
- **Problem:**
  - `src/data/prepare.py` (lines 80–94) tokenizes an entire file, applies `apply_fim()` to the full-length sequence, and writes flat shards. `src/data/dataset.py` (line 40) then slices a uniformly random `2048`-token window (`start = random.randint(0, max_start)`) with no awareness of document or FIM boundaries.
  - For any code file longer than ≈2048 tokens, `<|fim_prefix|>`, `<|fim_suffix|>`, and `<|fim_middle|>` land in completely different training windows, feeding orphaned FIM tokens with no matching prefix/suffix in context.
  - `src/data/fim.py` only implements **PSM** (no **SPM** branch; standard practice is 50/50 PSM/SPM).
- **Fix:** Store raw `<|endoftext|>`-delimited code in shards and apply **50/50 PSM + SPM FIM on the fly** inside `dataset.py` per sampled document span within the `2048`-token window (respecting `eot_token = 0` boundaries).
- **Status:** Confirmed directly in `prepare.py`, `fim.py`, and `dataset.py`.

### Issue 5 — Context-length mismatch on `OpenThoughts-114k` (≈8k tokens/trace vs `max_seq_len = 2048`) & random slicing of structured tool traces
- **Problem:**
  - `open-thoughts/OpenThoughts-114k` is ≈3.55 GB across 114k rows (≈31 KB/row ≈ 8,000–9,000 tokens per trace). Random 2048-token slicing means the model rarely sees `<think>` and `</think>` in the same window, training a 211M-active model to open `<think>` and never close it.
  - Random mid-stream slicing on `<tools>...<tool_call>...<tool_response>` breaks JSON schemas and detaches tool calls from their `<tools>` header.
- **Fix:** Hard-filter reasoning and tool-use trajectories (or split multi-turn trajectories at valid assistant turn boundaries) to **`<= 2048` tokens**, and pack complete episodes into **2048-aligned blocks** with padding targets masked via `ignore_index = -100`.
- **Status:** Confirmed.

### Issue 6 — `xlam-function-calling-60k` is too small (≈93 MB / ≈24–29M tokens), and `Nanbeige/ToolMind` has PyArrow schema gotchas
- **Problem:** Relying primarily on `xlam-60k` for a 0.33B–1.2B agentic allocation forces 5–10x+ repetition. `Nanbeige/ToolMind` (≈3.96 GB, 360k+ multi-turn reasoning-enhanced tool trajectories, Apache 2.0) solves the scale problem, but raw `datasets.load_dataset("Nanbeige/ToolMind")` can fail on PyArrow schema inference across heterogeneous nested `tools` JSON.
- **Correction on mirror:** Do **not** use `tuandunghcmut/toolmind` — it is manually gated (not open / not auto-approve) and is an unofficial personal re-upload with unclear provenance.
- **Fix:** Stream `Nanbeige/ToolMind` raw `.jsonl` files directly via `huggingface_hub` + Python `json.loads` (bypassing PyArrow schema inference) alongside `xlam-60k` and `NousResearch/hermes-function-calling-v1`.
- **Status:** Confirmed (with mirror rejected in favor of direct `Nanbeige/ToolMind` JSONL streaming).

### Issue 7 — LR schedule "Option A" (`6e-5 -> 0` with no re-warmup) + empirically confirmed special-token embedding suppression
- **Problem:** Phase 1 ends at `min_lr_ratio = 0.1` (`lr = 6e-5`). Throughout all 19,073 steps of Phase 1, the 14 unused special tokens (`<|im_start|>`, `<|im_end|>`, 4 FIM, 6 tool, 2 think) sat inside `real_vocab_size = 49,164` (`scripts/train.py` line 140) with tied embeddings — receiving negative softmax pressure every step plus AdamW `weight_decay = 0.1` with zero positive target reinforcement.
- **Empirical measurement on `step_9000.pt`:**
  - Regular vocabulary (actively trained): embedding norm **≈ 1.28 mean**
  - Special tokens (`<think>`, `<tool_call>`, FIM, etc.): embedding norm **≈ 1.09–1.11**
  - Padded tail (`49164..49279`, excluded from softmax): embedding norm **≈ 0.42**
  - This confirms measurable suppression (below normal vocab, above pure-decay padding).
- **Fix:** Load `step_19073.pt` weights **and** Muon/AdamW optimizer states via a dedicated `--init_from` path (separate from crash-recovery `--resume`, which would exit immediately since `start_step == total_steps`), apply a short **WSD re-warmup** (e.g., `6e-5 -> 2.4e-4` over ≈300 steps), hold a stable plateau for bulk code + English replay (≈80–85%), then anneal `2.4e-4 -> 6e-6` over the final ≈15–20% while upweighting curated code + agentic data.
- **Caution:** The exact numbers (`300` steps, `2.4e-4`, `80/20`) are informed starting points, not settled law — the **100-step smoke gate on H100** (checking initial English val loss, `grad_norm`, and `lb_loss`) is load-bearing.
- **Status:** Confirmed + empirically verified on `step_9000.pt`.

### Issue 8 — Missing code/format eval gates (and scope boundary on code-execution evals)
- **Problem:** Running only `hellaswag / piqa / arc / winogrande / lambada` checks English anti-forgetting, but gives zero visibility into whether code, FIM, `<tool_call>`, or `<think>` are actually improving.
- **Fix (must-have):** Add lightweight per-pool held-out validation loss (`val/en`, `val/code`, `val/fim`, `val/agentic`) every ≈250 steps plus a deterministic non-executing format check (valid PSM/SPM infilling, valid JSON inside `<tool_call>...</tool_call>`, clean `</think>` closure within 2048 tokens).
- **Scope caution (defer):** `HumanEval` / `MBPP` `pass@1` requires **executing model-generated code**, which needs a real isolated sandbox. Treat sandboxed code-execution benchmarks as a separate, security-scoped follow-up rather than bundling them into Phase 2 training infra.
- **Status:** Confirmed, with execution-eval scope separated.

---

## 2. Scope Triage (Must-Have vs. Optional)

| Priority | Item | Why |
|---|---|---|
| **Must-fix (blocker)** | Direct-text datasets (`Avelina/python-edu-cleaned`, `bigcode/starcoderdata`, `opc-annealing-corpus`, direct `Nanbeige/ToolMind` JSONL) + `--data_dir` in `prepare.py` | Prevents 404s, SWH S3 detour, and PyArrow crashes |
| **Must-fix (correctness)** | Window-level 50/50 PSM+SPM FIM in `fim.py` / `dataset.py` | Fixes real bug where FIM sentinels split across 2048 windows |
| **Must-fix (correctness)** | `<= 2048` filter + 2048-aligned packing + `-100` pad masking for agentic/reasoning | Prevents broken `<tools>` JSON and unclosed `<think>` traces |
| **Must-fix (correctness)** | `MultiPoolTokenDataset` + `--init_from` WSD continual schedule in `train.py` | Required to continue past `step_19073` with replay + anneal |
| **Must-have (verification)** | Per-pool held-out val loss + 100-step H100 smoke gate + unit tests for FIM/packing/schedule | Catches silent data/LR regressions before burning 38–48h of H100 time |
| **Defer / separate** | Sandboxed `HumanEval` / `MBPP` execution eval | Requires proper code-execution sandboxing; not needed to run Phase 2 |
| **Phase 3 / Phase 4** | Assistant-masked SFT (Phase 3, ≈3h) → optional verifiable-reward RL/GRPO (Phase 4) | Separate post-training stages after Phase 2 base checkpoint finishes |

---

## 3. Remaining Open Decision Before Rewriting `plan-02`

- **Phase 2 token budget tier:**
  - **6B floor** (≈19h on H100)
  - **12B–15B sweet spot** (≈38–48h on H100; e.g. `10B–12.5B` plateau + `2B–2.5B` anneal, ≈1.0B–1.25B code tokens per expert)
  - **20B–25B** (≈64–80h on H100)
