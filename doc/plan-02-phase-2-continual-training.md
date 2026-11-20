# Phase 2 Execution Plan: Code + Agentic Continual Pretraining (with English Replay)

**Status:** Finalized design, data prep in progress on the secondary GPU instance.
**Corrections history:** this plan supersedes the first draft after an audit that found
real bugs and dataset-access problems — see [`issue-01-phase2.md`](issue-01-phase2.md)
for the full audit (confirmed issues, independent verification notes, and what was
pushed back on). Design rationale (why continue from checkpoint, why a shifting
mixture, why anneal pairs with LR decay) remains in
[`continual-training.md`](continual-training.md).

**Security update (post-finalization):** `OpenCoder-LLM/opc-annealing-corpus` was
**dropped entirely** after checking Hugging Face's actual ClamAV security-scan API
(not just the dataset page, which is client-rendered and misses this on a plain
fetch) — **59 of 85 sampled files (69%)** across exactly the two configs this plan
had selected (`algorithmic_corpus`, `synthetic_code_snippet`) came back
`status: unsafe`. That's pervasive, not an isolated false positive, so the whole
dataset is excluded rather than cherry-picked around. Its 0.98B-token allocation
was redistributed into `starcoderdata`'s existing language pools (section 3).
Separately, `bigcode/starcoderdata` itself has 4 ClamAV-flagged files repo-wide —
verified via the same API to all sit in `git-commits-cleaned/` (3 files) and
`github-issues-filtered-structured/` (1 file), **neither of which are in this
plan's 5 selected languages** — no change needed there, but noted so this isn't
re-litigated if the language set is ever expanded.

## 1. Phase 1 status

| | |
|---|---|
| Current step | 12,090 / 19,073 (~63%) |
| Measured steady-state throughput | ~87,500 tok/s |
| **Estimated time until Phase 1 completes** | **~11.6 hours** from current progress |

Phase 2 continued-training cannot start on the training host until `step_19073.pt`
exists. The training host must not be interrupted or shared with any other job while
Phase 1 is running. Data prep (section 5) has no such constraint and runs in parallel
on a separate host.

## 2. Finalized token budget: 12B tokens

12B tokens (the lower end of the "sweet spot" tier — large enough for each of the
64 experts to see a meaningful number of code/agentic tokens under top-8 routing,
without committing to a multi-day run). Scales linearly to 15B if more H100 time is
approved later — same ratios, same datasets, just a longer Stage 2A.

**GPU time estimate:** 12B tokens ÷ ~87,000 tok/s measured throughput ≈ **~38 hours**.

**Mixture — 26% English replay / 63.2% Code / 10.8% Agentic** (revised twice: first
from the original draft's 71.5/21.9/6.6, which would have left the finished model at
>90% English — see Issue 1 in the audit; then again after real data-prep yields came
in far short on the agentic pool — see "Agentic pool revision" below):

| Pool | Overall share | Tokens | Unique tokens on disk | Oversampling |
|---|---|---|---|---|
| English replay | 26.0% | 3.12B | 3.12B (1.60B reused Phase 1 shards + 1.52B `cosmopedia-v2`) | ~1.0x |
| Code | 63.2% | 7.58B | 6.48B | ~1.17x (trivial — `starcoderdata` has far more headroom if ever needed) |
| Agentic (tool-use + reasoning) | 10.8% | 1.30B | ~375M (see revision below) | ~3.47x |

**Agentic pool revision:** the original 20% (2.4B) target assumed ~1.1B unique
agentic tokens after the ≤2048 filter. Real data-prep yield came in at only ~297M
across the first three sources (`hermes` 8.9M, `openthoughts` 17.3M, `toolmind`
270.4M — `OpenThoughts-114k` in particular kept only 8,575 of 114,000 rows, 92.5%
dropped, since full CoT traces average ~8k tokens). Two more sources were added
specifically to close the gap: `glaiveai/glaive-function-calling-v2` (112,431
episodes kept, only 0.47% dropped, **58.9M tokens** — mostly short single/few-turn
function calls, a strong yield) and `bespokelabs/Bespoke-Stratos-17k` (3,450 kept,
**5.5M tokens** — same `<|begin_of_thought|>`-tagged lineage as OpenThoughts but
more concise, a smaller but free addition). New total: **~375M unique tokens**.
Rather than force ~6.4x repetition to hit the original 2.4B target, the agentic
target was lowered to 1.3B (~3.47x — a reasonable repetition factor, not an
overfitting risk) and the freed 1.1B moved into Code, which already has far more
headroom than it needs.

### Two-stage WSD schedule

| Sub-phase | Steps (Phase 2's own counter) | Tokens | LR | English | Code | Agentic |
|---|---|---|---|---|---|---|
| **2A — Plateau** | `0 → 19,073` | 10.0B | `6e-5 → 2.4e-4` linear warmup over first 300 steps, then stable `2.4e-4` | 28% (2.80B) | 64% (6.40B) | 8% (0.80B) |
| **2B — Anneal** | `19,073 → 22,888` (3,815 steps) | 2.0B | Cosine/linear decay `2.4e-4 → 6e-6` | 16% (0.32B) | 59% (1.18B) | 25% (0.50B) |
| **Total** | **22,888 steps** | **12.0B** | WSD | **26.0% (3.12B)** | **63.2% (7.58B)** | **10.8% (1.30B)** |

During Stage 2B, the Code pool's internal weighting shifts toward the curated
`python-edu-cleaned` source and away from raw `starcoderdata`, so the lowest-LR
updates concentrate on the highest-quality code — same principle as concentrating
agentic data in the anneal window. (`opc-annealing-corpus` was originally slated for
this curated-anneal role but was dropped for pervasive ClamAV flags — see the
security update above — so `python-edu-cleaned` carries that role alone.)

Peak LR `2.4e-4` = 0.4x Phase 1's peak (`6e-4`) — large enough for Muon to
meaningfully re-specialize experts and for AdamW to pull the 14 suppressed special
tokens into real use (see Issue 7 in the audit — their embedding norms measured
~1.10 on `step_9000.pt` vs. ~1.28 for normal vocabulary, confirming real but
non-catastrophic suppression), without being so large it risks destabilizing what
Phase 1 already learned. **These exact numbers (300-step warmup, `2.4e-4` peak, the
19,073/3,815 split) are informed starting points, not settled — the Stage 2.1 smoke
gate in section 6 is what actually validates them before the full run.**

## 3. Finalized dataset manifest

All datasets below were independently verified this session via the live HF API /
dataset-viewer — not taken on claim. `bigcode/the-stack-v2-dedup` and
`HuggingFaceTB/python-edu` are **dropped entirely**: both only ship Software Heritage
blob-ID pointers, not file content (confirmed live — `python-edu`'s real schema is
`blob_id, repo_name, path, length_bytes, score, int_score`, no text field).

### Pool A — English replay (3.12B tokens)

| Source | Tokens | Access |
|---|---|---|
| FineWeb-Edu | 1.60B | **Reuse Phase 1's existing shards** (this *is* the replay — resampling old-task data is the correct mechanism, not a shortcut) |
| `HuggingFaceTB/smollm-corpus`, config `cosmopedia-v2` | 1.52B | `text_field="text"` — verified live schema: `prompt, text, token_length, audience, format, seed_data` |

### Pool B — Code (6.48B tokens, narrowed to 5 languages — not 13)

| Source | Language / config | Tokens | Access |
|---|---|---|---|
| `Avelina/python-edu-cleaned` | Python (curated/educational) | 1.5B | `text_field="text"` — verified: same schema as `python-edu` plus a hydrated `text` column, 980 junk files already removed |
| `bigcode/starcoderdata` | `data_dir="python"` | 1.25B | `text_field="content"`, **gated — accept ToS + set `HF_TOKEN`** |
| `bigcode/starcoderdata` | `data_dir="typescript"` | 1.25B | same |
| `bigcode/starcoderdata` | `data_dir="javascript"` | 1.0B | same |
| `bigcode/starcoderdata` | `data_dir="go"` | 0.75B | same |
| `bigcode/starcoderdata` | `data_dir="sql"` + `shell` + `json`/`yaml`/`dockerfile`/`markdown` | 0.73B | same, combined bucket |

Language scope was narrowed from the original 13-language spread
(Python/JS/TS/JSX/TSX/HTML/CSS/Vue/Java/Go/PHP/C#/Ruby/SQL) to these 5, so each
language gets 4-6x more signal rather than ~85M tokens apiece — see Issue 1.

`OpenCoder-LLM/opc-annealing-corpus` (originally slated here for ~0.98B curated
algorithmic/synthetic tokens) was dropped for pervasive ClamAV flags (see security
update above) — its allocation was redistributed proportionally across the
`starcoderdata` language pools, which have effectively unlimited headroom (250B+
tokens total) and no comparable flag issue in the selected languages. All 4
flagged `starcoderdata` files were individually confirmed to sit outside these
5 languages via Hugging Face's authenticated security API (`git-commits-cleaned/`
×3, `github-issues-filtered-structured/` ×1).

### Pool C — Agentic tool-use + reasoning (~375M unique tokens, hard-filtered to ≤2048 tokens, 1.3B drawn via ~3.47x oversampling)

All five sources formatted by `scripts/prepare_agentic.py` into one wire format
(`<|im_start|>role\n...\n<|im_end|>`, tool schemas wrapped in `<tools>`, calls in
`<tool_call>`/`<tool_response>`), hard-filtered to ≤2048 tokens per episode (dropped,
never truncated), and packed into 2048-aligned blocks with `<|fim_pad|>`-padded
tails. Real measured yields, not pre-run estimates:

| Source | Kept episodes | Tokens | Drop rate | Access |
|---|---|---|---|---|
| `Nanbeige/ToolMind` | 192,448 | 270.4M | 48% (of 368,611) | **Direct repo, not the `tuandunghcmut/toolmind` mirror** (manually-gated, unofficial re-upload, unclear license provenance — wrong fit for a repo carrying an "everything is open source" banner). Streamed as raw `.jsonl` via `huggingface_hub` + `json.loads`, bypassing `datasets.load_dataset`'s PyArrow schema inference, which crashes on ToolMind's heterogeneous nested `tools` JSON |
| `glaiveai/glaive-function-calling-v2` | 112,431 | **58.9M** | 0.47% (of 112,960) | Added after the original 3 sources came in far short (see below). Mostly short single/few-turn function calls — by far the best keep-rate found. `{system, chat}` schema with a custom `USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:` flat-text format, parsed with a dedicated regex rather than ShareGPT/OpenAI-style turns |
| `NousResearch/hermes-function-calling-v1` | 9,717 | 8.9M | 16% (of 11,578, across all 5 configs) | ShareGPT-style; system messages already carry `<tools>...</tools>` XML natively |
| `bespokelabs/Bespoke-Stratos-17k` | 3,450 | 5.5M | 79% (of 16,710) | Added alongside `glaive_fc_v2` — same `<|begin_of_thought|>`-tagged lineage as OpenThoughts, more concise, but still mostly long-form reasoning |
| `open-thoughts/OpenThoughts-114k` | 8,575 | 17.3M | **92.5%** (of 114,000) | `<think>`/`</think>` reasoning content (`<|begin_of_thought|>`/`<|end_of_thought|>` remapped). Full traces average ~8,000 tokens (3.55GB / 114k rows), so the ≤2048 filter drops nearly all of them — this is the single biggest reason the original ~1.1B unique-token estimate turned out wrong |

**Combined unique agentic pool: ~375M tokens** (not the originally estimated
~1.1B). `ToolMind` and `OpenThoughts-114k` were in the original plan; `glaive_fc_v2`
and `Bespoke-Stratos-17k` were added specifically to close the gap once real yields
came in short — `Salesforce/xlam-function-calling-60k` was dropped from the plan
since `ToolMind` already incorporates a richer, reasoning-annotated version of the
same underlying xLAM queries (confirmed by comparing file sizes: ToolMind's
`xlam-function-calling-60k-query.jsonl` alone is 476MB vs. the standalone repo's
97MB). At the revised 1.3B allocation (section 2), that's ~3.47x oversampling —
a reasonable repetition factor, not an overfitting risk the way ~6-8x would have
been against the 2.4B target.

## 4. Confirmed bugs this plan fixes (full detail in `issue-01-phase2.md`)

1. **File-level FIM breaks across window boundaries.** `src/data/prepare.py` currently
   tokenizes a whole file, applies `apply_fim()` to the full sequence, then writes
   flat shards; `src/data/dataset.py` slices a uniformly random 2048-token window
   with no document/FIM-boundary awareness (`start = random.randint(0, max_start)`,
   confirmed by reading the code directly). For any file over ~2048 tokens, the three
   FIM sentinels can land in different windows entirely.
   **Fix:** store raw `<|endoftext|>`-delimited code in shards; apply FIM dynamically
   inside `dataset.py`, per sampled document span, respecting `eot_token = 0`
   boundaries, so every FIM sample is guaranteed prefix+suffix+middle in one window.
2. **`src/data/fim.py` is PSM-only** (confirmed by reading the file — no SPM branch
   exists). **Fix:** add SPM, 50/50 split with PSM (standard per Bavarian et al. /
   StarCoder / Qwen-Coder).
3. **Random mid-stream slicing breaks structured agentic data.** A random start index
   through `<tools>...<tool_call>...<tool_response>` detaches tool calls from their
   schema header. **Fix:** pack complete ≤2048-token agentic episodes into
   2048-aligned blocks (`start = k * 2048`), pad remainders with loss masked via
   `ignore_index = -100` (not currently supported in `scripts/train.py`'s
   unmasked `F.cross_entropy` call — needs adding).
4. **`prepare.py` has no `--data_dir` support**, which `starcoderdata` requires for
   language selection (confirmed: `load_dataset()` is currently only called with
   `name=args.dataset_config`). Note: `--text_field` already exists as a flag, so
   reading `content` instead of `text` needs no new code — only `--data_dir` is
   actually missing.
5. **Resuming past `step_19073` would exit immediately** — `scripts/train.py`'s loop
   is `for step in range(start_step, total_steps)`, so `--resume step_19073.pt` with
   Phase 1's own `total_steps=19073` does nothing. **Fix:** add a distinct
   `--init_from <ckpt_path>` path that loads `model` + both optimizer states from
   `step_19073.pt` but starts a fresh Phase 2 step counter and LR schedule.

## 5. Infra plan

- Data prep (download, tokenize, format) runs on the **secondary GPU instance already
  used for evaluation** — tokenization is CPU/disk-bound, no GPU needed, so this never
  touches or risks the running Phase 1 training host.
- Token shards land in a new `data/phase2/<pool_name>/` directory, separate from
  Phase 1's English-only shard set.
- Finished shards transfer to the training host's persistent filesystem via the same
  ephemeral-SSH-key + `rsync` pattern already used for checkpoint transfers this
  project (~85-90MB/s measured, multi-GB files in ~1-2 minutes).
- `bigcode/starcoderdata`'s gate (accept ToS on its HF page) and an `HF_TOKEN` on the
  data-prep host are one-time setup steps before streaming it.

## 6. Concrete work items

1. **`src/data/prepare.py`:** add `--data_dir`; stop applying file-level FIM (move
   that logic out); keep writing clean `<|endoftext|>`-delimited shards per pool;
   write a small held-out `val.bin` (~10M tokens) per pool, excluded from training
   shards.
2. **`src/data/fim.py`:** add SPM alongside PSM, 50/50 split via `fim_rate=0.5`.
3. **New `scripts/prepare_agentic.py`:** formats Pool C sources into one consistent
   wire format (`<|im_start|>system ... <tools>...</tools> ... <|im_end|>`, assistant
   turns with optional `<think>...</think>` then `<tool_call>...</tool_call>`, tool
   results as `<|im_start|>user ... <tool_response>...</tool_response><|im_end|>`),
   rejects any trajectory over 2048 tokens, and packs the rest into 2048-aligned
   blocks with `-100`-masked padding.
4. **`src/data/dataset.py`:** add a `MultiPoolTokenDataset` that samples from named
   pools (`en_web`, `en_cosmo`, `code_python_edu`, `code_starcoder_*`,
   `code_opc_anneal`, `agentic_packed`) according to a step-dependent ratio schedule
   (the Stage 2A/2B table above), and applies the dynamic window-level FIM from
   item 2 to code-pool samples. `agentic_packed` samples use block-aligned starts,
   not random ones.
5. **`scripts/train.py` / `src/config.py` / new `configs/phase2.yaml`:** add
   `--init_from` (distinct from `--resume`) to load Phase 1's weights + Muon/AdamW
   optimizer state with a fresh Phase 2 step range and the WSD schedule from
   section 2; add `ignore_index=-100` to the cross-entropy call for packed-block
   padding.
6. **Unit tests, run on CPU before any GPU time:**
   - FIM window invariants (PSM + SPM, `<|endoftext|>` boundary safety, all 3
     sentinels always co-resident in one 2048 window).
   - Agentic packing (every special token = 1 id, no sample over 2048, block-aligned
     starts, padded positions produce `y == -100`).
   - Multi-pool schedule (Stage 2A/2B weights each sum to 1.0; `--init_from` loads
     Phase 1 weights/optimizer state while starting Phase 2 at step 0).

## 7. Eval plan

- **Must-have:** per-pool held-out validation loss (`val/en`, `val/code`,
  `val/agentic`) every ~250 steps, plus a deterministic (non-executing) format check
  every ~2,000 steps — valid PSM/SPM infilling, valid JSON inside
  `<tool_call>...</tool_call>`, clean `</think>` closure within 2048 tokens.
- **Anti-forgetting gate:** continue the existing `hellaswag / piqa / arc_easy /
  arc_challenge / winogrande / lambada_openai` suite at the same cadence used in
  Phase 1 — these should hold within ~1-1.5 points of the `step_19073` baseline
  throughout Phase 2. A regression here is the signal the whole replay design exists
  to prevent.
- **Deferred, out of scope for this plan:** `HumanEval`/`MBPP` `pass@1` requires
  **executing model-generated code**, which needs a real isolated sandbox. That's a
  separate, security-scoped follow-up — not bundled into Phase 2's training infra.

## 8. Execution sequence

1. **While Phase 1 finishes (~11.6h window):** implement section 6's items locally,
   pass CPU unit tests; accept the `starcoderdata` gate and set `HF_TOKEN` on the
   data-prep host; run a short profiling pass (`--max_tokens 50_000_000` on one code
   subset) to get a real data-prep throughput number instead of trusting the earlier
   rough 2-6 hour estimate; download/tokenize all Phase 2 pools; run
   `scripts/sanity_check_data.py` against each pool; `rsync` shards to the training
   host.
2. **Smoke gate (100 steps on H100, from `step_19073.pt` via `--init_from`):** confirm
   initial `val/en` matches Phase 1's final loss (proving clean checkpoint load),
   `grad_norm` stays stable through the 300-step warmup to `2.4e-4`, `lb_loss` stays
   in a sane range as the router adapts to new token distributions, and
   `torch.compile` doesn't recompile excessively on the new data shapes. Do not
   proceed to the full run until this passes.
3. **Full run (22,888 steps, ~38h):** monitor the same signals as Phase 1 (loss,
   `grad_norm`, `tokens_per_expert` balance) plus the new per-pool val losses and
   format checks from section 7.

## 9. Scope reminder

Phase 2 is still base pretraining, not chat or instruction-following. Per-codebase
sequencing after Phase 2:

- **Phase 3 — SFT** (~3h on H100): same `train.py` + Muon/AdamW mechanics, with
  assistant-only loss masking (`y = -100` on user/system/`<tool_response>` tokens) so
  the model learns turn-taking and to actually stop at `<|im_end|>`.
- **Phase 4 — optional RL/GRPO** (~4-6h), only after Phase 3: multiple rollouts per
  prompt scored with deterministic, verifiable rewards (valid tool-call JSON against
  schema, passing unit tests for code, correct final answers for math) — not
  attempted before SFT, since a base/Phase-2-only model rarely produces clean
  stops or valid tool-call JSON on its own, so RL would get near-zero reward signal
  without SFT first.

At ~211M active parameters, the realistic goal remains learning the FIM/tool-call/
think *formats* and basic code/agentic patterns — not becoming a strong standalone
coding model.
