# Phase 2: Continual Pretraining (English + Code + Agentic Mixture)

Status: **Planned — not yet downloaded or wired in.**

Phase 1 (in progress, finishes at `step_19073` / 10B tokens) produces a pure English base checkpoint trained on `HuggingFaceFW/fineweb-edu` (`sample-10BT`). The `bubblspace-tokenizer/` already includes FIM (`<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>`), tool-use (`<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`), and reasoning (`<think>`, `</think>`) tokens, plus `src/data/fim.py`, but none of them receive training signal in Phase 1. Phase 2 is where those tokens get real content.

---

## Core Design

### 1. Continue from `step_19073.pt`, not from scratch

Phase 1's final checkpoint (`step_19073.pt`) contains the full model weights plus both optimizer states (Muon's momentum buffers and AdamW's first/second moments), the step counter, and RNG state. Starting Phase 2 from this checkpoint means the model begins already fluent in English rather than randomly initialized.

Re-initializing and training from scratch on the English + code + agentic mixture would re-spend the entire 10B-token English learning curve (≈32 hours on an H100) a second time — strictly wasteful compute when Phase 1 already paid for it.

### 2. Why a shifting mixture, not hard sequential stages

Hard sequential stages (all-English in Phase 1, then switching to 100% code/agentic in Phase 2) risk **catastrophic forgetting**:

- With zero gradient signal reinforcing English, weights drift to serve only the new distribution and general English capability (FineWeb-Edu perplexity, PIQA/HellaSwag/ARC) can collapse.
- **MoE-specific risk:** in a 64-expert MoE, a sudden domain switch hits the router as well as the weights — experts that specialized in English patterns stop receiving English tokens and get reassigned wholesale to code syntax.

The standard mitigation is **rehearsal**: keep a nonzero share of English in every batch throughout Phase 2 so every optimizer step still carries English gradient signal.

Phase 2 never does 100% English or 100% code — it is always a blend, with the mixture ratio shifting over Phase 2's step range:

- **Early in Phase 2 (English-dominant):** e.g., illustrative ≈85% English / 15% code + agentic — enough for the model and MoE router to start learning the new tokens and formats without shocking the weights.
- **Late in Phase 2 (code/agentic upweighted — final anneal):** shifts toward e.g., illustrative ≈40% English / 45% code / 15% agentic, while still keeping enough English in the mix to anchor the base.

*(Percentages above are illustrative, not finalized.)*

### 3. Why the heaviest code/agentic concentration goes at the end (anneal)

The data-ratio shift pairs with the learning-rate schedule:

- Late in training, as the learning rate decays toward its minimum (`min_lr_ratio` or zero), each optimizer step makes smaller, more surgical updates.
- Data seen during the LR cooldown has an outsized effect on final checkpoint behavior (the same principle behind Warmup-Stable-Decay / WSD training, which `configs/base.yaml` already supports via `schedule.decay: wsd`, and used in SmolLM2 and Qwen).
- Putting the highest concentration of specialized data (code and `<tool_call>`/`<think>` trajectories) in that low-LR window imprints those patterns without the large, potentially destructive gradient updates that cause forgetting earlier at high LR.

---

## Recommended Data Mixture

From `README.md`'s "Training objective / data mixture" section — English, code, and agentic data blended together rather than code-only:

| Stage / Pool | Dataset | Role / Formatting |
|---|---|---|
| English (base) | `HuggingFaceFW/fineweb-edu` | Already wired; Phase 1's dataset |
| English (breadth) | `HuggingFaceTB/cosmopedia-v2` | Synthetic textbook-style text; adds breadth beyond FineWeb-Edu |
| Code — Python | `HuggingFaceTB/python-edu` | Passed through `prepare.py --is_code --fim_rate <r>` (FIM-formatted) |
| Code — frontend | `bigcode/the-stack-v2-dedup` (`javascript`, `typescript`, `jsx`/`tsx`, `html`, `css`, `vue`) | FIM-formatted |
| Code — backend | `bigcode/the-stack-v2-dedup` (`java`, `go`, `php`, `c-sharp`, `ruby`, `sql`) | FIM-formatted |
| Agentic (tool-use) | `Salesforce/xlam-function-calling-60k` (or `NousResearch/hermes-function-calling-v1`) | Formatted with `<tools>`, `<tool_call>`, `<tool_response>` |
| Agentic (reasoning) | `open-thoughts/OpenThoughts-114k` (or similar CoT set) | Formatted with `<think>` / `</think>` |

**Data caveats:**

- The Stack v2 requires accepting its license/attribution terms on Hugging Face before `datasets` can stream it.
- The agentic datasets are small (tens of thousands of examples, not billions of tokens) — they require upsampling / multiple epochs during the anneal phase rather than being treated as a token-matched stage on their own.

---

## Concrete Phase 2 Work Items

1. **Wire up code data** — run `src/data/prepare.py --is_code --fim_rate <r>` against `HuggingFaceTB/python-edu` and `bigcode/the-stack-v2-dedup` language subsets so FIM transforms apply to real code.
2. **Wire up agentic data** — add a formatting/tokenization path for tool-use (`<tools>`/`<tool_call>`/`<tool_response>`) and CoT reasoning (`<think>`/`</think>`) datasets so those special tokens receive real training signal.
3. **Add multi-pool / scheduled mixture sampling** — update `src/data/dataset.py` (`ShardedTokenDataset`), which currently reads a single `data_dir` and samples shards uniformly by length (`shard_weights`), to draw from multiple shard pools (English, code, agentic) using a step-dependent ratio schedule parallel to the LR schedule.
4. **Add a continual-training / weight-init path in `scripts/train.py`** — Phase 1 runs `cosine` decay down to `min_lr_ratio: 0.1` by `step_19073`. Passing `--resume step_19073.pt` verbatim into the current training loop would exit immediately because `start_step == total_steps` (`for step in range(start_step, total_steps)`). Phase 2 needs a way to load `step_19073.pt` (weights, and optionally optimizer state) with a fresh Phase 2 step range and LR schedule.
5. **Decide token budget & LR schedule** — pin down total Phase 2 tokens, exact per-pool percentages over step ranges, and the LR schedule choice below.

---

## Open Decisions

- **Phase 2 LR schedule:**
  - *Option A (literal WSD anneal — likely better fit):* treat Phase 1's ending LR (`min_lr_ratio`, or a modest stable plateau) as the stable phase and add a short decay-to-zero tail in the final anneal window, since Phase 1 already completed a full warmup + cosine cycle.
  - *Option B (fresh cycle):* run a short new warmup + cosine/WSD decay sized to Phase 2's own token budget.
- **Optimizer state on Phase 2 start:** whether to carry over Muon/AdamW moments from `step_19073.pt` (smoother transition under Option A) or reset optimizer state when re-warming LR (common under Option B).
- **Token budget and per-language split:** total tokens for Phase 2 and the token share across Python, frontend, and backend languages.

---

## Scope & Expectations

- **Still base pretraining, not chat/SFT:** even after Phase 2 finishes, the model will not reliably chat or follow instructions. Supervised fine-tuning (SFT) is a separate later pass on top of the Phase 2 checkpoint.
- **Capacity caveat:** at ≈211M active parameters (≈915M total), the realistic target for Phase 2 is learning the FIM/tool-call/think formats and basic coding/agentic patterns — not a strong standalone coding model.
