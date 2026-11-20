# MoE-0.9B-A0.2B

A from-scratch Mixture-of-Experts language model (~915M total / ~211M active params), trained on a single Lambda Cloud H100 80GB. Architecture is Qwen3-Next-inspired: GQA + QK-norm + RoPE + an elementwise attention output-gate, 64-expert dropless top-8 SwiGLU MoE with ZeroCenteredRMSNorm, trained with Muon (hidden matrices) + AdamW (embedding/router/norms).

**Training goal**: this model is intended for coding and AI-agentic use — hence the tokenizer's FIM tokens (fill-in-the-middle, for code) and tool-call/think special tokens (for agentic tool-use and reasoning traces). The current concrete data pipeline tokenizes FineWeb-Edu as the base text corpus; a code + agentic-trajectory data mixture is the natural next phase (see "Training objective" below) and the tokenizer/FIM plumbing is already in place for it.

- **Weights + runnable inference code**: [huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b)
- **License**: [Apache 2.0](LICENSE)

## Status

Pretraining (phase 1: English-only base checkpoint, FineWeb-Edu, 10B tokens, single H100, no instruction tuning yet) is in progress. See [`benchmark/README.md`](benchmark/README.md) for benchmark results tracked across checkpoints as training progresses, and [`doc/eval-notes.md`](doc/eval-notes.md) for a full technical walkthrough of how evaluation is wired up for this custom (non-`transformers`-native) architecture.

## Architecture summary

| Component | Spec |
|---|---|
| Layers | 16, `d_model=1024` |
| Attention | 16 heads / 4 KV heads (GQA), head_dim 64, QK-norm before RoPE, elementwise output gate, no bias |
| MoE | 64 experts, top-8, SwiGLU, expert_hidden 256, dropless |
| Norm | ZeroCenteredRMSNorm (`1+w`, `w` init 0, weight-decayed) |
| Vocab | 49,280 (bubblspace-tokenizer, padded), tied embedding/lm_head |
| Context | 2048 tokens |

| | Total | Active |
|---|---|---|
| Embedding | 50.5M | 50.5M |
| Attention (incl. output gate) | 58.7M | 58.7M |
| Routers | 1.05M | 1.05M |
| Experts | 805.3M | 100.7M |
| **Total** | **~915M** | **~211M** |

Optimizer: Muon (lr 6e-4, momentum 0.95, Nesterov, 5 Newton-Schulz steps, update_scale `0.2*sqrt(max(rows,cols))`, weight_decay 0.1) on every 2D/per-expert-stacked hidden matrix (attn q/k/v/o/gate, expert gate/up/down); AdamW (lr 6e-4, betas 0.9/0.95, weight_decay 0.1) on the tied embedding, routers, and all norm weights. Cosine LR schedule, 2000-step warmup, min_lr_ratio 0.1. Global batch 524288 tokens (256×2048), bf16, `torch.compile`.

Full config: [configs/base.yaml](configs/base.yaml). Cheap smoke-test config: [configs/debug.yaml](configs/debug.yaml).

## Tokenizer

[scripts/build_tokenizer.py](scripts/build_tokenizer.py) builds `bubblspace-tokenizer/` from `HuggingFaceTB/SmolLM2-360M`'s tokenizer (49,152 tokens) — **no new tokenizer is trained**, it only extends the vocabulary:

| Added | Tokens |
|---|---|
| Chat | `<\|im_start\|>`, `<\|im_end\|>` *(already present in the SmolLM2 base tokenizer — skipped)* |
| FIM | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>`, `<\|fim_pad\|>` |
| Tools | `<tools>`, `</tools>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>` |
| Reasoning | `<think>`, `</think>` |

Result: 49,152 + 12 new = 49,164 real tokens, padded to **49,280** (multiple of 128) for `model.vocab_size`. The 116 padding rows are never produced by the tokenizer and are sliced out of the loss entirely in `scripts/train.py` (not just left to cross-entropy's target-matching — they're excluded from the softmax too). Embeddings stay tied.

Document separator for pretraining: `<|endoftext|>` (the tokenizer's `eos_token`, id 0), appended after every document. Chat/tool/think tokens are **never** inserted into raw pretraining text; FIM tokens are inserted only into code samples when `--is_code --fim_rate > 0` is passed to `src/data/prepare.py` (see [src/data/fim.py](src/data/fim.py), standard PSM prefix/suffix/middle layout).

Verified locally (`scripts/build_tokenizer.py` + `scripts/sanity_check_data.py`): all 15 special tokens encode to a single id each, round-trip (`decode(encode(x)) == x`) holds on web/Python/math samples. Bytes-per-token vs. GPT-2, measured on real ~800KB held-out samples (real FineWeb-Edu docs, real Python stdlib source, real FineMath docs — not toy strings):

| Domain | bubblspace bytes/token | GPT-2 bytes/token | Result |
|---|---|---|---|
| Web (FineWeb-Edu) | 4.541 | 4.639 | ~parity (bubblspace ~2% more tokens) |
| Python (stdlib) | 3.752 | 2.191 | **bubblspace ~1.7x more efficient** (~58% as many tokens as GPT-2) |
| Math (FineMath) | 3.326 | 3.561 | bubblspace ~7% more tokens (slightly behind) |

Net: no efficiency reason to prefer GPT-2 for this model — bubblspace-tokenizer is clearly better on code, roughly tied on general English, marginally behind on math.

## Benchmarks

Standard base-LM benchmark suite (HellaSwag, PIQA, ARC-Easy/Challenge, WinoGrande, LAMBADA) via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), scored with a custom adapter since this isn't a `transformers`-native architecture — see [`src/eval/lm_eval_adapter.py`](src/eval/lm_eval_adapter.py) and [`doc/eval-notes.md`](doc/eval-notes.md) for how that's wired up, and [`benchmark/README.md`](benchmark/README.md) for full results tracked per checkpoint, including a comparison against GPT-2-124M, Pythia-160M/410M, and SmolLM2-360M at matched (or smaller) training-token budgets.

```bash
pip install -e ".[eval]"
python scripts/run_benchmarks.py \
    --checkpoint /path/to/step_N.pt \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --output benchmark/step_N.json
```

## Training objective / data mixture

The tokenizer (FIM + tool-call + think tokens) and `src/data/fim.py` are built for coding and AI-agentic training, but the actual data mixture for those is **not yet wired in** — the current `src/data/prepare.py` run targets FineWeb-Edu only. Before a code/agentic-focused run, this needs: (1) one or more code datasets passed through `prepare.py --is_code --fim_rate <r>` so FIM transforms apply, and (2) agentic trajectory data formatted with the `<tool_call>`/`<tool_response>`/`<think>` tokens at the point it's tokenized (not inserted into generic pretraining text). Dataset selection below is a recommendation, not yet downloaded or wired in.

Recommended mixture (shifting ratio — English-dominant early, code/agentic upweighted in a final anneal phase, rather than strict non-overlapping sequential stages, to avoid forgetting earlier stages):

| Stage | Dataset | Notes |
|---|---|---|
| English (base) | `HuggingFaceFW/fineweb-edu` (already wired in) | high-quality filtered web text |
| English (breadth) | `HuggingFaceTB/cosmopedia-v2` | synthetic textbook-style text; same corpus SmolLM2 itself trained alongside FineWeb-Edu |
| Code — Python | `HuggingFaceTB/python-edu` | educational-quality Python; pairs with `--is_code --fim_rate` |
| Code — frontend/JS | `bigcode/the-stack-v2-dedup`, language configs `javascript`, `typescript`, `jsx`/`tsx`, `html`, `css`, `vue` | single corpus, filter by `data_dir`/language; covers React/Vue-style frontend code |
| Code — backend (multi-language) | same `bigcode/the-stack-v2-dedup`, language configs `java`, `go`, `php`, `c-sharp`, `ruby`, `sql` | broadens beyond Python for server-side code; apply the same FIM transform |
| Agentic (tool-use) | `Salesforce/xlam-function-calling-60k` or `NousResearch/hermes-function-calling-v1` | formats map directly onto `<tools>`/`<tool_call>`/`<tool_response>` |
| Agentic (reasoning) | `open-thoughts/OpenThoughts-114k` (or similar CoT set) | gives `<think>`/`</think>` real content instead of unused wrapper tokens |

Caveats: (1) at ~211M active params this won't be a strong coding/agentic model, but enough to learn the formats and basic patterns; (2) The Stack v2 requires accepting its license/attribution terms on the dataset page before `datasets` can stream it; (3) the agentic datasets are small (tens of thousands of examples, not billions of tokens) — they need upsampling/multiple epochs during the anneal phase rather than being a token-matched stage on their own; (4) per-language token budget across frontend/backend isn't pinned down yet — a reasonable starting split is roughly even across the languages listed unless profiling shows a particular language underrepresented relative to your actual use case.

## Design notes

The architecture spec was a one-line gloss on a few details; these were resolved with a documented choice rather than left ambiguous — revisit if training behaves oddly:

1. **QK-norm/RoPE order**: norm applied before rotation (Qwen3-style).
2. **Global-batch load balancing**: the router's load-balance loss uses a *differentiable* per-micro-batch mean probability combined with a *detached* running expert-count fraction accumulated across the 16 gradient-accumulation micro-batches of one optimizer step (reset after each step). A literal "single aux-loss backward after all 16 micro-batches" isn't compatible with standard gradient accumulation.
3. **MoE fallback backend**: a dense-masked variant exists only as a correctness oracle in `tests/test_moe_backends.py` (computes every token through every expert — never used in training); the actual fallback is a gather + per-expert Python loop in the forward pass (`src/model/moe.py`). That loop is fine — only the Muon optimizer's Newton-Schulz step is required to avoid a per-expert loop, and it does (batched via `bmm`).
4. **`torch._grouped_mm`** is a private, experimental PyTorch API whose signature can shift between versions. `moe.backend: auto` probes it at startup (`src/model/moe.py:probe_grouped_mm`) and falls back automatically if it's unavailable or errors.
5. **`torch.compile`** + dropless MoE's data-dependent per-expert token counts can trigger excessive recompilation. Watch for this during the debug run; `training.compile: false` in the config is the documented escape hatch.
6. **Tokenizer vs. GPT-2**: measured for real (see "Tokenizer" above) — bubblspace-tokenizer is ~1.7x more token-efficient on Python code, ~parity on general English, ~7% behind on math. No change needed; resolved in favor of keeping the SmolLM2-based tokenizer.

## Reproducing training (Lambda Cloud / any CUDA host)

### 1. Storage

Attach a persistent filesystem to your GPU instance and mount it (Lambda: can only be attached to an instance in the same region as the filesystem). Checkpoints and tokenized data live there, never in this repo. Verify with a write test before starting a paid run:

```bash
ssh -i ~/.ssh/<your-key>.pem ubuntu@<your-instance-ip> "df -h /lambda/nfs/h100-f1 && touch /lambda/nfs/h100-f1/._t && rm /lambda/nfs/h100-f1/._t && echo OK"
```

### 2. Environment setup

```bash
ssh -i ~/.ssh/<your-key>.pem ubuntu@<your-instance-ip>
git clone <this-repo-url> moe0.9b-a0.2b && cd moe0.9b-a0.2b
bash scripts/setup_fire01.sh
wandb login                            # or put WANDB_API_KEY in .env (copy .env.example)
```

`setup_fire01.sh` confirms the GPU, sets up the venv, probes `torch._grouped_mm` availability, and verifies the storage mount.

### 3. Debug run (always do this before the full run)

```bash
source .venv/bin/activate
python scripts/build_tokenizer.py --out_dir bubblspace-tokenizer   # skip if already committed to the repo
python -m src.data.prepare --out_dir /lambda/nfs/h100-f1/data/debug --max_tokens 150_000_000
python scripts/sanity_check_data.py --data_dir /lambda/nfs/h100-f1/data/debug
python scripts/train.py configs/base.yaml --debug
```

Confirm before proceeding to the full run: loss decreases steadily (no NaN/explosion), `tokens_per_expert` trends toward uniform across experts, checkpoint save + `--resume` round-trips cleanly, and `torch.compile` isn't recompiling excessively.

### 4. Full run

```bash
python -m src.data.prepare --out_dir /lambda/nfs/h100-f1/data/full --max_tokens 10_000_000_000
python scripts/train.py configs/base.yaml
```

~19k optimizer steps at 524288 tokens/step. Checkpoints land in `/lambda/nfs/h100-f1/checkpoints/<run_name>/` every 1000 steps. Monitor the first few hundred steps closely before leaving it unattended.

### 5. Inference / benchmarking a checkpoint

```bash
python scripts/generate.py --checkpoint /path/to/step_N.pt --prompt "Your prompt here" --max_new_tokens 100
pip install -e ".[eval]"
python scripts/run_benchmarks.py --checkpoint /path/to/step_N.pt --output benchmark/step_N.json
```

## Local development / tests

The model, optimizer, and config code can be unit-tested on CPU without a GPU or any data:

```bash
pip install -e ".[dev]"
pytest tests/
```

`tests/test_moe_backends.py`'s `torch._grouped_mm` comparison is skipped automatically on machines without CUDA / that API.

## Project layout

```
bubblspace-tokenizer/  SmolLM2 base + chat/FIM/tool/reasoning special tokens (committed, built by scripts/build_tokenizer.py)
configs/        base.yaml (full spec) + debug.yaml (override-only, smaller shapes)
src/model/      transformer, attention (GQA+QK-norm+RoPE+output-gate), MoE (2 backends), norm, init
src/optim/      Muon (batched Newton-Schulz), AdamW/Muon param-group split, LR schedule
src/data/       tokenization (prepare.py), FIM transform (fim.py), memmap dataset reader (dataset.py)
src/eval/       lm-evaluation-harness adapter for this custom (non-transformers-native) architecture
scripts/        build_tokenizer.py, setup_fire01.sh, sanity_check_data.py, train.py,
                generate.py (inference), run_benchmarks.py, push_to_hf.py
benchmark/      per-checkpoint benchmark results (raw JSON + readable summary)
doc/            design/implementation notes, including a full eval-harness walkthrough
storyboard/     chronological narrative of how this project was built
tests/          CPU-feasible unit tests (param count, Muon math, MoE backend equivalence, tiny overfit wiring check)
```

Checkpoints and tokenized data live on the mounted persistent storage (`/lambda/nfs/h100-f1/...`), never in this repo. `bubblspace-tokenizer/` is the one data artifact that *is* committed (it's ~3.4MB and defines the model's vocabulary, not training data).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- Tokenizer base: [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- Pretraining corpus: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Muon optimizer: Keller Jordan's Newton-Schulz orthogonalization approach, with the Moonlight weight-decay variant
- Evaluation: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
