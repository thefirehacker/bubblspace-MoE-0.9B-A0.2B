# MoE-0.9B-A0.2B

> **📌 Current checkpoint on Hugging Face (`main`): Phase 2, step 9,000 of 22,888.** It's mid-run (Stage 2A, peak learning rate, not annealed yet).
> The Phase 1 final checkpoint (`step_19073`, English only) stays available on the HF branch [`phase1-final`](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b/tree/phase1-final).

A from-scratch Mixture-of-Experts language model (~915M total / ~211M active params), trained on a single Lambda Cloud H100 80GB. Architecture is Qwen3-Next-inspired: GQA + QK-norm + RoPE + an elementwise attention output-gate, 64-expert dropless top-8 SwiGLU MoE with ZeroCenteredRMSNorm, trained with Muon (hidden matrices) + AdamW (embedding/router/norms).

**Training goal**: this model is intended for coding and AI-agentic use — hence the tokenizer's FIM tokens (fill-in-the-middle, for code) and tool-call/think special tokens (for agentic tool-use and reasoning traces). Phase 1 built an English base on FineWeb-Edu. Phase 2 (running now) continues it on English + code + agentic + contrastive-binding data.

- **Weights + runnable inference code**: [huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b) (`main` = Phase 2 step 9,000; `phase1-final` branch = Phase 1 final)
- **Live training graphs/logs (W&B, public)**: Phase 1 [goa8jk6e](https://forge.coreweave.com/wandb/firehacker/moe0-9b-a0-2b/runs/goa8jk6e?nw=nwuserfirehacker), Phase 2 [lec9ww5h](https://forge.coreweave.com/wandb/firehacker/moe0-9b-a0-2b/runs/lec9ww5h) (same project; Phase 2's `global_step` continues from 19,073)
- **License**: [Apache 2.0](LICENSE)

## Status

| Phase | What | State |
|---|---|---|
| 1 | English base, FineWeb-Edu `sample-10BT`, 10B tokens, 19,073 steps | ✅ done (`step_19073`, ~32.7h on one H100) |
| 2 | Continual training: English + code + agentic + contrastive, 12B tokens, 22,888 steps | 🟡 running, step ~10,000 (Stage 2A plateau; Stage 2B anneal from step 19,073) |
| 3 | SFT (assistant-only loss masking) | planned |

**Phase 2 so far** (full table in [`benchmark/README.md`](benchmark/README.md)):

| | Phase 1 final | P2 step 5,000 | P2 step 9,000 |
|---|---|---|---|
| CRUXEval-O pass@1 | 0.0% | 8.5% | next at step 19,000 |
| BFCL simple / multiple / parallel (decontaminated) | 0 / 0 / 0% | 9 / 6 / 0% | next at step 19,000 |
| Held-out agentic tool-call format (valid, correct name) | 0% | 30% | 30% (peak 48% at step 7,000) |
| **HumanEval pass@1** (164, sandboxed execution) | 0.0% | — | **9.1%** |
| **MBPP pass@1** (500, sandboxed execution) | not run (no code training) | — | **8.4%** |
| Reasoning: closes `</think>` within 512 tokens | 0% | 0% | 12% |
| English 6-task average | 47.18 | 45.40 | **46.19** (−1.0, recovering) |
| Binding probe v2 / v3 | 24/100, 4/70 | 25/100, 5/70 | 24/100, 3/70 |
| Binding probe v4 (held-out phrasings + entities; chance ≈ 28/112) | 13/112 | — | 11/112 |

**Mid-run changes (applied at step 19,000 by resuming from the checkpoint):**
- Stage 2B keeps FineWeb-Edu at ~5.7% instead of 0%.
- The contrastive binding data switches to v2 ([`scripts/gen_contrastive_v2.py`](scripts/gen_contrastive_v2.py)): 15 sentence frames instead of 1, pronoun resolution, and varied question formats. The v1 corpus was being memorized as templates.

See [`doc/eval-notes.md`](doc/eval-notes.md) for how every eval is wired up for this custom (non-`transformers`-native) architecture. Storyboards in [`doc/storyboard/`](doc/storyboard/) give the chronological build narrative.

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

**Phase 1:** 100% FineWeb-Edu `sample-10BT` (10B tokens, single pass).

**Phase 2** ([`configs/phase2.yaml`](configs/phase2.yaml), design in [`doc/plan-02-phase-2-continual-training.md`](doc/plan-02-phase-2-continual-training.md) and [`doc/storyboard/04-Phase2-Trainig.md`](doc/storyboard/04-Phase2-Trainig.md)) runs 16 pools in four groups, sampled per row by stage-dependent group weights:

| Group | Pools (sources) | Stage 2A (steps 0–19,073, 10B tok) | Stage 2B anneal (19,073–22,888, 2B tok) |
|---|---|---|---|
| English + knowledge + STEM | Wikipedia `20231101.en`, Cosmopedia-v2, FineMath-4plus, FineWeb-Edu replay | 41.5% | 34% (FineWeb-Edu ~5.7% of all 2B tokens) |
| Code (window-level FIM, 50/50 PSM/SPM) | `Avelina/python-edu-cleaned`; `bigcode/starcoderdata` Python, TypeScript, JavaScript, Go, SQL/shell/config | 49% | 38% (python-edu ×3) |
| Agentic (block-aligned episodes ≤2048 tokens, pad loss masked) | ToolMind, glaive-function-calling-v2, OpenThoughts-114k, hermes-function-calling-v1, Bespoke-Stratos-17k | 8% | 22% |
| Contrastive binding (synthetic minimal pairs) | 2A: v1 `scripts/gen_contrastive.py`, 61.9M unique tokens; 2B: v2 `scripts/gen_contrastive_v2.py`, 133.5M unique tokens | 1.5% | 6% |

Learning rate: re-warm 6e-5 → 2.4e-4 over 300 steps, hold, then linear decay to 6e-6 over the final 3,815 steps (the anneal). Attention uses a causal + same-document mask (`flex_attention`) so packed documents can't attend to each other. Every dataset was checked with Hugging Face's malware scan before use. `OpenCoder-LLM/opc-annealing-corpus` was dropped after 69% of sampled files came back flagged.

Caveat: at ~211M active params the realistic Phase 2 goal is learning the FIM / tool-call / think formats and basic code and agentic patterns, not a strong standalone coding model.

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
doc/storyboard/ chronological narrative of how this project was built
tests/          CPU-feasible unit tests (param count, Muon math, MoE backend equivalence, tiny overfit wiring check)
```

Checkpoints and tokenized data live on the mounted persistent storage (`/lambda/nfs/h100-f1/...`), never in this repo. `bubblspace-tokenizer/` is the one data artifact that *is* committed (it's ~3.4MB and defines the model's vocabulary, not training data).

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

This project — code, training pipeline, model checkpoints, and any associated
dataset preparation scripts — is published **for learning and research purposes
only**. It is an experimental, independently-trained small language model, not
a finished or production-grade product.

**AIEDX Private Limited bears no responsibility or liability for any output
produced by this model or any software built using this repository.** Use of
the code, checkpoints, or data pipeline is entirely at your own risk, and is
provided "AS IS" without warranty of any kind, express or implied, consistent
with Section 7 of the Apache License 2.0 above.

**On the nature of AI output:** this is a small, base-pretrained (non-instruction-tuned)
language model. Its outputs are statistically plausible continuations of text,
not verified facts, advice, or reasoning. It can be confidently wrong, biased,
inconsistent, or nonsensical, especially at this parameter scale and training
stage — see [`benchmark/README.md`](benchmark/README.md) for where it stands
on standard evaluations. Do not rely on anything it generates for factual,
medical, legal, financial, or safety-critical purposes.

## Acknowledgments

- Tokenizer base: [HuggingFaceTB/SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)
- Pretraining corpus: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- Muon optimizer: Keller Jordan's Newton-Schulz orthogonalization approach, with the Moonlight weight-decay variant
- Evaluation: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
