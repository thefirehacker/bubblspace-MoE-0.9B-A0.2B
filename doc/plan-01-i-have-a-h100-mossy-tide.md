# MoE-0.9B-A2b: train a 917M/213M-active MoE LLM on a single Lambda H100

## Progress so far (this plan is unchanged — implementation is mid-flight)

Local repo scaffolding and all of Phases 2–6's code have been written already: `.gitignore`, `pyproject.toml`, `.env.example`, `configs/{base,debug}.yaml`, `src/config.py`, the full `src/model/` package (`norm.py`, `rope.py`, `attention.py`, `moe.py`, `block.py`, `transformer.py`, `init.py`), the full `src/optim/` package (`muon.py`, `param_groups.py`, `schedule.py`), `src/data/{prepare,dataset}.py`, `scripts/{setup_fire01.sh,train.py}`, and all four `tests/*.py`. A local `.venv_test` was being set up to actually run the CPU-feasible tests (`test_param_count.py`, `test_muon_ns.py`, `test_overfit_tiny.py`, and the fallback half of `test_moe_backends.py`) when plan mode was re-entered — that install was mid-flight in the background and has not been confirmed to pass yet.

**Still outstanding:** running the local test suite to confirm the code is actually correct; the README.md deliverable (not yet written); Phase 1's real storage-region check and Phase 2's real env setup on `fire01` itself (nothing has touched the remote instance yet — all work so far is local files only); and Phases 7–8 (debug run, full run), which depend on SSH access to `fire01`.

## Context

The user has a Lambda Cloud on-demand H100 80GB instance (`fire01`, region `us-west-3`) already running, and wants to train a from-scratch Mixture-of-Experts language model on it against FineWeb-Edu (~10B tokens). The repo (`bubblspace-MoE0.9B-A2b`) is confirmed completely empty — just `git init`, no commits, no remote — so this is a greenfield build, not a modification.

Two architecture specs were given in chat; the user confirmed **v2** ("Qwen3-Next-inspired": 64 experts/top-8, ZeroCenteredRMSNorm, attention output-gate, global-batch load balancing) is the target, superseding v1. The user also flagged a real risk up front: their persistent storage filesystem (believed to be in "Utah") may or may not be in the same Lambda region as `fire01` (`us-west-3`) — and Lambda filesystems cannot be mounted across regions, so this must be resolved before any data download or paid GPU time is spent. The user also wants W&B logging, both a fast (`torch._grouped_mm`) and a portable fallback MoE kernel, and a README + .gitignore as explicit deliverables.

**Outcome:** a working, debug-tested training pipeline for this exact architecture, with the storage question resolved first, that can be left running unattended for the full ~10B-token / ~19k-step run.

## Repository layout

```
bubblspace-MoE0.9B-A2b/
├── README.md
├── .gitignore
├── pyproject.toml
├── .env.example                 # WANDB_API_KEY=, HF_TOKEN=
├── configs/
│   ├── base.yaml                 # full v2 spec
│   └── debug.yaml                # override-only: smaller shapes, ~100M tokens
├── src/
│   ├── config.py                  # dataclasses + base/debug merge
│   ├── model/
│   │   ├── norm.py                 # ZeroCenteredRMSNorm
│   │   ├── rope.py
│   │   ├── attention.py            # GQA + qk_norm + rope + output_gate
│   │   ├── moe.py                  # router + MoE FFN, 2 backends
│   │   ├── block.py
│   │   ├── transformer.py
│   │   └── init.py
│   ├── optim/
│   │   ├── muon.py                 # batched Newton-Schulz (2D and 3D-stacked experts)
│   │   ├── param_groups.py         # Muon vs AdamW split
│   │   └── schedule.py             # warmup+cosine, WSD alt
│   └── data/
│       ├── prepare.py              # download + tokenize fineweb-edu -> uint16 shards
│       └── dataset.py              # memmap reader
├── scripts/
│   ├── setup_fire01.sh             # env + deps + torch._grouped_mm probe
│   └── train.py                    # entrypoint (config, --debug, --resume)
└── tests/
    ├── test_param_count.py
    ├── test_moe_backends.py
    ├── test_muon_ns.py
    └── test_overfit_tiny.py
```

Checkpoints and tokenized data live under the mounted persistent storage (e.g. `/mnt/moe-storage/{checkpoints,data}`), never in the git repo.

## Phase 1 — Storage region gate (first, before anything else)

This blocks all downstream work since it governs whether data/checkpoints have anywhere durable to live.

1. Check the filesystem's region via the Lambda console **Storage** tab (or `GET https://cloud.lambdalabs.com/api/v1/filesystems`) and compare against `fire01`'s region (`us-west-3`).
2. **If they match:** copy the exact NFS mount command Lambda provides for that filesystem (don't hand-construct it — the DNS name is filesystem-specific) and mount at `/mnt/moe-storage`; add it to `/etc/fstab` or a mount step in `setup_fire01.sh` so it survives reboot. No transfer fee beyond the standard ~$0.20/GB/month storage rate.
3. **If they don't match:** Lambda has no cross-region mount. Document two options in the README, recommending the first:
   - Relaunch an H100 instance in the filesystem's actual region and re-point SSH.
   - Stay on `fire01`, use local ephemeral NVMe (wiped on termination) and push checkpoints out periodically — flag this as materially riskier for a multi-hour paid run.
4. Verify with `df -h /mnt/moe-storage` and a write test before proceeding.

## Phase 2 — fire01 environment setup

- Confirm `nvidia-smi` (H100 80GB, driver/CUDA), set up a venv, install a torch build matching the instance's CUDA.
- **Probe `torch._grouped_mm` availability** (`hasattr(torch, "_grouped_mm")`) immediately — this decides whether the fast MoE backend is usable; default `moe.backend` to the fallback if absent.
- Install `tiktoken`, `datasets`, `numpy`, `wandb`, `pyyaml`, `tqdm`, `python-dotenv`.
- `wandb login` (or `.env`, gitignored).
- Push repo to a GitHub remote, clone onto `fire01` (code syncs via git, not via the storage mount).

## Phase 3 — Config system

`configs/base.yaml` holds the full v2 spec verbatim; `configs/debug.yaml` holds only the differing keys (4 layers, d_model 512, 16 experts, top_k 4, expert_hidden 128, ~100M tokens) merged on top at load time — keeps debug and full runs from drifting apart on shared hyperparams. Plain dataclasses in `config.py`, no Hydra/OmegaConf.

## Phase 4 — Model (`src/model/`)

Build bottom-up, testable on CPU with tiny shapes before touching the GPU:

- **`norm.py`**: ZeroCenteredRMSNorm — effective weight `1 + w`, `w` init 0 and weight-decayed (decay pulls toward the *identity* transform, not toward zero gain — worth a one-line comment since it's a deliberate, non-default choice).
- **`rope.py` / `attention.py`**: GQA (16 heads / 4 kv-heads), QK-norm applied per-head before RoPE (assumption — standard Qwen3-style ordering, flagged for confirmation), SDPA with `enable_gqa=True` where available, output-gate = `sigmoid(X @ W_g)` (full `d_model×d_model`) multiplying the SDPA output before `o_proj`, `bias=false` throughout.
- **`moe.py`**: router `Linear(1024→64, bias=False)`, `std=0.02` init (never zero), softmax→top-8→renormalize. Experts stored as stacked `nn.Parameter` tensors `(64,1024,256)` etc., not a `ModuleList`, so both the Muon batched step and grouped-GEMM forward are natural. Two backends behind `moe.backend` config: (a) `torch._grouped_mm` dropless fast path; (b) portable fallback — a dense-masked variant used only as the correctness oracle in tests, and a gather-per-expert-loop variant as the actual usable fallback (a Python loop over 64 experts in the *forward pass* is fine — only the optimizer's NS step needs to avoid looping). Load-balance aux loss (`coef 0.01`) and router z-loss (`coef 0.001`) per spec; global-batch load balancing interpreted as: differentiable per-micro-batch `P_i`, detached running `f_i` accumulated across the 16 grad-accum micro-batches (a literal single-backward-after-16-batches isn't compatible with standard grad accumulation — flagged as an assumption).
- **`block.py` / `transformer.py`**: pre-norm residual block (attn then MoE), tied embedding/lm_head.
- **`init.py`**: `std=0.02` default, `residual_proj_std = 0.02/sqrt(2*16)` on `attn.o_proj` and `experts.W_down`.

**Verify:** `tests/test_param_count.py` builds the model on `meta` device and asserts ≈917M total / ≈213M active params (hand-derived: embedding 51.5M, attention 41.9M, output-gate 16.8M, experts 805M total/100.7M active, routers 1.05M) — cheap, catches shape mistakes before any GPU time.

## Phase 5 — Optimizer (`src/optim/`)

- **`muon.py`**: one batched Newton-Schulz path for everything — unsqueeze ordinary 2D matrices to a batch of 1, stacked experts are already batch-64; quintic NS (`ns_steps=5`), `update_scale = 0.2*sqrt(max(rows,cols))` (constant `6.4` for every Muon matrix in this architecture — a good invariant to assert in a test), momentum 0.95 + Nesterov, weight_decay 0.1 (note: Moonlight variant adds WD vs. original Muon's none — intentional per spec, not an oversight).
- **`param_groups.py`**: Muon = attn q/k/v/o + output-gate proj + expert gate/up/down. AdamW = tied embedding, routers, all norm weights (including the zero-centered norm's `w`, which **is** decayed — easy to "fix" by accident later, worth a comment). Test: every param appears in exactly one group.
- **`schedule.py`**: linear warmup (2000 steps) + cosine decay to `min_lr_ratio=0.1`; WSD implemented as a config-selectable alternative per the spec's note.

**Verify:** `tests/test_muon_ns.py` — NS output ≈ orthogonal (singular values ≈1); batched NS output on a stack of 64 matrices exactly matches looping the single-matrix NS function 64 times (this loop exists only in the test, never in production code).

## Phase 6 — Data, training loop, logging, checkpointing

- **`data/prepare.py`**: stream `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenize with `tiktoken` GPT-2 BPE + EOT separators, write `uint16` memmap shards (vocab 50304 < 65536). Tokenize a small ~100-200M token prefix first for the debug run; the full run just continues from the same ongoing tokenization (no re-download/re-tokenize).
- **`data/dataset.py`**: memmap reader yielding random `(micro_batch=16, seq_len=2048)` windows. Verify by decoding a sampled batch back to text.
- **`scripts/train.py`**: builds model, splits params, constructs Muon+AdamW, runs 16-step grad-accumulation micro-batches (bf16 autocast) per optimizer step, clips grad norm to 1.0, steps both optimizers/schedules. `torch.compile` applied to the model — **real risk flagged**: dropless MoE has data-dependent per-expert token counts that can trigger excessive recompilation; watch this specifically during the debug run and have `compile:false` (or compiling only attention) ready as a fallback rather than discovering it mid-way through the paid full run. Checkpoints every 1000 steps to `/mnt/moe-storage/checkpoints/<run>/step_<n>.pt` (model + both optimizer states + step/RNG). `max_attn_logit` computed periodically (every ~100 steps) via an unfused attention recompute on one micro-batch, not every step. W&B logs: per-step loss/lb_loss/z_loss, per-optimizer-step grad_norm/tokens_per_sec, periodic tokens_per_expert histogram and max_attn_logit.

## Phase 7 — Debug run (gate before the full run)

1. `tests/test_overfit_tiny.py` — tiny model, one fixed batch, loss → ~0 (wiring check, CPU-feasible, cheapest possible smoke test).
2. Run `configs/debug.yaml` end-to-end on the H100 GPU with the small real data prefix: confirm data loading, forward/backward, Muon step, routing, checkpoint save, W&B logging all work together.
3. Kill and `--resume` from a checkpoint — confirm step/token counters and loss continue smoothly, no spike.
4. Confirm `tokens_per_expert` trends toward uniform across debug experts as load-balance loss takes effect.
5. Confirm loss decreases steadily, no NaN/explosion, `grad_norm` stable.
6. Check `torch.compile` recompilation behavior under the debug config before trusting it for the full run.
7. If `torch._grouped_mm` is available, run debug once per backend and confirm loss curves track closely as an integration-level cross-check.

Only proceed to Phase 8 once all of these pass.

## Phase 8 — Full run

Launch `scripts/train.py configs/base.yaml` (no `--debug`) once the full ~10B-token dataset is tokenized and the Phase 7 gate has passed. Monitor the first few hundred steps closely (loss, grad_norm, routing balance, tokens/sec vs. expected H100 bf16 throughput for ~213M active params) before leaving it unattended.

## Deliverables

- **README.md**: fire01/storage setup (with the actual resolved mount command), how to run debug then full, architecture/param-count summary table, the open assumptions listed below.
- **.gitignore**: `__pycache__/`, `*.pyc`, `.venv/`, `wandb/`, `*.pt`, `*.ckpt`, `checkpoints/`, `data/`, `*.bin`, `*.idx`, `.env`.
- **pyproject.toml**: pinned `torch`, `tiktoken`, `datasets`, `numpy`, `wandb`, `pyyaml`, `tqdm`, `python-dotenv`.

## Open assumptions to flag in the README (spec was a one-line gloss on each)

1. QK-norm applied before RoPE (norm-then-rotate).
2. Global-batch load balancing = differentiable per-micro-batch router prob, detached running count across the 16 grad-accum micro-batches (not a literal single aux-loss backward after all 16).
3. MoE fallback = dense-masked (test oracle only) + gather-per-expert-loop (actual fallback, loop lives in forward pass, not in Muon).
4. `torch.compile` scope may need to narrow (e.g. exclude MoE routing) depending on what the debug run shows.

## Critical files (highest implementation risk)

- [src/model/moe.py](src/model/moe.py) — two backends, router, load-balance/z-loss; most spec ambiguity.
- [src/optim/muon.py](src/optim/muon.py) — unified batched NS across ordinary and per-expert-stacked weights.
- [src/optim/param_groups.py](src/optim/param_groups.py) — correctness-critical Muon/AdamW split.
- [scripts/train.py](scripts/train.py) — integration point exercised by the debug run.
- Phase 1 storage gate — not code, but the step everything else is sequenced around.
