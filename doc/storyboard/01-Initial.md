# Storyboard 01 — From architecture spec to a passing debug run

This covers everything from the first message to the debug run finishing successfully on `fire01`. Chronological, including the bugs.

## 1. The ask

You had an H100 instance (`fire01`, Lambda Cloud, region `us-west-3`) already running and a hand-written architecture spec for a ≈0.9B-total/≈0.2B-active MoE language model (GQA + QK-norm + RoPE, 64-expert dropless top-8 SwiGLU MoE, ZeroCenteredRMSNorm, Muon + AdamW optimizers), to be trained on FineWeb-Edu. You also gave a persistent-storage filesystem ID and asked, up front, whether mounting it would involve any cross-region transfer cost.

## 2. Planning

Before writing any code: explored the repo (confirmed it was completely empty — just `git init`, no commits), then produced an implementation plan covering, in order:

1. **Storage region gate** — check whether the filesystem is in the same Lambda region as `fire01` before downloading anything, since Lambda filesystems can't be mounted cross-region.
2. **Environment setup** on `fire01`.
3. **Config system** (`configs/base.yaml` + `configs/debug.yaml`, debug as an override-merge on top of base).
4. **Model code**, built bottom-up and CPU-testable before touching the GPU.
5. **Muon optimizer**.
6. **Data pipeline + training loop + logging + checkpointing**.
7. **A cheap debug run**, as a gate before spending real money on the full run.
8. **The full run**.

A few spec ambiguities were flagged and resolved with documented assumptions rather than guessed at silently: QK-norm/RoPE ordering, how "global-batch" load balancing should actually work under gradient accumulation, and the fact that `torch._grouped_mm` (the fast MoE kernel) is a private, unstable PyTorch API that needed a runtime probe and a fallback.

## 3. Writing the code, bottom-up

In roughly this order:

- **Scaffolding**: `.gitignore`, `pyproject.toml`, `.env.example`, `configs/base.yaml`, `configs/debug.yaml`, `src/config.py` (plain dataclasses + a base/debug deep-merge, no Hydra).
- **Model** (`src/model/`): `norm.py` (ZeroCenteredRMSNorm), `rope.py`, `attention.py` (GQA + QK-norm + RoPE + an elementwise output gate), `moe.py` (router + two interchangeable MoE backends — `torch._grouped_mm` fast path and a gather/per-expert-loop fallback — plus the load-balance and router z-loss), `block.py`, `transformer.py`, `init.py`.
- **Optimizer** (`src/optim/`): `muon.py` (batched Newton-Schulz orthogonalization, unified across ordinary 2D matrices and per-expert 3D stacks so there's no Python loop over experts in the optimizer), `param_groups.py` (the Muon-vs-AdamW parameter split), `schedule.py` (warmup + cosine, with WSD as an alternative).
- **Data** (`src/data/`): `prepare.py` (tokenize into `uint16` shards), `dataset.py` (memmap reader).
- **Scripts**: `setup_fire01.sh`, `train.py`.
- **Tests**: `test_param_count.py`, `test_moe_backends.py`, `test_muon_ns.py`, `test_overfit_tiny.py` — all written to run on CPU, no GPU or real data needed, so bugs could be caught before spending any GPU time.

### Bug #1 — MoE expert weights left uninitialized

Running the local test suite for the first time, `test_moe_backends.py` came back with **NaN everywhere**. Traced it to `MoEFFN`'s expert weight tensors (`W_gate`, `W_up`, `W_down`) being created with `torch.empty(...)` — garbage memory — and only ever getting properly initialized by the *Transformer*-level `init_weights` function. Testing `MoEFFN` standalone (exactly what the unit test does) skipped that step entirely, so the weights stayed as uninitialized garbage.

**Fix**: gave `MoEFFN` a sane default `std=0.02` init of its own at construction time, so it's safe to use standalone; the Transformer-level `init_weights` still rescales `W_down` to the residual-stream-scaled init when used inside a full model.

### Bug #2 — an overly strict Newton-Schulz test (test bug, not a code bug)

`test_newton_schulz_orthogonality` failed: after 5 quintic Newton-Schulz iterations on a random 64×64 matrix, the output wasn't close enough to a perfect identity Gram matrix. Investigated with real singular-value decomposition — turned out this is expected behavior of the actual Muon algorithm (matches Keller Jordan's reference implementation): a few NS iterations don't fully converge every singular value of a full-rank random matrix, especially the smallest ones. The test's expectation was wrong, not the implementation.

**Fix**: rewrote the test to check the realistic invariant (largest singular value lands near 1, nothing explodes) instead of demanding a perfect identity matrix.

Once both were fixed, the full local CPU test suite passed (7 passed, 1 skipped — the `grouped_mm` comparison needs CUDA).

## 4. Onto `fire01` for real

SSH'd in (found the right key, `wsp.pem`, after a couple of wrong guesses), ran `nvidia-smi` and confirmed: H100 PCIe 80GB. Checked the storage question from the plan — a Lambda filesystem was **already mounted** at `/lambda/nfs/h100-f1` (confirmed via `mount`, `df -h`, and a live write test). Since Lambda only allows attaching a same-region filesystem at instance launch, this being mounted at all already proved it was region-matched — no separate transfer-cost question to resolve.

Ran `setup_fire01.sh`: built the venv, installed dependencies, and — critically — probed `torch._grouped_mm` availability for real. It worked: `torch 2.14.1+cu130`, CUDA available, `grouped_mm usable: True`. So the fast MoE path, not the fallback, would be the default on this box.

## 5. Tokenizer swap (mid-stream requirement change)

You then asked to replace the GPT-2 tokenizer with SmolLM2's base tokenizer plus 12 BubblSpace special tokens (chat `<|im_start|>`/`<|im_end|>`, FIM tokens for code, tool-call tokens, `<think>`/`</think>` for reasoning) — no new tokenizer to be trained, just extended. Also asked to start planning for coding + AI-agentic training, not just general English.

Wrote `scripts/build_tokenizer.py`, ran it for real: base SmolLM2 tokenizer is 49,152 tokens; `<|im_start|>`/`<|im_end|>` were already present (correctly skipped, not duplicated); the other 12 were added, bringing it to 49,164; padded to **49,280** (multiple of 128) for `model.vocab_size`. Verified every special token encodes to exactly one id, and round-trip (`decode(encode(x)) == x`) holds.

Wired it through: `configs/base.yaml` updated, `src/data/prepare.py` rewritten to tokenize with this tokenizer instead of `tiktoken`/GPT-2, `src/data/fim.py` added (PSM fill-in-the-middle transform, applied only to code when explicitly enabled), `scripts/train.py` updated to slice the 116 padding vocab rows out of the loss entirely (not just rely on cross-entropy to ignore them — they're excluded from the softmax too), `scripts/sanity_check_data.py` added for round-trip / bytes-per-token / max-token-id checks.

**A correction worth recording**: the first bytes-per-token comparison against GPT-2 used a 56-byte toy string and I misread my own output, telling you SmolLM2's tokenizer was "less efficient on code." It was backwards. Re-measured properly on ≈800KB of *real* held-out text (real FineWeb-Edu docs, real Python stdlib source, real FineMath docs) and got the opposite, correct conclusion: SmolLM2-based tokenizer is **≈1.7x more token-efficient on Python code** than GPT-2, roughly tied on general English, and only ≈7% behind on math. No reason to switch.

Also added recommended datasets for the coding/agentic goal (English: FineWeb-Edu + Cosmopedia-v2; code: Python-Edu + Stack-v2 subsets for JS/frontend and multi-language backend, per your later ask; agentic: xLAM/Hermes function-calling + OpenThoughts reasoning traces) to the README as a documented plan — not yet downloaded or wired into the pipeline.

## 6. Naming fix

You caught that the project was being called "MoE-0.9B-A2b" when it should be "**MoE-0.9B-A0.2B**" (0.2B active params, not 2B). Fixed everywhere: README title, `pyproject.toml` package name, both configs' `run_name`/W&B `project` fields, comment headers.

## 7. Git

You committed and pushed manually at a couple of points. At one point I also found and fixed a `.gitignore` bug myself (see Bug #3 below) but, per your stated preference, didn't commit it — just described the fix and let you commit it, which you then did yourself (and it turned out you'd independently made the identical fix).

### Bug #3 — `.gitignore` silently deleting the data pipeline from every commit

Re-cloned the repo fresh onto `fire01` (cleaner than the original `rsync`) and the debug data-prep script immediately failed: `ModuleNotFoundError: No module named 'src.data'`. The entire `src/data/` directory — `prepare.py`, `dataset.py`, `fim.py`, `__init__.py` — was missing from the pushed repo, even though it existed locally.

Root cause: my own `.gitignore` had a bare `data/` pattern (intended to exclude a local staging directory for downloaded datasets). Gitignore patterns without a leading slash match a directory of that name at **any depth** — so it was silently matching `src/data/` too and excluding the entire source subpackage from every commit, invisibly, since nothing ever flagged it as "ignored" to look at.

**Fix**: rooted the pattern (`/data/`, `/checkpoints/`) so it only matches at the repo root, and swapped an overly-broad `*.bin` for the more specific `shard_*.bin`. You independently made and pushed the identical fix before I did.

Also found, along the way, a genuinely stale leftover: `moe0.9b-a2b` on `fire01` — my very first `rsync`'d copy from before the project was renamed, with no git tracking at all. Confirmed it had nothing of value and deleted it, once you'd confirmed deletion was fine.

## 8. The debug run — and Bug #4

With the data pipeline finally present on `fire01`, re-ran `src.data.prepare`: tokenized 150,000,475 real FineWeb-Edu tokens into 2 shards in under 4 minutes. `sanity_check_data.py` confirmed every token id stayed within the real vocab range and every special token round-tripped correctly.

You asked for W&B to be included (not run offline) and set up the API key yourself on `fire01`, both via `.env` and interactive `wandb login`. Confirmed both were in place, then launched the debug training run (detached via `nohup`/`disown` so it would survive SSH disconnects).

It crashed almost immediately.

### Bug #4 — `torch._grouped_mm` isn't autocast-aware

```
torch._dynamo.exc.TorchRuntimeError: ... Expected mat_a to be BFloat16 or
supported Float16 matrix, got torch.float32.
```

The training loop wraps the forward pass in `torch.autocast(dtype=torch.bfloat16)`, which normally casts eligible ops to bf16 automatically. `torch._grouped_mm` is a private, experimental op that autocast doesn't know about — it never got cast, received plain `float32` tensors, and the native kernel rejected them. This only showed up under `torch.compile`'s fake-tensor tracing, on the real GPU — impossible to catch with the earlier CPU-only tests (no CUDA, so that code path was always skipped locally).

**Fix**: in `src/model/moe.py`'s `_forward_grouped_mm`, explicitly cast the gathered tokens and the expert weights to `bfloat16` right before each `torch._grouped_mm` call, instead of relying on ambient autocast. Verified the fix by copying just that one file onto `fire01` directly (not yet committed — again, following your preference to handle commits yourself) and re-running the test suite there: all 8 tests passed, including the real `grouped_mm`-vs-fallback comparison that only runs with CUDA present.

## 9. The debug run, take two — it worked

Relaunched. This time it ran clean, start to finish:

- **3051/3051 steps** completed, ≈220–250k tokens/sec on the H100.
- **Loss**: trended from ≈4.2–4.3 down to ≈3.8–4.0.
- **Load-balance loss**: stayed at ≈1.01–1.02 throughout — very close to the theoretical ideal of 1.0, meaning the 16 debug experts were being routed to almost perfectly evenly.
- **grad_norm**: stable the whole run, no spikes.
- Checkpoints saved at steps 1400, 3000, and the final 3051.
- W&B logged live, online, to your account the whole time.
- MoE backend confirmed resolved to `grouped_mm` (the fast path) and used throughout — not the fallback.

Then ran the checkpoint-resume test explicitly: killed nothing, just loaded the `step_3000` checkpoint fresh and resumed training from there. Loss continued in the same 3.7–4.1 range with no spike or discontinuity, `grad_norm` comparable to the original run — confirming the Muon momentum buffers, AdamW moment estimates, and LR schedule state all round-trip through a checkpoint save/load correctly.

## 10. Where things stand now

Every Phase 7 "debug gate" criterion from the original plan is met. The `torch._grouped_mm` dtype fix is sitting on `fire01` (and in your local working tree) but not yet committed — that's next. After that, the open decision is whether to launch the full ≈10B-token FineWeb-Edu-only run now, or wire in the expanded code/agentic data mixture (Python-Edu, Stack-v2 JS/backend subsets, function-calling + reasoning datasets) first.
