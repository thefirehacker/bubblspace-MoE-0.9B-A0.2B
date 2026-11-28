# Recovery: continuing on a fresh H100 with a new NFS

Plan (27 Nov): after RL v1, the H100 (fire01) and its `h100-f1` filesystem may be deleted to save cost. Work must then be able to continue on a **new H100 with a new, empty NFS**, using only GitHub, Hugging Face and W&B. This document lists what lives where and how to bring a new machine up.

**28 Nov: fire01 (the H100) has been terminated.** Its `h100-f1` filesystem may still exist in Lambda (1.15 TB); everything needed to continue is on Hugging Face and GitHub (§1–2). Lost with fire01: RL v1's training-log text after 18:15 UTC on 27 Nov (W&B keeps the curves) and files on its home disk (the run 02b / RL v1 quick checks, redone on the A10 in `benchmark/phase4/`).

**Tested 27 Nov:** the Phase 3 step 1,500 training state downloaded from Hugging Face and rebuilt with `restore_checkpoint.py` is **bit-identical** to the original `.pt` (all 915,573,760 parameters, both optimizers, schedulers, RNG, step).

## 1. What lives where

| What | Where | Notes |
|---|---|---|
| **Code, configs, docs, eval results** | GitHub: [thefirehacker/bubblspace-MoE-0.9B-A0.2B](https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B) | `benchmark/` holds every eval result (JSON/MD) |
| **Full training states** (weights + optimizer + schedulers + RNG): the **last two checkpoints of every stage** | HF: [bubblspace-moe0.9b-a0.2b-training-states](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states), one branch per checkpoint | see §2 |
| **Runnable models** (weights + code + tokenizer + `generate.py`) | HF: [bubblspace-moe0.9b-a0.2b](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b) | `main` = the current best model; one branch per published model |
| **RL v1** (done 27 Nov) | training-states repo: steps 100 / 300 / 500, **with every rollout** (`rollouts/step_NNNN.jsonl.gz`), eval samples, `train_log.jsonl`, `evals.jsonl`; model repo: the runnable final model | rollouts: prompt, answer, reward, advantage for every sample of every step |
| **Training curves** | W&B: [moe0-9b-a0-2b](https://forge.coreweave.com/wandb/firehacker/moe0-9b-a0-2b) | runs listed in README.md |
| **Our generated datasets** | HF datasets: [bubblspace-contrastive-binding](https://huggingface.co/datasets/TheFireHacker/bubblspace-contrastive-binding), [bubblspace-relational-passages-Qwen3-30B](https://huggingface.co/datasets/TheFireHacker/bubblspace-relational-passages-Qwen3-30B) | |
| **Third-party training data** (FineWeb-Edu, Wikipedia, Tülu 3, ToolMind, …) | their original sources; **not re-uploaded** | rebuilt with our scripts (§4) |

**Lost if the `h100-f1` NFS is deleted (by choice):**
- intermediate checkpoints (every 1,000 steps), about 550 GB. Only the last two of each stage are kept;
- tokenized data (about 140 GB) and raw downloads (290 GB), which our scripts rebuild;
- the HF cache.

**Training logs** (about 10 MB, 80 files from fire01 and the A10; IP addresses, e-mails and tokens redacted) are on the training-states repo under [`logs/`](https://huggingface.co/TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states/tree/main/logs) (snapshot 27 Nov 12:45 UTC; refresh after run 02b and RL v1).

## 2. Checkpoint branches (training-states repo)

| Stage | Branches (last two) | Config |
|---|---|---|
| Phase 1 (English base) | `phase1-step19000-tokens10B`, `phase1-step19073-tokens10B` | `configs/base.yaml` |
| Phase 2 | `phase2-step22000-tokens22B`, `phase2-step22888-tokens22B` (+ `phase2-step19000-tokens20B`, plateau end) | `configs/phase2.yaml` |
| Phase 2.5a | `phase25a-step29000-tokens37B`, `phase25a-step29723-tokens38B` (+ `step20000`, `step28000`) | `configs/phase25a.yaml` |
| Phase 2.5b (16k context; **the base model**) | `phase25b-step2000-tokens39B`, `phase25b-step2861-tokens39B` | `configs/phase25b.yaml` |
| Phase 3 SFT run 01 | `phase3-step1000-tokens40B`, `phase3-step1500-tokens40B` | `configs/phase3.yaml` |
| Phase 3 SFT run 02b | `phase3-run02b-step1000-tokens40B`, `phase3-run02b-step1500-tokens40B` | `configs/phase3_run02b.yaml` |
| Phase 4 RL v1 | `phase4-rlv1-step100-tokens40B`, `phase4-rlv1-step300-tokens40B`, `phase4-rlv1-step500-tokens40B` (each with rollouts, test answers, logs; also runnable branches of the same names on the model repo) | `configs/rl_v1.yaml` |

## 3. Bring up a new machine

```bash
# 0. Lambda: launch an H100 and create a filesystem in the SAME region (filesystems can't be mounted across regions);
#    mount it, e.g. at /lambda/nfs/<name>. Update data_dir / checkpoint_dir in the configs if the name differs.

# 1. code + environments
git clone https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B && cd bubblspace-MoE-0.9B-A0.2B
bash scripts/setup_fire01.sh          # training env (.venv): torch, our package, W&B
bash scripts/setup_rl_env.sh          # vLLM env (~/vllm_env, Python 3.12) + plugin + docker sandbox image (RL, fast evals)
hf auth login && wandb login          # tokens: yours (never in the repo)

# 2. a checkpoint from HF -> the exact training .pt
hf download TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states --revision phase25b-step2861-tokens39B --local-dir states/phase25b-step2861
python scripts/restore_checkpoint.py restore --dir states/phase25b-step2861 --pt /lambda/nfs/<name>/checkpoints/phase25b/step_2861.pt

# 3a. continue the same run:      python scripts/train.py <its config> --resume <step.pt>
# 3b. start a new stage from it:  python scripts/train.py <new config> --init_from <step.pt>
# 3c. RL from an SFT model:       python scripts/train_grpo.py configs/rl_v1.yaml --set init_checkpoint=<step.pt> ...
```

For inference or evals only, the runnable model repo is enough: `hf download TheFireHacker/bubblspace-moe0.9b-a0.2b --revision <branch>`, then `python generate.py --chat ...`. For fast batched evals, build a vLLM model folder: `python scripts/make_vllm_model_dir.py --weights model.safetensors ...`.

## 4. Rebuilding data (only what a new stage needs)

| Needed for | Rebuild with | Time (A10 CPUs) |
|---|---|---|
| SFT (Phase 3 run 02b recipe) | `scripts/phase3_download.py` → `scripts/phase3_gen_own_sets.py` (+ `--only own_facts_cmd`, `--only own_realtime`) → `scripts/phase3_build_sft.py` → `scripts/phase3_decon.py` (index: `phase3_decon.py index`) | about 1–2 h |
| RL task set | download `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints` + MBPP; `scripts/rl_build_tasks.py` (seeded, so the same tasks come out) → `scripts/rl_filter.py` with the chosen model | build minutes; filter about 2.5 h on the A10 |
| Continued pretraining (e.g. a knowledge stage) | the data scripts (`src/data/prepare.py`, `scripts/prepare_agentic.py`, `scripts/decontaminate.py`; recipes in doc/data-pipeline.md and the phase configs) | hours, CPU-bound |

Every downloaded dataset or model gets a malware scan (HF scan status, or ClamAV) before use, as before.

## 5. Before deleting the old NFS: checklist

- [x] All branches in §2 present on HF. Run the audit script: every training-state branch has `model.safetensors`, `optimizer.safetensors`, `rng.safetensors`, `trainer_state.json`, `restore_checkpoint.py`, `README.md`. *(28 Nov: all 18 branches pass; the RL branches also have `train_log.jsonl`, `evals.jsonl` and every rollout.)*
- [x] RL v1 final + second-to-last pushed with rollouts and logs. *(Steps 100 / 300 / 500, both repos.)*
- [x] `benchmark/` results from the A10 and fire01 copied into the repo and pushed to GitHub. *(`benchmark/phase4/`, merged to `main` in PR #32. fire01's home-disk quick checks were lost and redone on the A10.)*
- [ ] Logs refreshed on HF (`logs/`, redacted) after the last runs on this NFS. *(Still the 27 Nov 12:45 UTC snapshot. fire01's later logs can only come off `h100-f1` if it still mounts; the A10's Phase 4 benchmark logs are in `benchmark/phase4/`.)*
- [ ] Any experiment-only files still wanted (e.g. WSP edits) copied off.
- [ ] The A10's filesystem (`us-west-1-eval-inference`, California) is separate; decide whether to keep it.
