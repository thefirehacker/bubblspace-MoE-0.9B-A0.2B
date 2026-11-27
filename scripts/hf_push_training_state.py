"""Upload one exported training state (scripts/restore_checkpoint.py export --verify) to its own branch of the HF
training-states repo, OLMo-style: branch `<phase>-step<N>-tokens<X>B`, holding only that checkpoint at the root.

The branch starts from `main`; the checkpoint folders that `main` holds (phase1-*, phase2-*) are deleted on the branch in
the same commit, so each branch contains exactly one training state, a README and the restore script.

    python scripts/hf_push_training_state.py --folder ~/hf_export/phase25a-step20000-tokens32B \
        --phase 2.5a --step 20000 --global_step 61961 --wandb_run l2i2yoqf --config configs/phase25a.yaml
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi

REPO = "TheFireHacker/bubblspace-moe0.9b-a0.2b-training-states"
TOKENS_PER_STEP = 524_288

README = """# {branch}

Full training state of **BubblSpace MoE-0.9B-A0.2B**: Phase {phase}, step {step} (global step {global_step}, {tokens:.2f}B tokens seen in total).

| File | Content |
|---|---|
| `model.safetensors` | model weights |
| `optimizer.safetensors` | every tensor of the Muon and AdamW optimizer states |
| `rng.safetensors` | CPU/CUDA RNG state |
| `trainer_state.json` | step, both LR scheduler states, optimizer scalars/param groups, run config |

Exported pickle-free with `restore_checkpoint.py export --verify` (bit-exact round trip against the original `.pt`).

Restore and resume (or branch a new phase from it):

```bash
hf download {repo} --revision {branch} --local-dir {branch}
python scripts/restore_checkpoint.py restore --dir {branch} --pt step_{step}.pt
python scripts/train.py {config} --resume step_{step}.pt        # continue this phase (W&B run {wandb_run}: set WANDB_RUN_ID={wandb_run})
python scripts/train.py <next_phase>.yaml --init_from step_{step}.pt   # or start the next phase from it
```

Code: https://github.com/thefirehacker/bubblspace-MoE-0.9B-A0.2B
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--phase", required=True, help="e.g. 2.5a")
    ap.add_argument("--step", type=int, required=True, help="phase-local step")
    ap.add_argument("--global_step", type=int, required=True)
    ap.add_argument("--wandb_run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restore_script", default="scripts/restore_checkpoint.py")
    a = ap.parse_args()
    tokens = a.global_step * TOKENS_PER_STEP / 1e9
    branch = "phase%s-step%d-tokens%dB" % (a.phase.replace(".", ""), a.step, round(tokens))
    folder = os.path.expanduser(a.folder)
    if os.path.basename(folder.rstrip("/")) != branch:
        raise SystemExit("folder name %s doesn't match branch %s" % (os.path.basename(folder), branch))
    for f in ("model.safetensors", "optimizer.safetensors", "rng.safetensors", "trainer_state.json"):
        if not os.path.exists(os.path.join(folder, f)):
            raise SystemExit("missing %s in %s (run restore_checkpoint.py export --verify first)" % (f, folder))
    with open(os.path.join(folder, "README.md"), "w") as fh:
        fh.write(README.format(branch=branch, phase=a.phase, step=a.step, global_step=a.global_step, tokens=tokens,
                               repo=REPO, config=a.config, wandb_run=a.wandb_run))
    shutil.copy(a.restore_script, os.path.join(folder, "restore_checkpoint.py"))
    api = HfApi()
    if branch not in [b.name for b in api.list_repo_refs(REPO).branches]:
        api.create_branch(REPO, branch=branch)
    api.upload_folder(repo_id=REPO, folder_path=folder, revision=branch, delete_patterns=["phase*/**"],
                      commit_message="Training state %s (Phase %s step %d, %.2fB tokens)" % (branch, a.phase, a.step, tokens))
    files = sorted(api.list_repo_files(REPO, revision=branch))
    print("uploaded", branch, "->", files)


if __name__ == "__main__":
    main()
