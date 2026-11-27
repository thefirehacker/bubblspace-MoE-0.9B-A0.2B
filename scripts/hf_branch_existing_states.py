"""Move the training states that `main` of the HF training-states repo holds as folders (phase1-step19073, ...) into
their own OLMo-style branches, server side: files are copied within the repo (no re-upload), then the other
checkpoint folders are deleted on that branch, so each branch holds exactly one state at its root + README.

`main` itself is not changed.

    python scripts/hf_branch_existing_states.py
"""
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

from hf_push_training_state import README, REPO, TOKENS_PER_STEP

STATES = [  # folder on main, phase, phase-local step, global step, W&B run, config
    ("phase1-step19073", "1", 19073, 19073, "goa8jk6e", "configs/base.yaml"),
    ("phase2-step19000", "2", 19000, 19073 + 19000, "lec9ww5h", "configs/phase2.yaml"),
    ("phase2-step22888", "2", 22888, 19073 + 22888, "lec9ww5h", "configs/phase2.yaml"),
]
FILES = ("model.safetensors", "optimizer.safetensors", "rng.safetensors", "trainer_state.json")


def main():
    api = HfApi()
    main_files = api.list_repo_files(REPO)
    folders = sorted({f.split("/")[0] for f in main_files if f.startswith("phase") and "/" in f})
    existing = [b.name for b in api.list_repo_refs(REPO).branches]
    for folder, phase, step, gstep, run, cfg in STATES:
        tokens = gstep * TOKENS_PER_STEP / 1e9
        branch = "phase%s-step%d-tokens%dB" % (phase.replace(".", ""), step, round(tokens))
        if branch in existing:
            print("exists:", branch)
            continue
        api.create_branch(REPO, branch=branch)
        readme = README.format(branch=branch, phase=phase, step=step, global_step=gstep, tokens=tokens,
                               repo=REPO, config=cfg, wandb_run=run)
        ops = [CommitOperationCopy(src_path_in_repo=f"{folder}/{f}", path_in_repo=f) for f in FILES]
        ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
        api.create_commit(REPO, operations=ops, revision=branch,
                          commit_message=f"Training state {branch} (Phase {phase} step {step}, {tokens:.2f}B tokens)")
        api.create_commit(REPO, operations=[CommitOperationDelete(path_in_repo=f"{d}/", is_folder=True) for d in folders],
                          revision=branch, commit_message=f"{branch}: keep only this training state")
        print("branched", branch, "->", sorted(api.list_repo_files(REPO, revision=branch)))


if __name__ == "__main__":
    main()
