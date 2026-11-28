#!/bin/bash
# A10: export + push RL v1's last two checkpoints to the HF training-states repo, each WITH the rollouts / test answers /
# logs up to that step, and the final one as a runnable model branch on the model repo. Waits for scripts/rl_v1_relay.sh.
set -u
cd /home/ubuntu/moe0.9b-a0.2b
R=/lambda/nfs/us-west-1-eval-inference/Phase4_rl_v1
LOG=/home/ubuntu/phase3_logs/rl_v1_push.log
say () { echo "[$(date -u)] $*" >> $LOG; }
until [ -f $R/meta/META_DONE ]; do sleep 120; done
RUN=$(cat $R/meta/wandb_run.txt 2>/dev/null || echo unknown)
for s in 400 500; do
  BR=phase4-rlv1-step${s}-tokens40B; E=/home/ubuntu/hf_export/$BR
  .venv/bin/python scripts/restore_checkpoint.py export --pt $R/step_$s.pt --out_dir $E --verify >> $LOG 2>&1 || { say "EXPORT FAILED $BR"; continue; }
  mkdir -p $E/rollouts $E/eval_samples
  .venv/bin/python - "$R/meta" "$E" "$s" <<'PY'
import shutil, sys, json
from pathlib import Path
src, dst, upto = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for sub in ("rollouts", "eval_samples"):
    for f in sorted((src / sub).glob("step_*.jsonl.gz")):
        if int(f.name[5:9]) <= upto:
            shutil.copy(f, dst / sub / f.name)
for name in ("train_log.jsonl", "evals.jsonl"):
    rows = [json.loads(l) for l in open(src / name)] if (src / name).exists() else []
    (dst / name).write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= upto))
shutil.copy(src / "run_config.json", dst / "run_config.json")
PY
  .venv/bin/python scripts/hf_push_training_state.py --folder $E --phase 4-rlv1 --step $s --global_step 76045 \
    --wandb_run "$RUN" --config configs/rl_v1.yaml >> $LOG 2>&1
  say "training state $BR rc=$?"
done
BR=phase4-rlv1-step500-tokens40B
.venv/bin/python scripts/hf_push_inference_branch.py --export_dir /home/ubuntu/hf_export/$BR --branch $BR \
  --config configs/phase3_run02b.yaml --title "Phase 4 RL v1, step 500" --wandb_run "$RUN" --tags "sft,rl,grpo,chat,tool-calling" \
  --summary "Phase 4 **RL v1**: 500 steps of GRPO with verifiable rewards (maths, time problems, instruction constraints with known answers, code run in a sandbox, tool calls incl. parallel and no-tool-needed, who-did-what, refusal balance) on top of Phase 3 SFT run 02b. Method: doc/rl-primer-phase4.md; rollouts and logs on the training-states branch." >> $LOG 2>&1
say "model branch $BR rc=$?"
say "RL V1 PUSH DONE"
