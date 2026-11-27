#!/bin/bash
# A10: push EVERY relayed RL checkpoint (scripts/rl_relay_each.sh) to both HF repos, like the other stages:
#   training-states repo: branch phase<LABEL>-step<N>-tokens40B = full state (exported pickle-free, bit-exact verified)
#                         + rollouts / test answers / logs up to step N
#   model repo:           branch phase<LABEL>-step<N>-tokens40B = runnable model (weights, code, tokenizer, generate.py)
# then deletes its local copy of the relayed .pt (the original stays on fire01; the state is on HF).
#   RUN=Phase4_rl_v1 LABEL=4-rlv1 TITLE="Phase 4 RL v1" SUMMARY_LEAD="..." nohup bash scripts/rl_push_each.sh &
set -u
RUN=${RUN:?}; LABEL=${LABEL:?}; TITLE=${TITLE:?}; SUMMARY_LEAD=${SUMMARY_LEAD:?}
cd /home/ubuntu/moe0.9b-a0.2b
R=/lambda/nfs/us-west-1-eval-inference/$RUN
LOG=/home/ubuntu/phase3_logs/rl_push_each_$RUN.log
say () { echo "[$(date -u)] $*" >> $LOG; }
while true; do
  for READY in $(ls $R/READY_step* 2>/dev/null | sort -V); do
    s=${READY##*READY_step}
    [ -f $R/PUSHED_step$s ] && continue
    BR=phase${LABEL}-step${s}-tokens40B; E=/home/ubuntu/hf_export/$BR
    WRUN=$(cat $R/meta_step$s/wandb_run.txt 2>/dev/null || echo unknown)
    say "step $s: export"
    nice -n 10 .venv/bin/python scripts/restore_checkpoint.py export --pt $R/step_$s.pt --out_dir $E --verify >> $LOG 2>&1 || { say "EXPORT FAILED $BR"; touch $R/PUSHED_step$s; continue; }
    mkdir -p $E/rollouts $E/eval_samples
    .venv/bin/python - "$R/meta_step$s" "$E" "$s" <<'PY'
import json, shutil, sys
from pathlib import Path
src, dst, upto = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for sub in ("rollouts", "eval_samples"):
    for f in sorted((src / sub).glob("step_*.jsonl.gz")):
        if int(f.name[5:9]) <= upto:
            shutil.copy(f, dst / sub / f.name)
for name in ("train_log.jsonl", "evals.jsonl"):
    rows = [json.loads(l) for l in open(src / name)] if (src / name).exists() else []
    (dst / name).write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= upto))
if (src / "run_config.json").exists():
    shutil.copy(src / "run_config.json", dst / "run_config.json")
PY
    .venv/bin/python scripts/hf_push_training_state.py --folder $E --phase $LABEL --step $s --global_step 76045 \
      --wandb_run "$WRUN" --config configs/rl_v1.yaml >> $LOG 2>&1; rc1=$?
    .venv/bin/python scripts/hf_push_inference_branch.py --export_dir $E --branch $BR --config configs/phase3_run02b.yaml \
      --title "$TITLE, step $s" --wandb_run "$WRUN" --tags "sft,rl,grpo,chat,tool-calling" \
      --summary "$SUMMARY_LEAD GRPO with verifiable rewards (maths, time problems, instruction constraints with known answers, code run in a sandbox, tool calls incl. parallel and no-tool-needed, who-did-what, refusal balance) on top of Phase 3 SFT run 02b. Method: doc/rl-primer-phase4.md; rollouts and logs on the training-states branch of the same name." >> $LOG 2>&1; rc2=$?
    say "step $s: training state rc=$rc1, model branch rc=$rc2"
    [ $rc1 = 0 ] && [ $rc2 = 0 ] && rm -f $R/step_$s.pt
    touch $R/PUSHED_step$s
  done
  if [ -f $R/ALL_RELAYED ] && [ $(ls $R/READY_step* 2>/dev/null | wc -l) = $(ls $R/PUSHED_step* 2>/dev/null | wc -l) ]; then
    say "ALL PUSHED $RUN"; exit 0
  fi
  sleep 60
done
