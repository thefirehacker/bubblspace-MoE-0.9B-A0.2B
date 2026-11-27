#!/bin/bash
# A10: wait for a relayed checkpoint to arrive complete, then export it pickle-free (bit-exact check) and push it to its
# own OLMo-style branch of the HF training-states repo (scripts/hf_push_training_state.py). Standing rule: the final
# checkpoint of every stage is uploaded (doc/plan-04-phase-3-sft.md §3.3).
#
#   nohup bash scripts/hf_autopush_when_ready.sh <ckpt.pt> <phase> <step> <global_step> <wandb_run> <config> > /dev/null 2>&1 &
set -u
CK=$1; PHASE=$2; STEP=$3; GSTEP=$4; RUN=$5; CFG=$6
cd /home/ubuntu/moe0.9b-a0.2b
TOK=$(python3 -c "print(round($GSTEP * 524288 / 1e9))")
BR="phase${PHASE//./}-step${STEP}-tokens${TOK}B"
LOG=/home/ubuntu/phase3_logs/autopush_${BR}.log
echo "[$(date -u)] waiting for $CK -> $BR" >> $LOG
until [ -f "$CK" ] && .venv/bin/python - "$CK" <<'EOF'
import sys, zipfile
sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)
EOF
do sleep 120; done
prev=-1; while true; do cur=$(stat -c %s "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 60; done
# one export at a time (each reads 7.5 GB)
while pgrep -f "[r]estore_checkpoint.py export" > /dev/null; do sleep 60; done
echo "[$(date -u)] exporting ($cur bytes)" >> $LOG
nice -n 10 .venv/bin/python scripts/restore_checkpoint.py export --pt "$CK" --out_dir /home/ubuntu/hf_export/$BR --verify >> $LOG 2>&1 || { echo "EXPORT_FAILED" >> $LOG; exit 1; }
.venv/bin/python scripts/hf_push_training_state.py --folder /home/ubuntu/hf_export/$BR --phase "$PHASE" --step "$STEP" \
  --global_step "$GSTEP" --wandb_run "$RUN" --config "$CFG" >> $LOG 2>&1
echo "PUSH_RC=$?" >> $LOG
