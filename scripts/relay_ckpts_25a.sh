#!/bin/bash
# fire01: relay Phase 2.5a checkpoints to the A10 for evals (every 2,000 steps + final 29,723; early decay, was 38,147).
# Bandwidth-throttled so training's NFS reads are unaffected. Never touches training.
# rsync writes to a temp name and renames on completion, so the A10 only ever sees complete files.
# Usage (on fire01): nohup bash scripts/relay_ckpts_25a.sh > /dev/null 2>&1 &
CKD=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase025a
A10=${A10:?set A10=ubuntu@<a10-host>}
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
DST=/lambda/nfs/us-west-1-eval-inference/phase25a
LOG=/lambda/nfs/h100-f1/relay_ckpts_25a.log
$SSH $A10 "mkdir -p $DST"
# early decay (26 Nov): final is 29,723. Override to restart mid-run: STEPS="24000 26000 28000 29723"
STEPS=${STEPS:-"$(seq 2000 2000 28000) 29723"}
for s in $STEPS; do
  CK=$CKD/step_$s.pt
  echo "[$(date -u)] waiting for step_$s" >> $LOG
  until [ -f "$CK" ]; do sleep 60; done
  prev=-1; while true; do cur=$(stat -c %s "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 60; done
  rsync -a --bwlimit=40000 -e "$SSH" "$CK" $A10:$DST/step_$s.pt \
    && echo "[$(date -u)] relayed step_$s ($cur bytes)" >> $LOG \
    || echo "[$(date -u)] rsync FAILED for step_$s" >> $LOG
done
echo "[$(date -u)] relay 25a finished" >> $LOG
