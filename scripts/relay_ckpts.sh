#!/bin/bash
# fire01: relay the given checkpoints of a run to the A10 (bandwidth-limited so training's NFS reads are unaffected).
# rsync writes to a temp name and renames on completion, so the A10 only ever sees complete files.
#
#   A10=ubuntu@<a10-host> RUN=moe0-9b-a0-2b-phase03-sft DST_DIR=phase3 STEPS="1000 1500" nohup bash scripts/relay_ckpts.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}; RUN=${RUN:?set RUN}; DST_DIR=${DST_DIR:?set DST_DIR}; STEPS=${STEPS:?set STEPS}
CKD=/lambda/nfs/h100-f1/checkpoints/$RUN
DST=/lambda/nfs/us-west-1-eval-inference/$DST_DIR
LOG=/lambda/nfs/h100-f1/relay_${DST_DIR}.log
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
$SSH $A10 "mkdir -p $DST"
for s in $STEPS; do
  CK=$CKD/step_$s.pt
  echo "[$(date -u)] waiting for step_$s" >> $LOG
  until [ -f "$CK" ]; do sleep 60; done
  prev=-1; while true; do cur=$(stat -c %s "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 60; done
  rsync -a --bwlimit=40000 -e "$SSH" "$CK" "$A10:$DST/step_$s.pt" \
    && echo "[$(date -u)] relayed step_$s ($cur bytes)" >> $LOG || echo "[$(date -u)] rsync FAILED for step_$s" >> $LOG
done
echo "[$(date -u)] relay $DST_DIR finished" >> $LOG
