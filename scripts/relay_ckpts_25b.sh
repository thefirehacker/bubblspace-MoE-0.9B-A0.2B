#!/bin/bash
# fire01: relay Phase 2.5b checkpoints (1000, 2000, final 2861) to the A10 for evals.
# Same mechanics as relay_ckpts_25a.sh (bandwidth-limited rsync, waits until the file stops growing).
# Usage (on fire01): A10=ubuntu@<a10-host> nohup bash scripts/relay_ckpts_25b.sh > /dev/null 2>&1 &
CKD=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase025b
A10=${A10:?set A10=ubuntu@<a10-host>}
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
DST=/lambda/nfs/us-west-1-eval-inference/phase25b
LOG=/lambda/nfs/h100-f1/relay_ckpts_25b.log
$SSH $A10 "mkdir -p $DST"
for s in 1000 2000 2861; do
  CK=$CKD/step_$s.pt
  echo "[$(date -u)] waiting for step_$s" >> $LOG
  until [ -f "$CK" ]; do sleep 60; done
  prev=-1; while true; do cur=$(stat -c %s "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 60; done
  rsync -a --bwlimit=40000 -e "$SSH" "$CK" $A10:$DST/step_$s.pt \
    && echo "[$(date -u)] relayed step_$s ($cur bytes)" >> $LOG \
    || echo "[$(date -u)] rsync FAILED for step_$s" >> $LOG
done
echo "[$(date -u)] relay 25b finished" >> $LOG
