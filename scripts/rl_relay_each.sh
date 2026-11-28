#!/bin/bash
# fire01: relay EVERY checkpoint of an RL run to the A10 as soon as it is saved (user, 27 Nov: every 100-step RL
# checkpoint goes to both HF repos), with a snapshot of the rollouts / test answers / logs at that moment.
# scripts/rl_push_each.sh on the A10 exports and pushes each one. Exits when the run's chain is done and all are sent.
#   A10=... RUN=Phase4_rl_v1 DONE_PATTERN="RL V1 CHAIN DONE|ABORT" CHAIN_LOG=... [ONLY="100 300 500"] nohup bash scripts/rl_relay_each.sh &
# ONLY: relay just these steps (user, 27 Nov: at most 10 RL checkpoints on HF in total); empty = every checkpoint
set -u
A10=${A10:?}; RUN=${RUN:?}; DONE_PATTERN=${DONE_PATTERN:?}; CHAIN_LOG=${CHAIN_LOG:?}; ONLY=${ONLY:-}
OUT=/lambda/nfs/h100-f1/checkpoints/$RUN
DST=/lambda/nfs/us-west-1-eval-inference/$RUN
LOG=/lambda/nfs/h100-f1/relay_each_$RUN.log
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
$SSH $A10 "mkdir -p $DST" < /dev/null
WRUN_FILE=$OUT/wandb_run.txt
while true; do
  # the chain is done only if its LAST line says so (the log can hold an ABORT from an earlier, restarted attempt)
  finished=0; tail -1 $CHAIN_LOG 2>/dev/null | grep -qE "$DONE_PATTERN" && finished=1
  for CK in $(ls $OUT/step_*.pt 2>/dev/null | sort -V); do
    s=$(basename $CK .pt | sed s/step_//)
    [ -f $OUT/.relayed_$s ] && continue
    if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $s "; then continue; fi
    prev=-1; while true; do cur=$(stat -c %s "$CK"); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 30; done
    tr="/lambda/nfs/h100-f1/train_$(echo $RUN | tr '[:upper:]' '[:lower:]' | sed 's/phase4_//').log"
    grep -oE "runs/[a-z0-9]+" $tr 2>/dev/null | head -1 | cut -d/ -f2 > $WRUN_FILE
    rsync -a --bwlimit=60000 -e "$SSH" "$CK" "$A10:$DST/step_$s.pt" >> $LOG 2>&1 || { echo "[$(date -u)] rsync failed step $s" >> $LOG; continue; }
    rsync -a -e "$SSH" --exclude "*.pt" --exclude ".relayed_*" $OUT/ "$A10:$DST/meta_step$s/" >> $LOG 2>&1
    $SSH $A10 "touch $DST/READY_step$s" < /dev/null
    touch $OUT/.relayed_$s
    echo "[$(date -u)] relayed step $s ($cur bytes)" >> $LOG
  done
  [ $finished = 1 ] && { echo "[$(date -u)] chain done; all checkpoints relayed" >> $LOG; $SSH $A10 "touch $DST/ALL_RELAYED" < /dev/null; exit 0; }
  sleep 60
done
