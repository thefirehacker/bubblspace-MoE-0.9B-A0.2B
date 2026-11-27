#!/bin/bash
# fire01: once an RL run's chain is done, send its LAST TWO checkpoints and its rollouts / test answers / logs to the A10
# (scripts/rl_push_last2.sh exports and pushes them). Works for runs that stop on time (step numbers not known ahead).
#   A10=... RUN=Phase4_rl_v1b DONE_PATTERN="RL V1B CHAIN DONE" CHAIN_LOG=/lambda/nfs/h100-f1/rl_v1b_chain.log nohup bash scripts/rl_relay_last2.sh &
set -u
A10=${A10:?}; RUN=${RUN:?}; DONE_PATTERN=${DONE_PATTERN:?}; CHAIN_LOG=${CHAIN_LOG:?}
cd /home/ubuntu/moe0.9b-a0.2b
OUT=/lambda/nfs/h100-f1/checkpoints/$RUN
DST=/lambda/nfs/us-west-1-eval-inference/$RUN
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
until grep -q "$DONE_PATTERN" $CHAIN_LOG 2>/dev/null; do sleep 120; done
STEPS=$(ls $OUT | grep -oE "step_[0-9]+" | sed s/step_// | sort -n | tail -2 | tr "\n" " ")
A10=$A10 RUN=$RUN DST_DIR=$RUN STEPS="$STEPS" bash scripts/relay_ckpts.sh
grep -oE "runs/[a-z0-9]+" /lambda/nfs/h100-f1/train_$(echo $RUN | tr "[:upper:]" "[:lower:]" | sed "s/phase4_//").log 2>/dev/null | head -1 | cut -d/ -f2 > $OUT/wandb_run.txt
echo "$STEPS" > $OUT/STEPS.txt
$SSH $A10 "mkdir -p $DST/meta"
rsync -a -e "$SSH" --exclude "*.pt" $OUT/ "$A10:$DST/meta/" && $SSH $A10 "touch $DST/meta/META_DONE"
