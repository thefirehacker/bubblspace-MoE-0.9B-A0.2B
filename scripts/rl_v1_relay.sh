#!/bin/bash
# fire01: after RL v1 finishes, send its last two checkpoints (steps 400, 500) and its rollouts / test answers / logs to the
# A10, where scripts/rl_v1_push.sh exports and pushes them to Hugging Face (standing rule: last two checkpoints per stage;
# RL with detailed rollouts).   A10=ubuntu@<a10-host> nohup bash scripts/rl_v1_relay.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
cd /home/ubuntu/moe0.9b-a0.2b
OUT=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_v1
DST=/lambda/nfs/us-west-1-eval-inference/Phase4_rl_v1
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
until grep -qE "RL v1 exited|ABORT" /lambda/nfs/h100-f1/rl_v1_chain.log 2>/dev/null; do sleep 120; done
grep -q "RL v1 exited rc=0" /lambda/nfs/h100-f1/rl_v1_chain.log || exit 1
A10=$A10 RUN=Phase4_rl_v1 DST_DIR=Phase4_rl_v1 STEPS="400 500" bash scripts/relay_ckpts.sh
grep -oE "runs/[a-z0-9]+" /lambda/nfs/h100-f1/train_rl_v1.log | head -1 | cut -d/ -f2 > $OUT/wandb_run.txt
$SSH $A10 "mkdir -p $DST/meta"
rsync -a -e "$SSH" --exclude "*.pt" $OUT/ "$A10:$DST/meta/" && $SSH $A10 "touch $DST/meta/META_DONE"
