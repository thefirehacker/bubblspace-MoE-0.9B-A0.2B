#!/bin/bash
# fire01: RL v1c, continuing RL v1b until 12:00 IST (user's request, 27 Nov): same steps, from v1b's last checkpoint:
#   1. refresh the difficulty filter of task set v3 with the RL v1 model (step 500): questions it now always gets right
#      teach nothing (W&B: zero_adv_groups rose 0.3 -> ~0.55 during v1)
#   2. continue RL from RL v1 step 500 with the KL still anchored to the SFT model (run 02b), configs/rl_v1.yaml settings,
#      tests every 50, checkpoints every 500 + one at the stop (user: at most 10 RL checkpoints in total), HARD STOP at 06:00 IST (00:30 UTC Nov 28): final test + checkpoint, then exit; RL v1c continues (rl_v1c_chain.sh)
#   3. quick check on the final checkpoint (refusal check + 23-conversation chat test)
#   A10=ubuntu@<a10-host> nohup bash scripts/rl_v1b_chain.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
cd /home/ubuntu/moe0.9b-a0.2b
LOG=/lambda/nfs/h100-f1/rl_v1c_chain.log
PREV=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_v1b
SFT=/lambda/nfs/h100-f1/checkpoints/Phase3_run02b/step_1500.pt
OUT=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_v1c
VDIR=/home/ubuntu/vllm_models/Phase4_rl_v1b-last
STOP=$(date -u -d "2026-11-28 06:30:00" +%s)   # 12:00 IST
say () { echo "[$(date -u)] $*" >> $LOG; }
say "waiting for RL v1b + its quick check"
until tail -1 /lambda/nfs/h100-f1/rl_v1b_chain.log 2>/dev/null | grep -qE "RL V1B CHAIN DONE|ABORT"; do sleep 60; done
tail -1 /lambda/nfs/h100-f1/rl_v1b_chain.log | grep -q "RL V1B CHAIN DONE" || { say "ABORT: RL v1b did not finish"; exit 1; }
V1=$PREV/step_$(ls $PREV | grep -oE "step_[0-9]+" | sed s/step_// | sort -n | tail -1).pt
while pgrep -f "scripts/train_grp[o]|scripts/eval_chat.p[y]|rl_filte[r]|phase3_chat_smok[e]" > /dev/null; do sleep 30; done
[ $(date +%s) -lt $((STOP - 3600)) ] || { say "ABORT: less than an hour left before the stop time"; exit 1; }
.venv/bin/python scripts/make_vllm_model_dir.py --config configs/phase3_run02b.yaml --checkpoint $V1 \
  --tokenizer bubblspace-tokenizer --out $VDIR >> $LOG 2>&1 || { say "ABORT: model dir"; exit 1; }
say "1/3 refreshing the difficulty filter with $V1"
PATH=~/vllm_env/bin:$PATH python -u scripts/rl_filter.py --model_dir $VDIR --tasks ~/rl_tasks/tasks_v3.jsonl \
  --out ~/rl_tasks/filtered_v3_rlv1b.jsonl --limit_per_area 2500 --gpu_mem 0.6 --sandbox_workers 32 \
  > /lambda/nfs/h100-f1/rl_filter_v3_rlv1.log 2>&1 < /dev/null || { say "ABORT: filter failed"; exit 1; }
cat ~/rl_tasks/filtered_v3_rlv1b.summary.json >> $LOG
say "2/3 RL v1c from $V1 (KL reference $SFT) on $(wc -l < ~/rl_tasks/filtered_v3_rlv1b.jsonl) tasks, stop at 06:30 UTC"
PATH=~/vllm_env/bin:$PATH python -u scripts/train_grpo.py configs/rl_v1.yaml --set run_name=Phase4_rl_v1c out_dir=$OUT \
  init_checkpoint=$V1 ref_checkpoint=$SFT vllm_model_dir=$VDIR tasks=/home/ubuntu/rl_tasks/filtered_v3_rlv1b.jsonl \
  steps=4000 eval_every=50 save_every=500 stop_at_epoch=$STOP \
  > /lambda/nfs/h100-f1/train_rl_v1c.log 2>&1 < /dev/null
rc=$?; say "RL v1c exited rc=$rc"
LAST=$(ls $OUT | grep -oE "step_[0-9]+" | sed s/step_// | sort -n | tail -1)
[ $rc = 0 ] && [ -n "$LAST" ] || { say "ABORT: RL v1c did not finish"; exit 1; }
say "3/3 quick check on step $LAST"
cd /home/ubuntu/sfteval; P=/home/ubuntu/moe0.9b-a0.2b/.venv/bin/python
$P -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_$LAST.pt --tasks refusal \
  --out benchmark/phase3/refusal3_Phase4_rl_v1c_step$LAST.json >> $LOG 2>&1
$P -u scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_$LAST.pt \
  --out benchmark/phase3/chat_smoke_Phase4_rl_v1c_step$LAST.md > /dev/null 2>&1
say "RL V1C CHAIN DONE (last step $LAST)"
