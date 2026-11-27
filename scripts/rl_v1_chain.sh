#!/bin/bash
# fire01: Phase 4 RL v1 (user's go, 27 Nov): start model Phase 3 run 02b step 1500, task set v3.
#   1. difficulty filter of task set v3 for run 02b (H100, vLLM; up to 2,500 tasks per subject, 32 sandbox workers)
#   2. RL v1: configs/rl_v1.yaml (500 steps, tests every 25, checkpoints every 100, W&B Phase4_rl_v1)
#   3. quick check on the final checkpoint: refusal check (harmless / harmful / live data) + the 23-conversation chat test
#   A10=ubuntu@<a10-host> nohup bash scripts/rl_v1_chain.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
cd /home/ubuntu/moe0.9b-a0.2b
LOG=/lambda/nfs/h100-f1/rl_v1_chain.log
OUT=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_v1
VDIR=/home/ubuntu/vllm_models/Phase3_run02b-step1500
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
say () { echo "[$(date -u)] $*" >> $LOG; }
while pgrep -f "scripts/train.p[y]|scripts/train_grp[o]|scripts/eval_chat.p[y]|rl_filte[r]" > /dev/null; do sleep 30; done
mkdir -p ~/rl_tasks
rsync -a -e "$SSH" "$A10:/home/ubuntu/rl_tasks/tasks_v3.jsonl" ~/rl_tasks/ >> $LOG 2>&1 || { say "ABORT: rsync tasks"; exit 1; }
say "1/3 difficulty filter: task set v3 ($(wc -l < ~/rl_tasks/tasks_v3.jsonl) tasks) for run 02b"
PATH=~/vllm_env/bin:$PATH python -u scripts/rl_filter.py --model_dir $VDIR --tasks ~/rl_tasks/tasks_v3.jsonl \
  --out ~/rl_tasks/filtered_v3_run02b.jsonl --limit_per_area 2500 --gpu_mem 0.6 --sandbox_workers 32 \
  > /lambda/nfs/h100-f1/rl_filter_v3_run02b.log 2>&1 < /dev/null || { say "ABORT: filter failed"; exit 1; }
cat ~/rl_tasks/filtered_v3_run02b.summary.json >> $LOG
say "2/3 RL v1: 500 steps on $(wc -l < ~/rl_tasks/filtered_v3_run02b.jsonl) filtered tasks"
PATH=~/vllm_env/bin:$PATH python -u scripts/train_grpo.py configs/rl_v1.yaml \
  > /lambda/nfs/h100-f1/train_rl_v1.log 2>&1 < /dev/null
rc=$?; say "RL v1 exited rc=$rc"
[ $rc = 0 ] && [ -f $OUT/step_500.pt ] || { say "ABORT: RL v1 did not finish"; exit 1; }
say "3/3 quick check on step 500"
cd /home/ubuntu/sfteval; P=/home/ubuntu/moe0.9b-a0.2b/.venv/bin/python
$P -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_500.pt --tasks refusal \
  --out benchmark/phase3/refusal3_Phase4_rl_v1_step500.json >> $LOG 2>&1
$P -u scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_500.pt \
  --out benchmark/phase3/chat_smoke_Phase4_rl_v1_step500.md > /dev/null 2>&1
say "RL V1 CHAIN DONE"
