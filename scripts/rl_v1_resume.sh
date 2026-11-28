#!/bin/bash
# fire01: restart RL v1 after the first attempt ran out of GPU memory (micro_batch_tokens 32768 -> 16384), reusing the
# finished difficulty filter (~/rl_tasks/filtered_v3_run02b.jsonl); then the quick check, as in scripts/rl_v1_chain.sh.
set -u
cd /home/ubuntu/moe0.9b-a0.2b
LOG=/lambda/nfs/h100-f1/rl_v1_chain.log
OUT=/lambda/nfs/h100-f1/checkpoints/Phase4_rl_v1
say () { echo "[$(date -u)] $*" >> $LOG; }
rm -rf $OUT
say "2/3 RL v1 (restart, micro_batch_tokens 16384): 500 steps on $(wc -l < ~/rl_tasks/filtered_v3_run02b.jsonl) filtered tasks"
PATH=~/vllm_env/bin:$PATH python -u scripts/train_grpo.py configs/rl_v1.yaml > /lambda/nfs/h100-f1/train_rl_v1.log 2>&1 < /dev/null
rc=$?; say "RL v1 exited rc=$rc"
[ $rc = 0 ] && [ -f $OUT/step_500.pt ] || { say "ABORT: RL v1 did not finish"; exit 1; }
say "3/3 quick check on step 500"
cd /home/ubuntu/sfteval; P=/home/ubuntu/moe0.9b-a0.2b/.venv/bin/python
$P -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_500.pt --tasks refusal \
  --out benchmark/phase3/refusal3_Phase4_rl_v1_step500.json >> $LOG 2>&1
$P -u scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint $OUT/step_500.pt \
  --out benchmark/phase3/chat_smoke_Phase4_rl_v1_step500.md > /dev/null 2>&1
say "RL V1 CHAIN DONE"
