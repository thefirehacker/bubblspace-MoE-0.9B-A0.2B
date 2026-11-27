#!/bin/bash
# fire01: after Phase 3 run 02b and its evals finish (never while the H100 trains), prepare RL and run the 20-step
# GRPO smoke test (configs/rl_smoke.yaml). The smoke uses Phase 3 run 01 (step 1500), whose difficulty filter ran on
# the A10; it tests the pipeline, the RL start model is chosen after the run 02b evals. Does NOT start RL v1.
#   A10=ubuntu@<a10-host> nohup bash scripts/rl_smoke_chain.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
REPO=/home/ubuntu/moe0.9b-a0.2b; LOG=/lambda/nfs/h100-f1/rl_smoke_chain.log
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
RUN01=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase03-sft/step_1500.pt
say () { echo "[$(date -u)] $*" >> $LOG; }
cd $REPO
say "waiting for run 02b + evals to finish"
until grep -qE "RUN02B FULL DONE|ABORT" /lambda/nfs/h100-f1/phase3_run02b_full.log 2>/dev/null; do sleep 120; done
while pgrep -f "scripts/train.p[y]|scripts/eval_chat.p[y]" > /dev/null; do sleep 60; done
say "H100 free; setting up the vLLM env"
bash scripts/setup_rl_env.sh >> $LOG 2>&1 || { say "ABORT: env setup failed"; exit 1; }
say "waiting for the difficulty filter on the A10"
until $SSH $A10 'test -s ~/rl_tasks/filtered_phase3_step1500.summary.json' < /dev/null 2>/dev/null; do sleep 120; done
mkdir -p ~/rl_tasks
rsync -a -e "$SSH" "$A10:/home/ubuntu/rl_tasks/" ~/rl_tasks/ >> $LOG 2>&1 || { say "ABORT: rsync tasks failed"; exit 1; }
cat ~/rl_tasks/filtered_phase3_step1500.summary.json >> $LOG
.venv/bin/python scripts/make_vllm_model_dir.py --config configs/phase3.yaml --checkpoint $RUN01 \
  --tokenizer bubblspace-tokenizer --out ~/vllm_models/phase3-step1500 >> $LOG 2>&1 || { say "ABORT: model dir"; exit 1; }
say "starting the 20-step RL smoke test"
PATH=~/vllm_env/bin:$PATH python -u scripts/train_grpo.py configs/rl_smoke.yaml --set \
  init_checkpoint=$RUN01 vllm_model_dir=/home/ubuntu/vllm_models/phase3-step1500 \
  tasks=/home/ubuntu/rl_tasks/filtered_phase3_step1500.jsonl model_config=configs/phase3.yaml \
  > /lambda/nfs/h100-f1/train_rl_smoke.log 2>&1 < /dev/null
say "RL smoke exited rc=$?"
say "RL SMOKE DONE"
