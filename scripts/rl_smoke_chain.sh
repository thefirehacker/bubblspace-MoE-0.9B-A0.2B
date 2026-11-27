#!/bin/bash
# fire01: prepare RL and run the 20-step GRPO smoke test (configs/rl_smoke.yaml) on Phase 3 run 01 (step 1500); it tests
# the pipeline, the RL start model is chosen after the run 02b evals. Never while the H100 trains. Does NOT start RL v1.
#   START_AFTER=evals (default): after run 02b AND its evals; uses the finished difficulty filter from the A10.
#   START_AFTER=train: right after run 02b training exits, alongside its evals (user's choice, 27 Nov): vLLM capped at
#     25% of the GPU and smaller training micro-batches so both fit in 80 GB; uses the tasks the A10 filter has kept so far.
#   A10=ubuntu@<a10-host> START_AFTER=train nohup bash scripts/rl_smoke_chain.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
REPO=/home/ubuntu/moe0.9b-a0.2b; LOG=/lambda/nfs/h100-f1/rl_smoke_chain.log
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
RUN01=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase03-sft/step_1500.pt
say () { echo "[$(date -u)] $*" >> $LOG; }
cd $REPO
START_AFTER=${START_AFTER:-evals}
if [ "$START_AFTER" = train ]; then
  say "waiting for run 02b training to exit (smoke will run alongside its evals)"
  until grep -qE "run 02b exited|ABORT" /lambda/nfs/h100-f1/phase3_run02b_full.log 2>/dev/null; do sleep 60; done
  grep -q "run 02b exited rc=0" /lambda/nfs/h100-f1/phase3_run02b_full.log || { say "ABORT: run 02b training failed"; exit 1; }
  while pgrep -f "scripts/train.p[y]" > /dev/null; do sleep 30; done
  EXTRA="vllm_gpu_mem=0.25 micro_batch_tokens=16384"
else
  say "waiting for run 02b + evals to finish"
  until grep -qE "RUN02B FULL DONE|ABORT" /lambda/nfs/h100-f1/phase3_run02b_full.log 2>/dev/null; do sleep 120; done
  while pgrep -f "scripts/train.p[y]|scripts/eval_chat.p[y]" > /dev/null; do sleep 60; done
  EXTRA=""
fi
say "training done; setting up the vLLM env"
bash scripts/setup_rl_env.sh >> $LOG 2>&1 || { say "ABORT: env setup failed"; exit 1; }
mkdir -p ~/rl_tasks
if [ "$START_AFTER" != train ]; then
  say "waiting for the difficulty filter on the A10"
  until $SSH $A10 'test -s ~/rl_tasks/filtered_phase3_step1500.summary.json' < /dev/null 2>/dev/null; do sleep 120; done
fi
rsync -a -e "$SSH" "$A10:/home/ubuntu/rl_tasks/" ~/rl_tasks/ >> $LOG 2>&1 || { say "ABORT: rsync tasks failed"; exit 1; }
TASKS=/home/ubuntu/rl_tasks/filtered_phase3_step1500.jsonl
# the A10 may still be writing the file: keep only complete JSON lines
python3 - $TASKS <<'PY'
import json, sys
p = sys.argv[1]; ok = []
for line in open(p):
    try:
        json.loads(line); ok.append(line)
    except ValueError:
        pass
open(p, "w").writelines(ok)
PY
say "smoke tasks: $(wc -l < $TASKS) kept by the difficulty filter so far"
.venv/bin/python scripts/make_vllm_model_dir.py --config configs/phase3.yaml --checkpoint $RUN01 \
  --tokenizer bubblspace-tokenizer --out ~/vllm_models/phase3-step1500 >> $LOG 2>&1 || { say "ABORT: model dir"; exit 1; }
say "starting the 20-step RL smoke test"
PATH=~/vllm_env/bin:$PATH python -u scripts/train_grpo.py configs/rl_smoke.yaml --set \
  init_checkpoint=$RUN01 vllm_model_dir=/home/ubuntu/vllm_models/phase3-step1500 \
  tasks=$TASKS model_config=configs/phase3.yaml $EXTRA \
  > /lambda/nfs/h100-f1/train_rl_smoke.log 2>&1 < /dev/null
say "RL smoke exited rc=$?"
say "RL SMOKE DONE"
