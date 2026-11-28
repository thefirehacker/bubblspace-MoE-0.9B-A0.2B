#!/bin/bash
# fire01: Phase 3 run 02b full SFT (1,500 steps from the Phase 2.5b final; configs/phase3_run02b.yaml), started on the
# user's go after the run 02b smoke test (doc/issue-02-phase3-over-refusal.md §6). Afterwards, on the final checkpoint:
# the three-way refusal check, the full chat evals (scripts/eval_chat.py: refusal, HumanEval, MBPP, GSM8K, IFEval, BFCL)
# and the 23-conversation chat test, all comparable with run 01. Checkpoints go to the A10 via scripts/relay_ckpts.sh.
#   A10=ubuntu@<a10-host> nohup bash scripts/phase3_run02b_full.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
REPO=/home/ubuntu/moe0.9b-a0.2b; EV=/home/ubuntu/sfteval
FINAL_B=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase025b/step_2861.pt
CK=/lambda/nfs/h100-f1/checkpoints/Phase3_run02b
LOG=/lambda/nfs/h100-f1/phase3_run02b_full.log
say () { echo "[$(date -u)] $*" >> $LOG; }
cd $REPO && source .venv/bin/activate
while pgrep -f "scripts/train.p[y]|scripts/eval_chat.p[y]|phase3_chat_smok[e]" > /dev/null; do sleep 60; done
say "starting run 02b: 1,500 steps from $FINAL_B"
A10=$A10 RUN=Phase3_run02b DST_DIR=Phase3_run02b STEPS="1000 1500" setsid nohup bash scripts/relay_ckpts.sh > /dev/null 2>&1 < /dev/null &
python3 -u scripts/train.py configs/phase3_run02b.yaml --init_from $FINAL_B > /lambda/nfs/h100-f1/train_phase3_run02b.log 2>&1 < /dev/null
rc=$?; say "run 02b exited rc=$rc"
[ $rc = 0 ] && [ -f $CK/step_1500.pt ] || { say "ABORT: training failed, no evals"; exit 1; }
cd $EV
python3 -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $CK/step_1500.pt --tasks refusal \
  --out benchmark/phase3/refusal3_Phase3_run02b_step1500.json >> $LOG 2>&1
say "refusal check done; starting the full chat evals"
python3 -u scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint $CK/step_1500.pt \
  --out benchmark/phase3/chat_smoke_Phase3_run02b_step1500.md > /dev/null 2>&1
python3 -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $CK/step_1500.pt --tasks humaneval,mbpp,gsm8k,ifeval,bfcl \
  --out benchmark/phase3/chat_Phase3_run02b_step1500.json > benchmark/phase3/chat_Phase3_run02b_step1500.log 2>&1
say "chat evals rc=$?"
say "RUN02B FULL DONE"
