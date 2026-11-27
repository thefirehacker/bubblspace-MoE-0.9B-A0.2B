#!/bin/bash
# fire01: Phase 3 run 02b smoke test (27 Nov). Waits for the run 02b data on the A10 (scripts/phase3_run02b_data.sh),
# pulls the two new pools, verifies them against meta.json, waits for the GPU, runs the 200-step smoke test from the
# Phase 2.5b final (configs/phase3_run02b_smoke.yaml), then the over-refusal check + the 23-conversation chat test on the
# smoke checkpoint, and the over-refusal check on run 01's final for comparison. Does NOT start the full run.
#   A10=ubuntu@<a10-host> nohup bash scripts/phase3_run02_smoke.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
REPO=/home/ubuntu/moe0.9b-a0.2b; EV=/home/ubuntu/sfteval
FINAL_B=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase025b/step_2861.pt
RUN01=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase03-sft/step_1500.pt
SMOKE=/lambda/nfs/h100-f1/checkpoints/Phase3_run02b_smoke/step_200.pt
DATA=/lambda/nfs/h100-f1/data/phase3
LOG=/lambda/nfs/h100-f1/phase3_run02b_smoke.log
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
say () { echo "[$(date -u)] $*" >> $LOG; }
cd $REPO && source .venv/bin/activate

say "waiting for the run 02 data on the A10"
until $SSH $A10 'grep -q RUN02B_DATA_DONE ~/phase3_logs/run02b_data.log' < /dev/null 2>/dev/null; do sleep 60; done
$SSH $A10 'grep -E "rc=" ~/phase3_logs/run02b_data.log' < /dev/null >> $LOG 2>&1
$SSH $A10 'grep -E "rc=[^0]" ~/phase3_logs/run02b_data.log' < /dev/null > /dev/null 2>&1 && { say "ABORT: a data step failed"; exit 1; }
for p in tulu3_run02b own_realtime; do
  rsync -a -e "$SSH" "$A10:/home/ubuntu/phase3_decon/$p/" $DATA/$p/ >> $LOG 2>&1 || { say "ABORT: rsync $p failed"; exit 1; }
done
python3 - $DATA/tulu3_run02b $DATA/own_realtime >> $LOG 2>&1 <<'PY' || { echo "ABORT: shard check failed" >> /lambda/nfs/h100-f1/phase3_run02b_smoke.log; exit 1; }
import glob, json, os, sys
bad = 0
for d in sys.argv[1:]:
    m = json.load(open(d + "/meta.json"))
    sh = sorted(glob.glob(d + "/shard_*.bin")); mk = sorted(glob.glob(d + "/mask_*.bin"))
    tok = sum(os.path.getsize(f) for f in sh) // 2; msk = sum(os.path.getsize(f) for f in mk)
    ok = tok == msk == m["total_tokens_incl_padding"] and len(sh) == len(mk) == m["n_shards"]
    bad += not ok
    print(("OK " if ok else "BAD"), os.path.basename(d), tok, msk, m["total_tokens_incl_padding"], "target tokens after decon:", m.get("target_tokens_after_decon"))
sys.exit(1 if bad else 0)
PY
say "data verified; waiting for the GPU"
while pgrep -f "scripts/eval_chat.p[y]|scripts/train.p[y]" > /dev/null; do sleep 60; done
say "starting the 200-step smoke test from $FINAL_B"
python3 -u scripts/train.py configs/phase3_run02b_smoke.yaml --init_from $FINAL_B > /lambda/nfs/h100-f1/train_phase3_run02b_smoke.log 2>&1 < /dev/null
rc=$?; say "smoke run exited rc=$rc"
[ $rc = 0 ] && [ -f $SMOKE ] || { say "ABORT: smoke failed"; exit 1; }
cd $EV
# same three-way refusal check (harmless / harmful / live data) on run 02b smoke, run 02 smoke and run 01 final
SMOKE02=/lambda/nfs/h100-f1/checkpoints/Phase3_run02_smoke/step_200.pt
python3 -u scripts/eval_chat.py --config configs/phase3_run02b.yaml --checkpoint $SMOKE --tasks refusal --out benchmark/phase3/refusal3_Phase3_run02b_smoke_step200.json >> $LOG 2>&1
python3 -u scripts/eval_chat.py --config configs/phase3_run02.yaml --checkpoint $SMOKE02 --tasks refusal --out benchmark/phase3/refusal3_Phase3_run02_smoke_step200.json >> $LOG 2>&1
python3 -u scripts/eval_chat.py --config configs/phase3.yaml --checkpoint $RUN01 --tasks refusal --out benchmark/phase3/refusal3_phase3_step1500.json >> $LOG 2>&1
python3 -u scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint $SMOKE --out benchmark/phase3/chat_smoke_Phase3_run02b_smoke_step200.md > /dev/null 2>&1
say "SMOKE02B DONE"
