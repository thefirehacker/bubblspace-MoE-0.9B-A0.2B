#!/bin/bash
# fire01: add the 2.5a decay-stage mix (doc/plan-03-phase-2.5-knowledge-context.md §2.7) by an exact resume, then chain 2.5b.
# Same pattern as Phase 2's Stage 2B switch (doc/storyboard/04-Phase2-Trainig.md): --resume restores weights, both optimizers,
# both LR schedulers and the step counter; the same W&B run continues. The plateau mix is unchanged; only a second stage
# (steps 32,424 -> 38,147) is added, so switching at any checkpoint before 32,424 gives the same result.
#
#   A10=ubuntu@<a10-host> STEP=12000 nohup bash scripts/switch_25a_decay.sh > /dev/null 2>&1 &
#
# 1. checks: the edited configs/phase25a.yaml has the decay stage; configs/phase25a_before_decay.yaml exists (fallback)
# 2. waits until step_$STEP.pt is complete (size above minimum, stable for 2 min, valid zip)
# 3. stops launch_25.sh and handover_guard.sh FIRST (otherwise launch_25.sh would log "2.5a exited" and skip 2.5b),
#    then the training process; the 2.5a checkpoint relay and the A10 watchers keep running
# 4. relaunches 2.5a with --resume; after 15 min checks it resumed, else falls back to the launch config
# 5. when 2.5a ends cleanly with step_38147.pt, starts the A10 2.5b watcher + relay and trains 2.5b (as launch_25.sh did)
# Log lines keep launch_25.log's wording ("2.5a exited", "launching 2.5b", "2.5b exited") so existing monitors still work.
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
STEP=${STEP:-12000}
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
REPO=/home/ubuntu/moe0.9b-a0.2b
CKROOT=/lambda/nfs/h100-f1/checkpoints
CK=$CKROOT/moe0-9b-a0-2b-phase025a/step_$STEP.pt
FINAL_A=$CKROOT/moe0-9b-a0-2b-phase025a/step_38147.pt
MIN_BYTES=7400000000          # step_N.pt is 7,530,980,715 bytes; never wait for an exact size
RUN_ID=l2i2yoqf               # W&B run of 2.5a
LL=/lambda/nfs/h100-f1/launch_25.log
NEWLOG=/lambda/nfs/h100-f1/train_phase25a_decay.log
say () { echo "[$(date -u)] $*" >> $LL; }
cd $REPO && source .venv/bin/activate

# ---- 1. checks
grep -q "until_step: 32424" configs/phase25a.yaml && grep -q "cosmo: 0.04" configs/phase25a.yaml \
  && [ -f configs/phase25a_before_decay.yaml ] && [ -f /lambda/nfs/h100-f1/data/phase2/en_cosmo/meta.json ] \
  || { say "switch ABORT: decay config, fallback config or en_cosmo data missing"; exit 1; }
say "switch armed: waiting for $CK"

# ---- 2. wait for a complete checkpoint
until [ -f "$CK" ] && [ "$(stat -c %s "$CK")" -gt "$MIN_BYTES" ]; do sleep 30; done
stable=0; prev=-1
while [ $stable -lt 2 ]; do
  cur=$(stat -c %s "$CK"); if [ "$cur" = "$prev" ]; then stable=$((stable + 1)); else stable=0; fi
  prev=$cur; sleep 60
done
until python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$CK') else 1)"; do sleep 30; done
sleep 30
say "switch: checkpoint complete ($cur bytes)"

# ---- 3. stop the old chain, then training (PIDs checked by command line, never by a loose pattern)
pid_of () { pgrep -f "$1" | head -1; }
TRAIN=$(pid_of "scripts/train.py configs/phase25a.yaml --init_fro[m]")
LAUNCH=$(pid_of "bash scripts/launch_25.s[h]")
GUARD=$(pid_of "handover_guard.s[h]")
[ -n "$TRAIN" ] || { say "switch ABORT: 2.5a training process not found (crashed?). Not restarting automatically."; exit 1; }
for p in $LAUNCH $GUARD; do kill -TERM $p 2>/dev/null && say "switch: stopped pid $p ($(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-60))"; done
sleep 2
kill -TERM $TRAIN
for i in $(seq 1 60); do kill -0 $TRAIN 2>/dev/null || break; sleep 2; done
kill -0 $TRAIN 2>/dev/null && { say "switch: SIGTERM timeout, SIGKILL"; kill -KILL $TRAIN; sleep 5; }
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$used" -lt 2000 ] && break; sleep 2
done
say "switch: old training stopped (GPU ${used} MiB); resuming 2.5a from step $STEP with the decay-stage config"

# ---- 4. resume, with fallback
WANDB_RUN_ID=$RUN_ID WANDB_RESUME=allow python3 -u scripts/train.py configs/phase25a.yaml --resume "$CK" > $NEWLOG 2>&1 < /dev/null &
NEW=$!
sleep 900
if kill -0 $NEW 2>/dev/null && grep -q "resumed from" $NEWLOG; then
  say "switch OK: resumed 2.5a alive after 15 min (pid $NEW)"
else
  say "switch FAILED: resumed run not alive or did not resume; tail follows"; tail -c 2000 $NEWLOG >> $LL
  kill -0 $NEW 2>/dev/null && kill -TERM $NEW && sleep 20
  say "switch FALLBACK: resuming with the launch config (no decay stage)"
  NEWLOG=${NEWLOG%.log}_fallback.log
  WANDB_RUN_ID=$RUN_ID WANDB_RESUME=allow python3 -u scripts/train.py configs/phase25a_before_decay.yaml --resume "$CK" > $NEWLOG 2>&1 < /dev/null &
  NEW=$!
fi
wait $NEW
rc=$?
say "2.5a exited rc=$rc"
if [ $rc -ne 0 ] || [ ! -f $FINAL_A ]; then say "2.5a did not finish cleanly; NOT chaining 2.5b"; exit 1; fi

# ---- 5. Phase 2.5b (same as launch_25.sh step 4)
$SSH -n -f $A10 "cd moe0.9b-a0.2b && nohup bash scripts/eval_watch_25b.sh > /dev/null 2>&1 < /dev/null &"
A10=$A10 nohup bash scripts/relay_ckpts_25b.sh > /dev/null 2>&1 < /dev/null &
say "launching 2.5b from $FINAL_A"
python3 scripts/train.py configs/phase25b.yaml --init_from $FINAL_A > /lambda/nfs/h100-f1/train_phase25b.log 2>&1 < /dev/null
say "2.5b exited rc=$?"
