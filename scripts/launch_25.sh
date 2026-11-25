#!/bin/bash
# fire01: launch Phase 2.5a and chain Phase 2.5b, with checkpoint relays and the A10 eval watchers.
# Run ONLY after the explicit go for real training (doc/plan-03-phase-2.5-knowledge-context.md).
#
#   A10=ubuntu@<a10-host> nohup bash scripts/launch_25.sh > /dev/null 2>&1 &
#
# Steps:
#   1. preflight: GPU idle, base checkpoint present, every 2.5a / 2.5b pool path exists
#   2. start the A10 watchers (they wait for checkpoints) and the fire01 relays
#   3. train 2.5a from Phase 2 final (step_22888.pt): ≈65 h, 38,147 steps, 20B tokens
#   4. when 2.5a exits cleanly with step_38147.pt written, train 2.5b from it: ≈6 h, 2,861 steps at 16k
#      (set CHAIN_25B=0 to stop after 2.5a, e.g. if you want to review 2.5a evals first)
# Crash recovery is manual: python3 scripts/train.py <config> --resume <last step_N.pt>
set -uo pipefail
A10=${A10:?set A10=ubuntu@<a10-host>}
CHAIN_25B=${CHAIN_25B:-1}
SSH="ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes"
REPO=/home/ubuntu/moe0.9b-a0.2b
CKROOT=/lambda/nfs/h100-f1/checkpoints
BASE=$CKROOT/moe0-9b-a0-2b-phase02/step_22888.pt
LOG=/lambda/nfs/h100-f1/launch_25.log
cd $REPO && source .venv/bin/activate
say () { echo "[$(date -u)] $*" >> $LOG; }

# ---- 1. preflight
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
[ "$used" -lt 1000 ] || { say "ABORT: GPU busy (${used} MiB used)"; exit 1; }
[ -f $BASE ] || { say "ABORT: missing $BASE"; exit 1; }
for cfg in configs/phase25a.yaml configs/phase25b.yaml; do
  python3 - "$cfg" <<'EOF' >> $LOG 2>&1 || { say "ABORT: pool check failed for $cfg"; exit 1; }
import sys, yaml, pathlib
c = yaml.safe_load(open(sys.argv[1]))
missing = [n for n, p in c["phase2"]["pools"].items() if not list(pathlib.Path(p["path"]).glob("**/shard_*.bin"))]
groups = c["phase2"]["stages"][-1]["groups"]
assert abs(sum(groups.values()) - 1) < 1e-6, f"groups sum {sum(groups.values())}"
print(sys.argv[1], "pools ok" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
EOF
done
$SSH $A10 "test -f moe0.9b-a0.2b/benchmark/data/long-context/long_docs_heldout.npz" || { say "ABORT: A10 unreachable or long-context docs missing"; exit 1; }
say "preflight ok"

# ---- 2. watchers and relays
$SSH -n -f $A10 "cd moe0.9b-a0.2b && nohup bash scripts/eval_watch_25a.sh > /dev/null 2>&1 < /dev/null &"
A10=$A10 nohup bash scripts/relay_ckpts_25a.sh > /dev/null 2>&1 < /dev/null &
say "A10 watcher 25a + relay 25a started"

# ---- 3. Phase 2.5a
say "launching 2.5a from $BASE"
python3 scripts/train.py configs/phase25a.yaml --init_from $BASE > /lambda/nfs/h100-f1/train_phase25a.log 2>&1 < /dev/null
rc=$?
FINAL_A=$CKROOT/moe0-9b-a0-2b-phase025a/step_38147.pt
say "2.5a exited rc=$rc"
if [ $rc -ne 0 ] || [ ! -f $FINAL_A ]; then say "2.5a did not finish cleanly; NOT chaining 2.5b"; exit 1; fi
[ "$CHAIN_25B" = 1 ] || { say "CHAIN_25B=0: stopping after 2.5a"; exit 0; }

# ---- 4. Phase 2.5b
$SSH -n -f $A10 "cd moe0.9b-a0.2b && nohup bash scripts/eval_watch_25b.sh > /dev/null 2>&1 < /dev/null &"
A10=$A10 nohup bash scripts/relay_ckpts_25b.sh > /dev/null 2>&1 < /dev/null &
say "launching 2.5b from $FINAL_A"
python3 scripts/train.py configs/phase25b.yaml --init_from $FINAL_A > /lambda/nfs/h100-f1/train_phase25b.log 2>&1 < /dev/null
say "2.5b exited rc=$?"
