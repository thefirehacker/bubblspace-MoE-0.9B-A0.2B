#!/bin/bash
# fire01: once Phase 2.5b has finished cleanly, (1) pull the decontaminated SFT shards from the A10, (2) check them
# against their meta.json, (3) run the 200-step SFT smoke test (configs/phase3_smoke.yaml), (4) write a few chat-format
# samples from the smoke checkpoint. It does NOT start the full SFT run: that waits for the user's go after the report.
#
#   A10=ubuntu@<a10-host> nohup bash scripts/phase3_smoke_after_25b.sh > /dev/null 2>&1 &
set -u
A10=${A10:?set A10=ubuntu@<a10-host>}
REPO=/home/ubuntu/moe0.9b-a0.2b
LL=/lambda/nfs/h100-f1/launch_25.log
FINAL_B=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase025b/step_2861.pt
DATA=/lambda/nfs/h100-f1/data/phase3
LOG=/lambda/nfs/h100-f1/phase3_smoke.log
say () { echo "[$(date -u)] $*" >> $LOG; }
cd $REPO && source .venv/bin/activate

say "waiting for 2.5b to finish"
until grep -q "2.5b exited" $LL 2>/dev/null; do sleep 120; done
grep -q "2.5b exited rc=0" $LL && [ -f $FINAL_B ] || { say "ABORT: 2.5b did not finish cleanly; no smoke run"; exit 1; }
while pgrep -f "scripts/train.p[y]" > /dev/null; do sleep 30; done
say "2.5b done; pulling SFT shards from the A10"

mkdir -p $DATA
rsync -a -e "ssh -i $HOME/.ssh/a10_temp -o BatchMode=yes" "$A10:/home/ubuntu/phase3_decon/" $DATA/ >> $LOG 2>&1 || { say "ABORT: rsync failed"; exit 1; }
python3 - $DATA >> $LOG 2>&1 <<'EOF' || { echo "ABORT: shard check failed" >> /lambda/nfs/h100-f1/phase3_smoke.log; exit 1; }
import glob, json, os, sys
bad = 0
for d in sorted(glob.glob(sys.argv[1] + "/*/")):
    m = json.load(open(d + "meta.json"))
    sh = sorted(glob.glob(d + "shard_*.bin")); mk = sorted(glob.glob(d + "mask_*.bin"))
    tok = sum(os.path.getsize(f) for f in sh) // 2; msk = sum(os.path.getsize(f) for f in mk)
    ok = tok == msk == m["total_tokens_incl_padding"] and len(sh) == len(mk) == m["n_shards"]
    bad += not ok
    print(("OK " if ok else "BAD"), os.path.basename(d.rstrip("/")), tok, msk, m["total_tokens_incl_padding"])
sys.exit(1 if bad else 0)
EOF
say "shards verified; starting the 200-step smoke test"

python3 -u scripts/train.py configs/phase3_smoke.yaml --init_from $FINAL_B > /lambda/nfs/h100-f1/train_phase3_smoke.log 2>&1 < /dev/null
rc=$?
say "smoke run exited rc=$rc"
SMOKE=/lambda/nfs/h100-f1/checkpoints/moe0-9b-a0-2b-phase03-sft-smoke/step_200.pt
[ $rc -eq 0 ] && [ -f $SMOKE ] || exit 1

# chat-format samples (same wire format as SFT), greedy, no penalties
printf '%s\n' "/chat" "/greedy" "/rep 1.0" "/ngram 0" "/tokens 160" \
  "What is the capital city of India?" \
  "Ali had three apples. He gave one to Zara and ate one. How many apples does Ali have now?" \
  "Naledi was beaten by Kofi. Who won?" \
  "Write a Python function that returns the factorial of n." \
  "Who are you?" \
  "exit" | python3 scripts/chat_repl.py --config configs/phase3_smoke.yaml --checkpoint $SMOKE > /lambda/nfs/h100-f1/phase3_smoke_samples.txt 2>&1
say "SMOKE DONE (samples in phase3_smoke_samples.txt). Full SFT waits for the user's go."
