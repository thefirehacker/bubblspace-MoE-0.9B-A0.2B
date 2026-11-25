#!/bin/bash
# A10: bits-per-byte eval (scripts/eval_bpb.py) for the 2.5a FULL-tier checkpoints, run alongside the already-running
# scripts/eval_watch_25a.sh (that script isn't edited mid-run). Waits until the main watcher logs "done <step>" for each
# full-tier step (those checkpoints are kept), then scores it under the shared GPU lock and refreshes summary.md.
# Usage (on the A10): nohup bash scripts/bpb_watch_25a.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
OUT=benchmark/phase25a; LOG=$OUT/eval_watch.log
CKD=/lambda/nfs/us-west-1-eval-inference/phase25a
DATA=$HOME/eval_data/uncheatable-2026-10.parquet
for s in 8000 16000 24000 32000 38147; do
  T=phase25a_step$s
  until grep -q "\] done $T\$" $LOG 2>/dev/null; do sleep 300; done
  exec 9>/tmp/a10_gpu.lock; flock 9
  echo "[$(date -u)] bpb start $T" >> $LOG
  python3 scripts/eval_bpb.py --checkpoint $CKD/step_$s.pt --data $DATA --out $OUT/bpb_$T.json > $OUT/bpb_$T.log 2>&1
  python3 scripts/summarize_evals.py --dir $OUT --prefix phase25a_step --out $OUT/summary.md --baseline benchmark/phase2:phase2_step22888 > /dev/null 2>&1
  flock -u 9; exec 9>&-
  echo "[$(date -u)] bpb done $T" >> $LOG
done
