#!/bin/bash
# A10, one-off: bits-per-byte baselines (doc/plan-05-bits-per-byte-eval.md). The first run writes the fixed 2,000-document
# sample file that every later run reuses. Each job takes the shared GPU lock.
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
DATA=$HOME/eval_data/uncheatable-2026-10.parquet; O=benchmark/bpb; mkdir -p $O
K=/lambda/nfs/us-west-1-eval-inference
run () { flock /tmp/a10_gpu.lock python3 scripts/eval_bpb.py --data $DATA "$@"; }
run --checkpoint $K/keep/phase2_step_22888.pt --out benchmark/phase2/bpb_phase2_step22888.json > $O/phase2_step22888.log 2>&1
run --checkpoint $K/step_19073.pt --out $O/bpb_phase1_step19073.json > $O/phase1_step19073.log 2>&1
for m in HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B EleutherAI/pythia-410m; do
  run --hf_model $m --out $O/bpb_$(basename $m).json > $O/$(basename $m).log 2>&1
done
echo "[$(date -u)] bpb baselines done" >> $O/baselines.log
