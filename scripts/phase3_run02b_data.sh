#!/bin/bash
# A10: Phase 3 run 02b SFT data (27 Nov). Run 02's smoke test fixed over-refusal but stopped refusing harmful requests
# and made up live facts (doc/issue-02-phase3-over-refusal.md §6). Rebuilds:
#   tulu3_run02b   all of WildGuardMix, 10% of WildJailbreak, 10% of CoCoNot without its underspecified/real-time replies;
#                  refusal-opening replies elsewhere and OLMo identity rows still out (phase3_build_sft.py, src_tulu3_run02b)
#   own_realtime   live-data questions: no tools -> "I don't have live data"; matching tool -> tool call;
#                  tools present but not needed -> direct answer (phase3_gen_own_sets.py --only own_realtime)
# then decontaminates both. own_facts_cmd and every other pool are reused unchanged.
#   nohup bash scripts/phase3_run02b_data.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
LOG=~/phase3_logs/run02b_data.log; RAW=~/phase3_raw
say () { echo "[$(date -u)] $*" >> $LOG; }
say "start"
python3 scripts/phase3_gen_own_sets.py --only own_realtime --facts_parquet $RAW/nvidia__Nemotron-Fact-Seeking/Nemotron-Pretraining-Fact-Seeking/part_000000.parquet \
  --out $RAW/own --repo_root . >> $LOG 2>&1 || { say "ABORT gen"; exit 1; }
for s in tulu3_run02b own_realtime; do
  python3 scripts/phase3_build_sft.py --source $s --raw $RAW --out ~/phase3_build > ~/phase3_logs/build_$s.log 2>&1; say "build $s rc=$?"
  python3 scripts/phase3_decon.py clean --index ~/phase3_logs/sft_eval_index.npz --pool ~/phase3_build/$s --out_root ~/phase3_decon >> $LOG 2>&1; say "decon $s rc=$?"
done
say "RUN02B_DATA_DONE"
