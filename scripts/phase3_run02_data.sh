#!/bin/bash
# A10: Phase 3 run 02 SFT data (27 Nov). Run 01 over-refused harmless commands; this rebuilds the two changed pools:
#   tulu3_run02     Tülu 3 without WildJailbreak, CoCoNot and OLMo's hard-coded identity rows; WildGuardMix sampled to 10%;
#                   refusal-opening replies dropped from the other sources (scripts/phase3_build_sft.py, src_tulu3_run02)
#   own_facts_cmd   facts asked as commands and in formats (JSON / CSV / list / one line), country-capital tables
#                   without India and France (hand tests) (scripts/phase3_gen_own_sets.py --only own_facts_cmd)
# then decontaminates both with the run 01 eval index. Every other pool is reused from run 01 unchanged.
#   nohup bash scripts/phase3_run02_data.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
LOG=~/phase3_logs/run02_data.log; RAW=~/phase3_raw
say () { echo "[$(date -u)] $*" >> $LOG; }
say "start"
python3 scripts/phase3_gen_own_sets.py --only own_facts_cmd --facts_parquet $RAW/nvidia__Nemotron-Fact-Seeking/Nemotron-Pretraining-Fact-Seeking/part_000000.parquet \
  --out $RAW/own --repo_root . >> $LOG 2>&1 || { say "ABORT gen"; exit 1; }
for s in tulu3_run02 own_facts_cmd; do
  python3 scripts/phase3_build_sft.py --source $s --raw $RAW --out ~/phase3_build > ~/phase3_logs/build_$s.log 2>&1; say "build $s rc=$?"
  python3 scripts/phase3_decon.py clean --index ~/phase3_logs/sft_eval_index.npz --pool ~/phase3_build/$s --out_root ~/phase3_decon >> $LOG 2>&1; say "decon $s rc=$?"
done
say "RUN02_DATA_DONE"
