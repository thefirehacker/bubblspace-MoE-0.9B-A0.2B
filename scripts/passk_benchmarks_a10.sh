#!/bin/bash
# A10 (28 Nov): pass@8 chat benchmarks for SFT run 02b (the RL start) and RL v1 step 500 (plan-06 step 3): 8 samples per
# question at temperature 1.0 (scripts/eval_chat_passk.py), code run in the docker sandbox, then the summary with the
# greedy results next to it. Models from the vLLM folders in ~/vllm_models (weights in ~/hf_export, the A10's own disk).
#   nohup bash scripts/passk_benchmarks_a10.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b
export PATH=$HOME/vllm_env/bin:$PATH
OUT=benchmark/phase4/passk; mkdir -p $OUT; LOG=$OUT/passk.log
say () { echo "[$(date -u)] $*" >> $LOG; }
MODELS="phase3-run02b-step1500-tokens40B phase4-rlv1-step500-tokens40B"
for M in $MODELS; do
  say "generate (8 samples, T=1.0): $M"
  ~/vllm_env/bin/python -u scripts/eval_chat_passk.py --vllm_model_dir ~/vllm_models/$M --out $OUT/gen_$M.json \
    > $OUT/gen_$M.log 2>&1 || { say "FAILED generate $M"; continue; }
  say "sandbox: $M"
  .venv/bin/python scripts/sandbox_exec.py --codegen $OUT/gen_$M.json --out $OUT/codeexec_$M.json --workers 16 \
    > $OUT/codeexec_$M.log 2>&1 || say "FAILED sandbox $M"
  say "done $M: $(grep -hE '^\[(gsm8k|bfcl:[a-z]+)\] ' $OUT/gen_$M.log | tr '\n' ' ' | cut -c1-400)"
done
.venv/bin/python scripts/summarize_passk.py --dir $OUT --greedy_dir benchmark/phase4 --models $MODELS >> $LOG 2>&1
say "PASSK BENCHMARKS DONE"
