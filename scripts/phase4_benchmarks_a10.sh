#!/bin/bash
# A10 (28 Nov): benchmark Phase 3 SFT and Phase 4 RL v1 under identical conditions (fire01 is gone; models from the HF
# exports in ~/hf_export). For each model: chat evals with the vLLM backend (refusal check, HumanEval, MBPP, GSM8K, IFEval,
# BFCL incl. no-tool-needed), code run in the docker sandbox, IFEval scored with lm-eval. Then the 23-conversation chat
# test and the English 6-task suite (forgetting check) for run 02b vs RL v1 step 500, and a summary table.
#   nohup bash scripts/phase4_benchmarks_a10.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b
export PATH=$HOME/vllm_env/bin:$PATH   # vLLM builds kernels with ninja from this env
OUT=benchmark/phase4; mkdir -p $OUT; LOG=$OUT/benchmarks.log
say () { echo "[$(date -u)] $*" >> $LOG; }
MODELS="phase3-step1500 phase3-run02b-step1500-tokens40B phase4-rlv1-step100-tokens40B phase4-rlv1-step300-tokens40B phase4-rlv1-step500-tokens40B"
for M in $MODELS; do
  say "chat evals (vLLM): $M"
  ~/vllm_env/bin/python -u scripts/eval_chat.py --backend vllm --vllm_model_dir ~/vllm_models/$M --config configs/phase3_run02b.yaml \
    --out $OUT/chat_$M.json > $OUT/chat_$M.log 2>&1 || { say "FAILED chat evals $M"; continue; }
  .venv/bin/python scripts/sandbox_exec.py --codegen $OUT/chat_$M.json --out $OUT/codeexec_$M.json > $OUT/codeexec_$M.log 2>&1
  .venv/bin/python scripts/eval_chat.py --score_ifeval $OUT/chat_$M.json >> $OUT/chat_$M.log 2>&1
  say "done $M: $(grep -hE '^\[(refusal|gsm8k|bfcl:[a-z]+)\] ' $OUT/chat_$M.log | tr '\n' ' ' | cut -c1-300)"
done
for M in phase3-run02b-step1500-tokens40B phase4-rlv1-step500-tokens40B; do
  say "chat test + English 6: $M"
  .venv/bin/python scripts/phase3_chat_smoke.py --config configs/phase3_run02b.yaml --checkpoint ~/hf_export/$M/model.safetensors \
    --out $OUT/chat_smoke_$M.md > /dev/null 2>&1
  .venv/bin/python scripts/run_benchmarks.py --config configs/phase3_run02b.yaml --checkpoint ~/hf_export/$M/model.safetensors \
    --output $OUT/english6_$M.json > $OUT/english6_$M.log 2>&1
done
.venv/bin/python scripts/summarize_phase4.py --dir $OUT --out $OUT/summary.md >> $LOG 2>&1
say "PHASE4 BENCHMARKS DONE"
