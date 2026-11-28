#!/bin/bash
# A10 (28 Nov): plan-06 step 3, the last two checks on the RL model, with its SFT start for comparison.
#  1. Hand tests (doc/handtest-01): the 18 relational + 16 calculation prompts, greedy, repetition penalty 1.0, no n-gram
#     block. Raw text with 40 new tokens (as the 26 Nov base-model runs) and in the chat template with 200 tokens (as
#     OLMo-2 1B Instruct with its template). Scored by hand afterwards.
#  2. Long context (scripts/eval_long_context.py, as Phase 2.5b): passkey and 2-hop at 2k-16k plus loss by position,
#     raw text (comparable with the 2.5b numbers), then passkey and 2-hop again in the chat template (--chat).
# Weights: ~/hf_export/<branch>/model.safetensors (the HF exports on the A10's own disk).
#   nohup bash scripts/handtest_longctx_a10.sh > /dev/null 2>&1 < /dev/null &
cd /home/ubuntu/moe0.9b-a0.2b
OUT=benchmark/phase4/handtest_longctx; mkdir -p $OUT; LOG=$OUT/run.log
say () { echo "[$(date -u)] $*" >> $LOG; }
P=benchmark/handtest/prompts
cat $P/relational18.txt $P/calculation16.txt > $OUT/prompts34.txt
MODELS="phase3-run02b-step1500-tokens40B phase4-rlv1-step500-tokens40B"
CFG=configs/phase3_run02b.yaml
for M in $MODELS; do
  CK=$HOME/hf_export/$M/model.safetensors
  say "hand tests raw (40 tokens): $M"
  printf "/greedy\n/rep 1.0\n/ngram 0\n/tokens 40\n%s\nexit\n" "$(cat $OUT/prompts34.txt)" \
    | .venv/bin/python scripts/chat_repl.py --config $CFG --checkpoint $CK 2>&1 \
    | .venv/bin/python scripts/handtest_parse.py "$M raw" $OUT/prompts34.txt >> $OUT/handtest_raw.txt 2>> $LOG
  say "hand tests chat (200 tokens): $M"
  printf "/greedy\n/rep 1.0\n/ngram 0\n/tokens 200\n%s\nexit\n" "$(cat $OUT/prompts34.txt)" \
    | .venv/bin/python scripts/chat_repl.py --config $CFG --checkpoint $CK --chat 2>&1 \
    | .venv/bin/python scripts/handtest_parse.py "$M chat" $OUT/prompts34.txt >> $OUT/handtest_chat.txt 2>> $LOG
done
for M in $MODELS; do
  CK=$HOME/hf_export/$M/model.safetensors
  say "long context raw: $M"
  .venv/bin/python scripts/eval_long_context.py eval --config $CFG --checkpoint $CK \
    --docs benchmark/data/long-context/long_docs_heldout.npz --output $OUT/long_context_raw_$M.json \
    > $OUT/long_context_raw_$M.log 2>&1 || say "FAILED long context raw $M"
  say "long context chat: $M"
  .venv/bin/python scripts/eval_long_context.py eval --config $CFG --checkpoint $CK --chat --tests passkey,multihop \
    --docs benchmark/data/long-context/long_docs_heldout.npz --output $OUT/long_context_chat_$M.json \
    > $OUT/long_context_chat_$M.log 2>&1 || say "FAILED long context chat $M"
  say "done $M: $(grep -hE '^(passkey|multihop)' $OUT/long_context_raw_$M.log $OUT/long_context_chat_$M.log | tr '\n' ' ' | cut -c1-400)"
done
say "HANDTEST LONGCTX DONE"
