#!/bin/bash
# A10: eval watcher for Phase 2.5b (context 16,384; doc/plan-03-phase-2.5-knowledge-context.md §3).
# Checkpoints 1000, 2000 and the final 2861 arrive from fire01 via scripts/relay_ckpts_25b.sh.
#
# Every checkpoint: long-context evals (passkey, multi-hop, loss by position at 2k-16k) and the 2.5a fast tier
# (probes v2-v5, facts, TriviaQA/NQ 1k slices, English 6, quick suite), so short-context regressions show up.
# Final checkpoint: additionally the full tier (full TriviaQA/NQ, CRUXEval/BFCL, routing, HumanEval/MBPP).
# IMPORTANT: every script gets --config configs/phase25b.yaml (RoPE theta 1e6, max_seq_len 16384);
# evaluating a 2.5b checkpoint with base.yaml's theta 1e4 would be silently wrong.
# Usage (on the A10): nohup bash scripts/eval_watch_25b.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
OUT=benchmark/phase25b; mkdir -p $OUT; LOG=$OUT/eval_watch.log
CKD=/lambda/nfs/us-west-1-eval-inference/phase25b
C=configs/phase25b.yaml
P=benchmark/data/probefinding-winograndestyle
V2=$P/probefinding-winogrande-20261121T185948.json
V3=$P/probefinding-winogrande-v3-20261122.json
V4=$P/probefinding-winogrande-v4-heldout-20261122.json
V5=$P/probefinding-winogrande-v5-heldout-structures-20261123.json
FACTS=benchmark/data/facts-closedbook/facts-simple-v1-20261123.json
DOCS=benchmark/data/long-context/long_docs_heldout.npz
for f in $V2 $V3 $V4 $V5 $FACTS $DOCS $C; do [ -f $f ] || { echo "[$(date -u)] MISSING $f, abort" >> $LOG; exit 1; }; done

for s in 1000 2000 2861; do
  CK=$CKD/step_$s.pt; T=phase25b_step$s
  until [ -f $CK ]; do sleep 60; done
  exec 9>/tmp/a10_gpu.lock; flock 9   # GPU lock shared with the binding study (waits for its current job)
  echo "[$(date -u)] === $T ===" >> $LOG
  python3 scripts/eval_long_context.py eval --config $C --checkpoint $CK --docs $DOCS --output $OUT/long_context_$T.json > $OUT/long_context_$T.log 2>&1
  echo "[$(date -u)] long-context done $T" >> $LOG
  for v in 2 3 4 5; do
    eval cf=\$V$v
    python3 scripts/probe_entity_binding.py --config $C --checkpoint $CK --cases_file $cf --output $OUT/probe_v${v}_$T.json > $OUT/probe_v${v}_$T.log 2>&1
  done
  for v in 4 5; do  # does the model WRITE the bound answer? (doc/wsp-experiments.md §6.1)
    eval cf=\$V$v
    python3 scripts/eval_binding_generation.py --config $C --checkpoint $CK --cases_file $cf --output $OUT/gen_v${v}_$T.json > $OUT/gen_v${v}_$T.log 2>&1
  done
  python3 scripts/eval_facts.py --config $C --checkpoint $CK --facts $FACTS --output $OUT/facts_$T.json > $OUT/facts_$T.log 2>&1
  if [ $s = 2861 ]; then
    python3 scripts/run_benchmarks.py --config $C --checkpoint $CK --tasks triviaqa --num_fewshot 5 --limit 2000 --output $OUT/closedbook_triviaqa_$T.json > $OUT/closedbook_$T.log 2>&1
    python3 scripts/run_benchmarks.py --config $C --checkpoint $CK --tasks nq_open --num_fewshot 5 --output $OUT/closedbook_nq_$T.json >> $OUT/closedbook_$T.log 2>&1
  else
    python3 scripts/run_benchmarks.py --config $C --checkpoint $CK --tasks triviaqa --num_fewshot 5 --limit 1000 --output $OUT/closedbook_triviaqa1k_$T.json > $OUT/closedbook_$T.log 2>&1
    python3 scripts/run_benchmarks.py --config $C --checkpoint $CK --tasks nq_open --num_fewshot 5 --limit 1000 --output $OUT/closedbook_nq1k_$T.json >> $OUT/closedbook_$T.log 2>&1
  fi
  python3 scripts/run_benchmarks.py --config $C --checkpoint $CK --output $OUT/english6_$T.json > $OUT/english6_$T.log 2>&1
  python3 scripts/eval_phase2_quick.py --config $C --checkpoint $CK --pools_yaml configs/pools_a10.yaml --out $OUT/quick_$T.json > $OUT/quick_$T.log 2>&1
  if [ $s = 2861 ]; then
    python3 scripts/eval_phase2_milestone.py --config $C --checkpoint $CK --tasks cruxeval_o,bfcl --bfcl_exclude_json benchmark/phase2/contamination_bfcl.json --out $OUT/milestone_$T.json > $OUT/milestone_$T.log 2>&1
    mkdir -p $OUT/routing
    python3 scripts/analyze_routing.py --config $C --checkpoint $CK --pools_yaml configs/pools_a10.yaml --windows_per_group 48 --out $OUT/routing/routing_$T.json > $OUT/routing/routing_$T.log 2>&1
    python3 -u scripts/eval_code_gen.py --config $C --checkpoint $CK --tasks humaneval,mbpp --out $OUT/codegen_$T.json > $OUT/codegen_$T.log 2>&1
    python3 scripts/sandbox_exec.py --codegen $OUT/codegen_$T.json --out $OUT/codeexec_$T.json > $OUT/codeexec_$T.log 2>&1
  fi
  python3 scripts/summarize_evals.py --dir $OUT --prefix phase25b_step --out $OUT/summary.md \
    --baseline benchmark/phase2:phase2_step22888 --baseline benchmark/phase25a:phase25a_step38147 --long_context > /dev/null 2>&1
  flock -u 9; exec 9>&-
  echo "[$(date -u)] done $T" >> $LOG
done
echo "[$(date -u)] watch 25b finished" >> $LOG
