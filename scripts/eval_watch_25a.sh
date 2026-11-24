#!/bin/bash
# A10: two-tier eval watcher for Phase 2.5a (doc/plan-03-phase-2.5-knowledge-context.md §2.6).
#
#   FAST tier, every 2,000 steps (≈1B tokens), ≈1.5 h:
#     probes v2 / v3 / v4 / v5, simple-facts probe, TriviaQA + NQ-open (first 1,000 each, 5-shot),
#     English 6 (forgetting check), quick suite (FIM / tool-call format / reasoning / greedy samples)
#   FULL tier, at steps 8,000 / 16,000 / 24,000 / 32,000 and the final 38,147 (≈4B tokens apart), +≈3 h:
#     full TriviaQA (2,000) and NQ-open (3,610) instead of the 1,000 slices, CRUXEval-O + BFCL
#     (decontaminated), routing balance, HumanEval + MBPP generation with sandboxed execution
#     (execution runs in the background on CPU while the next checkpoint's GPU evals start)
#
# Checkpoints arrive from fire01 via scripts/relay_ckpts_25a.sh. Fast-tier-only checkpoints are deleted
# after their evals (originals stay on fire01); full-tier checkpoints are kept.
# After each checkpoint, scripts/summarize_evals.py rewrites $OUT/summary.md (one table, all steps).
# Usage (on the A10): nohup bash scripts/eval_watch_25a.sh > /dev/null 2>&1 &
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
OUT=benchmark/phase25a; mkdir -p $OUT; LOG=$OUT/eval_watch.log
CKD=/lambda/nfs/us-west-1-eval-inference/phase25a
P=benchmark/data/probefinding-winograndestyle
V2=$P/probefinding-winogrande-20261121T185948.json
V3=$P/probefinding-winogrande-v3-20261122.json
V4=$P/probefinding-winogrande-v4-heldout-20261122.json
V5=$P/probefinding-winogrande-v5-heldout-structures-20261123.json
FACTS=benchmark/data/facts-closedbook/facts-simple-v1-20261123.json
for f in $V2 $V3 $V4 $V5 $FACTS; do [ -f $f ] || { echo "[$(date -u)] MISSING $f, abort" >> $LOG; exit 1; }; done

FULL=" 8000 16000 24000 32000 38147 "
STEPS="$(seq 2000 2000 38000) 38147"
for s in $STEPS; do
  CK=$CKD/step_$s.pt; T=phase25a_step$s
  until [ -f $CK ]; do sleep 60; done
  is_full=0; [[ "$FULL" == *" $s "* ]] && is_full=1
  echo "[$(date -u)] === $T (full=$is_full) ===" >> $LOG

  # ---- fast tier ----
  for v in 2 3 4 5; do
    eval cf=\$V$v
    python3 scripts/probe_entity_binding.py --checkpoint $CK --cases_file $cf --output $OUT/probe_v${v}_$T.json > $OUT/probe_v${v}_$T.log 2>&1
  done
  python3 scripts/eval_facts.py --checkpoint $CK --facts $FACTS --output $OUT/facts_$T.json > $OUT/facts_$T.log 2>&1
  if [ $is_full = 1 ]; then
    python3 scripts/run_benchmarks.py --checkpoint $CK --tasks triviaqa --num_fewshot 5 --limit 2000 --output $OUT/closedbook_triviaqa_$T.json > $OUT/closedbook_$T.log 2>&1
    python3 scripts/run_benchmarks.py --checkpoint $CK --tasks nq_open --num_fewshot 5 --output $OUT/closedbook_nq_$T.json >> $OUT/closedbook_$T.log 2>&1
  else
    python3 scripts/run_benchmarks.py --checkpoint $CK --tasks triviaqa --num_fewshot 5 --limit 1000 --output $OUT/closedbook_triviaqa1k_$T.json > $OUT/closedbook_$T.log 2>&1
    python3 scripts/run_benchmarks.py --checkpoint $CK --tasks nq_open --num_fewshot 5 --limit 1000 --output $OUT/closedbook_nq1k_$T.json >> $OUT/closedbook_$T.log 2>&1
  fi
  python3 scripts/run_benchmarks.py --config configs/base.yaml --checkpoint $CK --output $OUT/english6_$T.json > $OUT/english6_$T.log 2>&1
  python3 scripts/eval_phase2_quick.py --checkpoint $CK --pools_yaml configs/pools_a10.yaml --out $OUT/quick_$T.json > $OUT/quick_$T.log 2>&1
  echo "[$(date -u)] fast tier done $T" >> $LOG

  # ---- full tier ----
  if [ $is_full = 1 ]; then
    python3 scripts/eval_phase2_milestone.py --checkpoint $CK --tasks cruxeval_o,bfcl --bfcl_exclude_json benchmark/phase2/contamination_bfcl.json --out $OUT/milestone_$T.json > $OUT/milestone_$T.log 2>&1
    mkdir -p $OUT/routing
    python3 scripts/analyze_routing.py --checkpoint $CK --pools_yaml configs/pools_a10.yaml --windows_per_group 48 --out $OUT/routing/routing_$T.json > $OUT/routing/routing_$T.log 2>&1
    python3 -u scripts/eval_code_gen.py --checkpoint $CK --tasks humaneval,mbpp --out $OUT/codegen_$T.json > $OUT/codegen_$T.log 2>&1
    ( python3 scripts/sandbox_exec.py --codegen $OUT/codegen_$T.json --out $OUT/codeexec_$T.json > $OUT/codeexec_$T.log 2>&1
      python3 scripts/summarize_evals.py --dir $OUT --prefix phase25a_step --out $OUT/summary.md --baseline benchmark/phase2:phase2_step22888 > /dev/null 2>&1
      echo "[$(date -u)] code exec done $T" >> $LOG ) &
    echo "[$(date -u)] full tier GPU part done $T (code exec in background)" >> $LOG
  fi

  python3 scripts/summarize_evals.py --dir $OUT --prefix phase25a_step --out $OUT/summary.md --baseline benchmark/phase2:phase2_step22888 > /dev/null 2>&1
  echo "[$(date -u)] done $T" >> $LOG
  [ $is_full = 0 ] && rm -f $CK
done
wait
echo "[$(date -u)] watch 25a finished" >> $LOG
