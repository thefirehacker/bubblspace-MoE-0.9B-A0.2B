#!/bin/bash
# A10: WSP on the direction (word-order) failure. Head-to-head with the data fix in
# doc/binding-study-02-priors-and-word-order.md §6.1: same base checkpoint (Phase 2 final), same probes and scoring.
#   - SOLVE the closed-form edit (scripts/wsp_closed_form.py) on probe v5 "passive" (40 cases, both contexts);
#     v5 relative_clause / respectively are the protected keys.
#   - SCORE on held-out cases the edit never saw: v6 passive_heldout_frame, v6 object_relative / fit_inverse, v4,
#     plus written answers (generation) on v5 / v6, and LAMBADA + PIQA as the damage check.
#   - Sweep layers x lambda. Edited checkpoints are deleted after scoring; the base checkpoint is never modified.
# Every GPU job takes the shared lock, so the eval watchers keep working. Base scores are reused from benchmark/binding_test.
#
# Usage (on the A10): bash scripts/wsp_direction.sh ["6 8 10"] ["5 10"]
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
set -u
LAYERS=${1:-"6 8 10"}; LAMS=${2:-"5 10"}
P=benchmark/data/probefinding-winograndestyle
V4=$P/probefinding-winogrande-v4-heldout-20261122.json
V5=$P/probefinding-winogrande-v5-heldout-structures-20261123.json
V6=$P/probefinding-winogrande-v6-order-20261125.json
BASE=/lambda/nfs/us-west-1-eval-inference/keep/phase2_step_22888.pt
W=/lambda/nfs/us-west-1-eval-inference/wsp_direction; O=benchmark/wsp_direction
mkdir -p $W $O; LOG=$O/run.log
L="flock /tmp/a10_gpu.lock"
say () { echo "[$(date -u)] $*" >> $LOG; }

score () {  # score <tag> <checkpoint>
  T=$1; CK=$2
  for v in 4 5 6; do
    eval cf=\$V$v
    $L python3 scripts/probe_entity_binding.py --checkpoint $CK --cases_file $cf --output $O/probe_v${v}_$T.json > $O/probe_v${v}_$T.log 2>&1
  done
  for v in 5 6; do
    eval cf=\$V$v
    $L python3 scripts/eval_binding_generation.py --checkpoint $CK --cases_file $cf --output $O/gen_v${v}_$T.json > $O/gen_v${v}_$T.log 2>&1
  done
  $L python3 scripts/run_benchmarks.py --checkpoint $CK --tasks lambada_openai,piqa --output $O/english2_$T.json > $O/english2_$T.log 2>&1
  python3 scripts/probe_prior_free.py $O/probe_v4_$T.json $O/probe_v5_$T.json $O/probe_v6_$T.json > $O/prior_free_$T.txt 2>&1
  say "scored $T"
}

say "start: base $BASE, layers [$LAYERS], lambdas [$LAMS]"
for Ly in $LAYERS; do
  for lam in $LAMS; do
    T=L${Ly}_lam${lam}; ED=$W/wsp_$T.pt
    $L python3 scripts/wsp_closed_form.py --checkpoint $BASE --cases_file $V5 --target_categories passive \
      --edit_layer $Ly --lam $lam --out_checkpoint $ED --diagnostic_output $O/edit_$T.json > $O/edit_$T.log 2>&1 \
      || { say "edit FAILED $T"; tail -3 $O/edit_$T.log >> $LOG; continue; }
    score $T $ED
    rm -f $ED
  done
done
say "wsp direction done"
