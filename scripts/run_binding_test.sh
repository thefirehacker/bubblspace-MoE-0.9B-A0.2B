#!/bin/bash
# A10: two-arm binding test (doc/binding-study-02-priors-and-word-order.md §6.1). Not a training phase.
# Same checkpoint (Phase 2 final), same steps and learning rate; control = contrastive v3, fix = v3 + order alternation.
# Every GPU job takes the shared lock, so the eval watchers keep working.
cd /home/ubuntu/moe0.9b-a0.2b && source .venv/bin/activate
P=benchmark/data/probefinding-winograndestyle
V2=$P/probefinding-winogrande-20261121T185948.json; V3=$P/probefinding-winogrande-v3-20261122.json
V4=$P/probefinding-winogrande-v4-heldout-20261122.json; V5=$P/probefinding-winogrande-v5-heldout-structures-20261123.json
V6=$P/probefinding-winogrande-v6-order-20261125.json
BASE=/lambda/nfs/us-west-1-eval-inference/keep/phase2_step_22888.pt
W=/lambda/nfs/us-west-1-eval-inference/binding_test; D=$HOME/binding_test_data; O=benchmark/binding_test
mkdir -p $W $D $O; LOG=$O/run.log
STEPS=${STEPS:-400}
say () { echo "[$(date -u)] $*" >> $LOG; }
L="flock /tmp/a10_gpu.lock"

# data (CPU): both arms from the same generator and seed; probe v6 is written once
python3 scripts/gen_contrastive_v4.py --arm control --n_docs 100000 --out $D/control.jsonl --probe_files $V2 $V3 $V4 $V5 >> $LOG 2>&1
python3 scripts/gen_contrastive_v4.py --arm fix --n_docs 100000 --out $D/fix.jsonl --probe_files $V2 $V3 $V4 $V5 --probe_v5 $V5 --probe_out $V6 >> $LOG 2>&1
[ -f $V6 ] || { say "ABORT: probe v6 not written"; exit 1; }
say "data ready"

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
  $L python3 scripts/binding_levels_bubbl.py --checkpoint $CK --winogrande --out $O/winogrande_$T.json > $O/winogrande_$T.log 2>&1
  $L python3 scripts/run_benchmarks.py --checkpoint $CK --tasks lambada_openai,piqa --output $O/english2_$T.json > $O/english2_$T.log 2>&1
  python3 scripts/probe_prior_free.py $O/probe_v4_$T.json $O/probe_v5_$T.json $O/probe_v6_$T.json > $O/prior_free_$T.txt 2>&1
  say "scored $T"
}

score base $BASE
for arm in control fix; do
  $L python3 scripts/binding_finetune_test.py --checkpoint $BASE --data $D/$arm.jsonl --out $W/$arm.pt --steps $STEPS > $O/finetune_$arm.log 2>&1
  [ -f $W/$arm.pt ] || { say "finetune $arm FAILED"; tail -3 $O/finetune_$arm.log >> $LOG; continue; }
  say "finetuned $arm"
  score $arm $W/$arm.pt
done
say "binding test done"
