#!/bin/bash
# WSP re-run on a newer checkpoint (planned for the Phase 2.5a final), with the evaluation fixes from
# doc/wsp-experiments.md §6-§7:
#   - the edit is SOLVED on probe v2's three target categories (as in attempt 2), but SCORED on the held-out
#     probes v4 / v5 (never seen by the edit) as well as v2 / v3, so solving and scoring no longer use the same cases
#   - every edit also gets the free-generation check (scripts/eval_binding_generation.py) on v4 / v5,
#     so a probe gain without a generation gain is visible immediately
#   - lambda (protection strength) is swept instead of fixed at 1.0, across a few layers in the
#     binding band found earlier (layers 5-11)
# Edited checkpoints are deleted after scoring (each is ≈3.7 GB); the base checkpoint is never modified.
#
# Usage: bash scripts/wsp_rerun.sh <base_checkpoint.pt> <out_dir> ["6 8 10"] ["1 2 5 10"]
set -u
CK=$1; OUT=$2; LAYERS=${3:-"6 8 10"}; LAMS=${4:-"1 2 5 10"}
P=benchmark/data/probefinding-winograndestyle
V2=$P/probefinding-winogrande-20261121T185948.json
V3=$P/probefinding-winogrande-v3-20261122.json
V4=$P/probefinding-winogrande-v4-heldout-20261122.json
V5=$P/probefinding-winogrande-v5-heldout-structures-20261123.json
TARGETS=comparative_value,emotional_attribution,possession_obligation
mkdir -p $OUT; LOG=$OUT/wsp_rerun.log

score () {  # $1 = checkpoint, $2 = tag
  for v in 2 3 4 5; do
    eval cf=\$V$v
    python3 scripts/probe_entity_binding.py --checkpoint $1 --cases_file $cf --output $OUT/probe_v${v}_$2.json > $OUT/probe_v${v}_$2.log 2>&1
  done
  for v in 4 5; do
    eval cf=\$V$v
    python3 scripts/eval_binding_generation.py --checkpoint $1 --cases_file $cf --output $OUT/gen_v${v}_$2.json > $OUT/gen_v${v}_$2.log 2>&1
  done
  echo "[$(date -u)] scored $2" >> $LOG
}

echo "[$(date -u)] base $CK, layers [$LAYERS], lambdas [$LAMS]" >> $LOG
score $CK base
for L in $LAYERS; do
  for lam in $LAMS; do
    T=L${L}_lam${lam}; ED=$OUT/wsp_$T.pt
    python3 scripts/wsp_closed_form.py --checkpoint $CK --cases_file $V2 --target_categories $TARGETS \
      --edit_layer $L --lam $lam --out_checkpoint $ED --diagnostic_output $OUT/edit_$T.json > $OUT/edit_$T.log 2>&1 \
      || { echo "[$(date -u)] edit FAILED $T" >> $LOG; continue; }
    score $ED $T
    rm -f $ED
  done
done

python3 - "$OUT" <<'EOF' > $OUT/summary.md
import json, os, sys
d = sys.argv[1]
J = lambda n: json.load(open(os.path.join(d, n))) if os.path.exists(os.path.join(d, n)) else None
tags = ["base"] + sorted({f[len("edit_"):-5] for f in os.listdir(d) if f.startswith("edit_") and f.endswith(".json")})
print("| edit | probe v2 | v3 | v4 (held-out) | v5 (held-out) | gen v4 flip / gap | gen v5 flip / gap | ‖ΔW‖ |")
print("|---|---|---|---|---|---|---|---|")
for t in tags:
    p = [J(f"probe_v{v}_{t}.json") for v in (2, 3, 4, 5)]
    g = [J(f"gen_v{v}_{t}.json") for v in (4, 5)]
    e = J(f"edit_{t}.json")
    ps = [f"{x['n_correct']}/{x['n_total']}" if x else "—" for x in p]
    gs = [f"{x['summary']['flip_hit']}/{x['summary']['n_cases']} / {x['summary']['mean_gap_nats']}" if x else "—" for x in g]
    print(f"| {t} | " + " | ".join(ps + gs) + f" | {e['delta_w_norm']:.2f} |" if e else f"| {t} | " + " | ".join(ps + gs) + " | — |")
EOF
cat $OUT/summary.md >> $LOG
echo "[$(date -u)] done" >> $LOG
