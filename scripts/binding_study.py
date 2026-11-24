"""Cross-model binding study: three-level scores (scripts/binding_levels_hf.py) across public pretraining
checkpoints and final base models. Resumable: a job whose output JSON exists is skipped.

  A10  (≤ 2B params):  Pythia 160M / 410M / 1B / 1.4B at 7 checkpoints, OLMo-2 1B at 8 checkpoints,
                       final SmolLM2 360M / 1.7B and Qwen2.5 0.5B / 1.5B
  H100 (bigger):       Pythia 2.8B-deduped / 6.9B / 12B at 7 checkpoints, OLMo-2 7B at 8 checkpoints,
                       final OLMo-2 13B and Qwen2.5 7B

Pythia checkpoints are matched by tokens seen (≈2.1M tokens per step): 2B, 8B, 21B, 42B (≈ BubblSpace after
Phase 2.5a), 105B, 210B, 300B. OLMo-2 branches are picked as the nearest stage-1 checkpoint to each token
target, plus `main` (after the stage-2 anneal), plus the anneal itself: last stage-1 checkpoint, 4 points
along stage-2 ingredient 1, and the final checkpoints of ingredients 2 and 3.

    python scripts/binding_study.py --machine a10 [--dry_run]
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

PYTHIA_STEPS = [1000, 4000, 10000, 20000, 50000, 100000, 143000]
OLMO_TOKENS_B = [2, 8, 21, 42, 105, 420, 1000, 2000]
P = "benchmark/data/probefinding-winograndestyle"
PROBES = [f"v2={P}/probefinding-winogrande-20261121T185948.json",
          f"v3={P}/probefinding-winogrande-v3-20261122.json",
          f"v4={P}/probefinding-winogrande-v4-heldout-20261122.json",
          f"v5={P}/probefinding-winogrande-v5-heldout-structures-20261123.json"]

PLAN = {
    "a10": {"pythia": ["160m", "410m", "1b", "1.4b"], "olmo": ["allenai/OLMo-2-0425-1B"],
            "final": ["HuggingFaceTB/SmolLM2-360M", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]},
    # pythia-2.8b is excluded: its intermediate revisions on the Hub hold identical weights (checked 24 Nov 2026,
    # same .bin sha256 at steps 1000/4000/20000 and main; identical scores). The deduped run is used instead.
    "h100": {"pythia": ["2.8b-deduped", "6.9b", "12b"], "olmo": ["allenai/OLMo-2-1124-7B"],
             "final": ["allenai/OLMo-2-1124-13B", "Qwen/Qwen2.5-7B"]},
}


def olmo_revisions(repo):
    from huggingface_hub import HfApi
    branches = [b.name for b in HfApi().list_repo_refs(repo).branches]
    st1 = []
    for b in branches:
        m = re.match(r"stage1-step(\d+)-tokens(\d+)B$", b)
        if m:
            st1.append((int(m.group(2)), b))
    picks = []
    for t in OLMO_TOKENS_B:
        tok, b = min(st1, key=lambda x: abs(x[0] - t))
        if b not in picks:
            picks.append(b)
    return picks + ["main"]


def olmo_anneal_revisions(repo):
    """Before/after the anneal: last stage-1 checkpoint, 4 points along stage-2 ingredient 1 (first, 1/4, 1/2, last),
    and the last checkpoint of ingredients 2 and 3 (OLMo-2 `main` averages the ingredient runs)."""
    from huggingface_hub import HfApi
    branches = [b.name for b in HfApi().list_repo_refs(repo).branches]
    st1 = max((int(m.group(1)), b) for b in branches if (m := re.match(r"stage1-step(\d+)-tokens\d+B$", b)))[1]
    ing = {}
    for b in branches:
        m = re.match(r"stage2-(ingredient\d+)-step(\d+)-tokens\d+B$", b)
        if m:
            ing.setdefault(m.group(1), []).append((int(m.group(2)), b))
    out = [st1]
    for k in sorted(ing):
        v = sorted(ing[k])
        if k == "ingredient1":
            idx = sorted({0, len(v) // 4, len(v) // 2, len(v) - 1})
            out += [v[i][1] for i in idx]
        else:
            out.append(v[-1][1])
    return out


def jobs(machine):
    plan = PLAN[machine]
    out = []
    for size in plan["pythia"]:
        for s in PYTHIA_STEPS:
            out.append((f"EleutherAI/pythia-{size}", f"step{s}"))
    for repo in plan["olmo"]:
        out += [(repo, rev) for rev in olmo_revisions(repo)]
    out += [(m, None) for m in plan["final"]]
    for repo in plan["olmo"]:  # added 24 Nov: anneal checkpoints (appended, so earlier jobs keep their order)
        out += [(repo, rev) for rev in olmo_anneal_revisions(repo)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=list(PLAN))
    ap.add_argument("--out_dir", default="benchmark/binding_levels")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    js = jobs(a.machine)
    log = Path(a.out_dir) / f"study_{a.machine}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for i, (model, rev) in enumerate(js):
        tag = model.split("/")[-1] + (f"_{rev}" if rev else "_main")
        out = Path(a.out_dir) / f"{tag}.json"
        print(f"[{i + 1}/{len(js)}] {model} {rev or 'main'} -> {out}" + (" (done)" if out.exists() else ""), flush=True)
        if a.dry_run or out.exists():
            continue
        cmd = [sys.executable, "scripts/binding_levels_hf.py", "--model", model, "--probes", *PROBES,
               "--winogrande", "--out", str(out)] + (["--revision", rev] if rev else [])
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if not l.startswith("Loading weights"))[-1200:]
        with open(log, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {model} {rev} rc={r.returncode} {time.time() - t0:.0f}s\n{tail}\n\n")
    print("study finished", flush=True)


if __name__ == "__main__":
    main()
