"""Collect binding_levels_hf.py results into one markdown table + CSV, ordered by model family, size and tokens.

    python scripts/binding_study_report.py --dir benchmark/binding_levels --out benchmark/binding_levels/report
"""

import argparse
import csv
import glob
import json
import os
import re

PYTHIA_TOK_PER_STEP = 2_097_152


def tokens_b(model, rev):
    if rev and rev.startswith("step") and "pythia" in model:
        return int(rev[4:]) * PYTHIA_TOK_PER_STEP / 1e9
    m = re.search(r"tokens(\d+)B", rev or "")
    if m:
        return float(m.group(1))
    return None  # final / main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="benchmark/binding_levels")
    ap.add_argument("--out", default="benchmark/binding_levels/report")
    a = ap.parse_args()
    rows = []
    for p in glob.glob(os.path.join(a.dir, "*.json")):
        d = json.load(open(p))
        if "suites" not in d:
            continue
        r = {"model": d["model"].split("/")[-1], "revision": d["revision"] or "main",
             "tokens_B": tokens_b(d["model"], d["revision"]), "params_M": round(d["n_params"] / 1e6), "gpu": d.get("gpu", "")}
        for suite in ("v2", "v3", "v4", "v5", "winogrande"):
            s = d["suites"].get(suite)
            for lvl in ("L1", "L2", "L3"):
                r[f"{suite}_{lvl}"] = s[lvl] if s else None
        wg = d["suites"].get("winogrande")
        r["wg_answered_with_option"] = wg["answered_with_option"] if wg else None
        r["_sha"] = "|".join(sorted(d.get("weight_sha256", {}).values()))
        rows.append(r)
    seen = {}
    for r in rows:  # flag different revisions of one model that load identical weights
        if r["_sha"]:
            seen.setdefault((r["model"], r["_sha"]), []).append(r["revision"])
    for r in rows:
        dup = seen.get((r["model"], r["_sha"]), [])
        r["duplicate_weights_of"] = ",".join(x for x in dup if x != r["revision"]) if r["_sha"] and len(dup) > 1 else ""
        del r["_sha"]
    fam = lambda m: re.sub(r"[-_]?\d+(\.\d+)?[mMbB]$", "", m.split("_")[0])
    rows.sort(key=lambda r: (fam(r["model"]), r["params_M"], r["tokens_B"] if r["tokens_B"] is not None else 1e9))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}"
    lines = ["| model | checkpoint | tokens (B) | v4 L1 / L2 / L3 | v5 L1 / L2 / L3 | WinoGrande L1 / L2 / L3 | WG answers in format |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = "final" if r["tokens_B"] is None else f"{r['tokens_B']:.0f}"
        trip = lambda s: " / ".join(pct(r[f"{s}_{l}"]) for l in ("L1", "L2", "L3"))
        flag = f" ⚠ same weights as {r['duplicate_weights_of']}" if r["duplicate_weights_of"] else ""
        lines.append(f"| {r['model']} | {r['revision']}{flag} | {t} | {trip('v4')} | {trip('v5')} | {trip('winogrande')} | {pct(r['wg_answered_with_option'])} |")
    note = ("\nAll numbers in %. L1 = single row (chance 50), L2 = both halves of the pair (chance 25), "
            "L3 = model writes the right answer in both halves. WinoGrande L1 here is over twin pairs only.\n")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n" + note)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
