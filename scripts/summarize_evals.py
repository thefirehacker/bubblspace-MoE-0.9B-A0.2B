"""Collect per-checkpoint eval JSONs into one markdown table (one row per step).

Reads the files written by scripts/eval_watch_25a.sh (and the Phase 2 watcher), e.g.
  probe_v4_<prefix><step>.json, facts_..., closedbook_triviaqa1k_..., english6_..., codeexec_..., milestone_...
Missing files show as "—". Optional --baseline adds reference rows (e.g. Phase 2 final) from another dir.

  python scripts/summarize_evals.py --dir benchmark/phase25a --prefix phase25a_step --out benchmark/phase25a/summary.md \
      --baseline benchmark/phase2:phase2_step22888
"""
import argparse
import glob
import json
import os
import re

ENG6 = [("hellaswag", "acc_norm,none"), ("piqa", "acc_norm,none"), ("arc_easy", "acc_norm,none"),
        ("arc_challenge", "acc_norm,none"), ("winogrande", "acc,none"), ("lambada_openai", "acc,none")]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def row_for(d, tag, long_context=False):
    """Return dict of column -> string for checkpoint tag (e.g. 'phase25a_step2000')."""
    j = lambda name: load(os.path.join(d, f"{name}_{tag}.json"))
    r = {}
    if long_context:
        lc = j("long_context")
        for L in ("4096", "16384"):
            pk = lc["passkey"]["by_len"].get(L) if lc and "passkey" in lc else None
            mh = lc["multihop"]["2hop_by_len"].get(L) if lc and "multihop" in lc else None
            r[f"passkey {int(L) // 1024}k"] = f"{100 * pk[0]:.0f}%" if pk and pk[0] is not None else "—"
            r[f"2-hop {int(L) // 1024}k"] = f"{100 * mh[0]:.0f}%" if mh and mh[0] is not None else "—"
        pl = lc["posloss"]["buckets"] if lc and "posloss" in lc else None
        r["loss 1-2k / 8-16k"] = f"{pl['1024-2048']:.2f} / {pl['8192-16384']:.2f}" if pl else "—"
    for v in (2, 3, 4, 5):
        p = j(f"probe_v{v}")
        r[f"probe v{v}"] = f"{p['n_correct']}/{p['n_total']}" if p else "—"
    for v in (4, 5):
        g = j(f"gen_v{v}")
        r[f"gen flip v{v}"] = f"{g['summary']['flip_hit']}/{g['summary']['n_cases']}" if g else "—"
    f = j("facts")
    r["facts"] = f"{f['summary']['correct']}/{f['summary']['n']}" if f else "—"
    for task, short in (("triviaqa", "TriviaQA"), ("nq", "NQ")):
        full, k1 = j(f"closedbook_{task}"), j(f"closedbook_{task}1k")
        src, lab = (full, "") if full else (k1, " (1k)")
        if src:
            res = next(iter(src["results"].values()))
            em = next(v for k, v in res.items() if k.startswith("exact_match,"))
            r[short] = f"{100 * em:.1f}%{lab}"
        else:
            r[short] = "—"
    e = j("english6")
    if e:
        vals = [e["results"][t][m] for t, m in ENG6 if t in e["results"]]
        r["English 6 avg"] = f"{100 * sum(vals) / len(vals):.2f}" if len(vals) == 6 else "—"
    else:
        r["English 6 avg"] = "—"
    q = j("quick")
    r["tool-call name"] = f"{100 * q['agentic']['name_eq_gold']:.0f}%" if q else "—"
    r["FIM"] = f"{100 * q['fim']['exact_match']:.1f}%" if q else "—"
    c = j("codeexec")
    pa = lambda t: f"{100 * c['summary'][t]['pass_at_1']:.1f}%" if c and t in c["summary"] else "—"
    r["HumanEval / MBPP"] = f"{pa('humaneval')} / {pa('mbpp')}"
    m = j("milestone")
    if m:
        b = m["bfcl"]
        r["CRUXEval-O"] = f"{100 * m['cruxeval_o']['pass_at_1']:.1f}%"
        r["BFCL s/m/p"] = f"{100 * b['simple']['ast_accuracy']:.0f}/{100 * b['multiple']['ast_accuracy']:.0f}/{100 * b['parallel']['ast_accuracy']:.0f}"
    else:
        r["CRUXEval-O"] = r["BFCL s/m/p"] = "—"
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", required=True, help="tag prefix before the step number, e.g. phase25a_step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", action="append", default=[], help="dir:tag reference rows, e.g. benchmark/phase2:phase2_step22888")
    ap.add_argument("--long_context", action="store_true", help="add passkey / multi-hop / position-loss columns (2.5b)")
    a = ap.parse_args()

    steps = set()
    for p in glob.glob(os.path.join(a.dir, f"*_{a.prefix}*.json")):
        m = re.search(re.escape(a.prefix) + r"(\d+)\.json$", p)
        if m:
            steps.add(int(m.group(1)))
    rows = [(f"ref: {tag}", row_for(d, tag, a.long_context)) for d, tag in (b.split(":", 1) for b in a.baseline)]
    rows += [(f"{s:,}", row_for(a.dir, f"{a.prefix}{s}", a.long_context)) for s in sorted(steps)]
    if not rows:
        open(a.out, "w").write("No results yet.\n")
        return
    cols = list(rows[0][1].keys())
    lines = ["| step | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    lines += ["| " + name + " | " + " | ".join(r[c] for c in cols) + " |" for name, r in rows]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
