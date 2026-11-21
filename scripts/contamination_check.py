"""BFCL vs. agentic-training-data contamination check (13-gram word overlap).

For every BFCL v3 simple/multiple/parallel question, build its set of
13-word shingles (lowercased, punctuation stripped). Stream the raw source
files of the agentic training pools (the same files scripts/prepare_agentic.py
downloaded) and flag any BFCL question sharing at least one 13-gram with the
training text. 13-grams follow the decontamination convention used for
GPT-3 / The Pile style checks.

Usage: python scripts/contamination_check.py --out benchmark/phase2/contamination_bfcl.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download

from scripts.prepare_agentic import HERMES_FILES, TOOLMIND_FILES

N = 13
WORD = re.compile(r"[a-z0-9]+")


def shingles(text):
    w = WORD.findall(text.lower())
    return {" ".join(w[i : i + N]) for i in range(len(w) - N + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    q_sh = {}
    for cat in ("simple", "multiple", "parallel"):
        for line in open(hf_hub_download(repo, f"BFCL_v3_{cat}.json", repo_type="dataset")):
            q = json.loads(line)
            turns = q["question"][0] if isinstance(q["question"][0], list) else q["question"]
            text = " ".join(t["content"] for t in turns)
            s = shingles(text)
            if s:
                q_sh[q["id"]] = s
    index = {}
    for qid, s in q_sh.items():
        for g in s:
            index.setdefault(g, set()).add(qid)
    print(f"[bfcl] {len(q_sh)} questions, {len(index)} distinct 13-grams")

    sources = [("Nanbeige/ToolMind", f) for f in TOOLMIND_FILES] + \
              [("NousResearch/hermes-function-calling-v1", f) for f in HERMES_FILES] + \
              [("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json")]
    hits = {}
    for repo_id, fname in sources:
        try:
            path = hf_hub_download(repo_id, fname, repo_type="dataset")
        except Exception as e:
            print(f"[skip] {repo_id}/{fname}: {e}")
            continue
        # fixed-size chunks with overlap (some sources are one giant JSON line);
        # 2,000 chars of overlap > any 13-word window, so no straddling n-gram is missed
        n_bytes, carry = 0, ""
        with open(path, errors="ignore") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                n_bytes += len(chunk)
                text = carry + chunk
                for g in shingles(text):
                    if g in index:
                        for qid in index[g]:
                            hits.setdefault(qid, set()).add(f"{repo_id}/{fname}")
                carry = text[-2000:]
        print(f"[scan] {repo_id}/{fname}: {n_bytes/1e6:,.0f} MB, contaminated questions so far: {len(hits)}")

    by_cat = {}
    for qid in q_sh:
        cat = qid.split("_")[0]
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        by_cat[cat][0] += qid in hits
    summary = {c: {"contaminated": a, "total": b, "rate": round(a / b, 4)} for c, (a, b) in by_cat.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"ngram": N, "by_category": summary,
                                          "contaminated_ids": {k: sorted(v) for k, v in hits.items()}}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
