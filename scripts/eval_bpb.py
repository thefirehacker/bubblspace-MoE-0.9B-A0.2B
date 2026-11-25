"""Bits per byte (BPB) on post-cutoff text: UncheatableEval-2026-10 (doc/plan-05-bits-per-byte-eval.md).

Text published in Oct 2026 can't be in any BubblSpace training data (newest source: 2025), so this is a
contamination-free, tokenizer-independent compression score. The same script scores our checkpoints and any HF model.

Scoring follows UncheatableEval's own evaluator (github.com/Jellyfish042/uncheatable_eval, evaluator.py) so numbers line up:
  - text = the dataset's `content` column (the truncated version their scorer uses)
  - each document is tokenized without special tokens and split into NON-overlapping chunks; every chunk gets a start
    token prepended (ours: <|endoftext|>, the doc separator we train with; HF: BOS, else EOS) and all chunk tokens are scored
  - per category: BPB = sum of -log2 p over all tokens / sum of UTF-8 bytes  (byte-weighted, as theirs)
  - also reported: their "compression rate %" = BPB / 8 * 100, and the mean of per-document BPB
Differences, kept the same for EVERY model so comparisons stay fair: chunk = 2,047 tokens + start token (our context is 2,048;
theirs defaults to 4,000), and log-softmax in fp32 (theirs: bf16 cross-entropy).

A fixed, category-stratified sample (default 2,000 of 7,500 docs, seed 0) keeps the A10 cost low; its row indices are
written to --sample_file on first use and reused afterwards, so every model is scored on identical documents.

    python scripts/eval_bpb.py --checkpoint step_N.pt --data ~/eval_data/uncheatable-2026-10.parquet --out bpb_stepN.json
    python scripts/eval_bpb.py --hf_model HuggingFaceTB/SmolLM2-360M --data ... --out bpb_smollm2-360m.json
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LN2 = math.log(2.0)


def stratified_sample(cats, n, seed):
    by = defaultdict(list)
    for i, c in enumerate(cats):
        by[c].append(i)
    rng = random.Random(seed)
    total = len(cats)
    out = []
    for c in sorted(by):
        k = max(1, round(n * len(by[c]) / total))
        out += sorted(rng.sample(by[c], min(k, len(by[c]))))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="BubblSpace step_N.pt / .safetensors")
    src.add_argument("--hf_model", help="Hugging Face model id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--config", default="configs/base.yaml", help="BubblSpace config (model shape, tokenizer)")
    ap.add_argument("--data", required=True, help="UncheatableEval parquet (ClamAV-scanned copy)")
    ap.add_argument("--n_docs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_file", default="benchmark/bpb/sample_uncheatable-2026-10_n2000_seed0.json")
    ap.add_argument("--chunk", type=int, default=2047, help="tokens per chunk, plus one start token")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    t0 = time.time()
    tab = pq.read_table(a.data, columns=["content", "category", "date"]).to_pydict()
    sp = Path(a.sample_file)
    if sp.exists():
        idx = json.loads(sp.read_text())["indices"]
    else:
        idx = stratified_sample(tab["category"], a.n_docs, a.seed)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"data": Path(a.data).name, "n_docs": len(idx), "seed": a.seed, "indices": idx}))

    torch.manual_seed(0)
    if a.checkpoint:
        from src.eval.determinism import make_deterministic
        make_deterministic()
        from binding_levels_bubbl import BubblLM
        lm = BubblLM(a.config, a.checkpoint)
        start = lm.eos  # <|endoftext|>: the separator every training document follows
        name, extra = "bubblspace-moe0.9b-a0.2b", {"checkpoint": a.checkpoint, "step": lm.step}
    else:
        import binding_levels_hf as bl
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        lm = bl.LM(a.hf_model, a.revision, str(Path("~/hf_bpb_cache").expanduser()), dtype)
        start = lm.tok.bos_token_id if lm.tok.bos_token_id is not None else lm.eos
        name, extra = a.hf_model, {"revision": a.revision, "weight_sha256": bl.WEIGHT_SHA}
    assert start is not None, "no start token"

    per_cat = defaultdict(lambda: {"bits": 0.0, "bytes": 0, "tokens": 0, "docs": 0, "doc_bpb": []})
    for n, i in enumerate(idx):
        text, cat = tab["content"][i], tab["category"][i]
        ids = lm.cont_ids(text)
        nll = 0.0
        for b in range(0, len(ids), a.chunk):
            nll -= lm.logprob([start], ids[b : b + a.chunk])
        nbytes = len(text.encode("utf-8"))
        c = per_cat[cat]
        c["bits"] += nll / LN2
        c["bytes"] += nbytes
        c["tokens"] += len(ids)
        c["docs"] += 1
        c["doc_bpb"].append(nll / LN2 / nbytes)
        if (n + 1) % 200 == 0:
            print(f"{n + 1}/{len(idx)} docs  {time.time() - t0:.0f}s", flush=True)

    def row(c):
        bpb = c["bits"] / c["bytes"]
        return {"bpb": round(bpb, 4), "compression_rate_pct": round(bpb / 8 * 100, 3),
                "mean_doc_bpb": round(sum(c["doc_bpb"]) / len(c["doc_bpb"]), 4),
                "docs": c["docs"], "bytes": c["bytes"], "tokens": c["tokens"],
                "bytes_per_token": round(c["bytes"] / c["tokens"], 3)}

    allc = {"bits": sum(c["bits"] for c in per_cat.values()), "bytes": sum(c["bytes"] for c in per_cat.values()),
            "tokens": sum(c["tokens"] for c in per_cat.values()), "docs": sum(c["docs"] for c in per_cat.values()),
            "doc_bpb": [x for c in per_cat.values() for x in c["doc_bpb"]]}
    cats = {k: row(v) for k, v in sorted(per_cat.items())}
    res = {"model": name, **extra, "data": Path(a.data).name, "sample_file": a.sample_file, "chunk_tokens": a.chunk + 1,
           "overall": row(allc),
           # unweighted mean over categories, the closest analogue of UncheatableEval's per-dataset average
           "mean_over_categories_bpb": round(sum(v["bpb"] for v in cats.values()) / len(cats), 4),
           # BubblSpace is trained on English only: also report without the two *_nonenglish categories
           "mean_over_english_categories_bpb": round(
               sum(v["bpb"] for k, v in cats.items() if "nonenglish" not in k)
               / max(1, sum("nonenglish" not in k for k in cats)), 4),
           "by_category": cats, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "elapsed_s": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("overall", "mean_over_categories_bpb", "mean_over_english_categories_bpb")}, indent=1))
    print({k: v["bpb"] for k, v in cats.items()})
    print(f"wrote {a.out} ({res['elapsed_s']} s)")
    if a.hf_model:
        import shutil
        shutil.rmtree(Path("~/hf_bpb_cache").expanduser(), ignore_errors=True)


if __name__ == "__main__":
    main()
