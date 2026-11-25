"""Long-context evals for Phase 2.5b (doc/plan-03-phase-2.5-knowledge-context.md §3).

Three tests at context lengths 2k / 4k / 8k / 16k, on real held-out text as the haystack:

  passkey   One sentence with a 5-digit passkey is hidden at depth 10/30/50/70/90% of the context.
            The prompt ends with "What is the pass key? The pass key is" and greedy decoding must
            produce the number. Pure retrieval.
  multihop  Two facts placed far apart, e.g. "Person lives in Town" near the start and "The bakery in
            Town is called Bakery" near the end, plus 3 distractor people/towns/bakeries elsewhere.
            The question asks for the bakery in the town where Person lives, so the model has to combine
            both facts (chance among the 4 listed bakeries = 25%). A 1-hop control asks directly for the
            bakery in Town, which separates "can't find it" from "can't combine it". All names are
            invented syllable strings, so prior knowledge can't help.
  posloss   Mean next-token loss by position bucket (0-512 ... 8k-16k) on held-out long documents.
            After 2.5b it should keep falling past 2,048; a model that can't use long context goes flat
            or rises there.

Needle tests alone would overstate how well the model *uses* long context; multi-hop is the main metric.

Step 1 (CPU, where the long pools live, e.g. fire01): cut held-out documents from the end of each pool's
last shard (the region train.py holds out as validation) into a small .npz:
    python scripts/eval_long_context.py make_docs --pools /lambda/nfs/h100-f1/data/phase25b/long_wiki \
        /lambda/nfs/h100-f1/data/phase25b/long_fineweb /lambda/nfs/h100-f1/data/phase25b/long_code \
        --out benchmark/data/long-context/long_docs_heldout.npz

Step 2 (GPU):
    python scripts/eval_long_context.py eval --config configs/phase25b.yaml --checkpoint step_N.pt \
        --docs benchmark/data/long-context/long_docs_heldout.npz --output long_context_stepN.json
For a checkpoint trained at 2k (baseline), keep its RoPE theta and only raise the cache length:
    ... eval --config configs/base.yaml --max_seq_len 16384 ...
Deterministic: fixed seed, greedy decoding.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LENGTHS = [2048, 4096, 8192, 16384]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
HOP_DEPTHS = [(0.1, 0.6), (0.4, 0.9), (0.8, 0.2)]  # (person fact, bakery fact); last one is reversed order
BUCKETS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384)]
GEN_TOKENS = 12
HELD_OUT_TOKENS = 2_097_152  # = phase2.val_tokens_per_pool, the tail train.py never trains on


# ---------------------------------------------------------------- make_docs (CPU only)
def make_docs(a):
    docs, srcs = [], []
    for pool in a.pools:
        # same shard order and hold-out rule as src/data/dataset.py (_Pool): recursive sorted glob,
        # tail of the last shard = min(val_tokens, len // 4 // seq_len * seq_len)
        metas = sorted(Path(pool).glob("**/meta.json"))
        eot = int(json.loads(metas[0].read_text()).get("eot_token", 0)) if metas else 0
        last = sorted(Path(pool).glob("**/shard_*.bin"))[-1]
        arr = np.memmap(last, dtype=np.uint16, mode="r")
        held = min(HELD_OUT_TOKENS, len(arr) // 4 // a.seq_len * a.seq_len)
        tail = np.asarray(arr[-held:])
        cuts = np.flatnonzero(tail == eot)
        n_pool = 0
        for s, e in zip(cuts[:-1], cuts[1:]):  # only complete docs (bounded by EOT on both sides)
            d = tail[s + 1:e]
            if len(d) >= a.min_tokens:
                docs.append(d[: a.max_tokens].astype(np.uint16))
                srcs.append(os.path.basename(pool.rstrip("/")))
                n_pool += 1
                if n_pool >= a.per_pool:
                    break
        print(f"{pool}: {n_pool} held-out docs >= {a.min_tokens} tokens")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(docs), dtype=object)  # 1-D ragged array (np.array would make 2-D for equal lengths)
    for i, d in enumerate(docs):
        arr[i] = d
    np.savez_compressed(a.out, docs=arr, sources=np.array(srcs))
    print(f"wrote {a.out}: {len(docs)} docs, {sum(map(len, docs)):,} tokens")


# ---------------------------------------------------------------- eval helpers
SYL = ["ka", "lo", "mi", "ra", "ve", "to", "zu", "ne", "si", "pa", "dor", "len", "vat", "quin", "mar", "bel", "tis", "ros"]


def fake_name(rng, n=3):
    return "".join(rng.choice(SYL) for _ in range(n)).capitalize()


class Ctx:
    def __init__(self, a):
        import torch
        from transformers import AutoTokenizer
        from scripts.run_benchmarks import load_weights
        from src.config import load_config
        from src.eval.determinism import make_deterministic
        from src.model.moe import probe_grouped_mm
        from src.model.transformer import Transformer

        make_deterministic()
        self.torch = torch
        cfg = load_config(a.config)
        if a.max_seq_len:
            cfg.model.max_seq_len = a.max_seq_len
        if a.rope_theta:
            cfg.model.attention.rope_theta = a.rope_theta
        if cfg.model.moe.backend == "auto":
            cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
        self.cfg = cfg
        self.max_len = cfg.model.max_seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
        self.model = Transformer(cfg.model).to(self.device)
        self.step = load_weights(self.model, a.checkpoint)
        self.model.eval()
        self.real_vocab = len(self.tok)
        self.eot = self.tok.convert_tokens_to_ids(cfg.tokenizer.doc_separator)

    def enc(self, s):
        return self.tok.encode(s, add_special_tokens=False)

    def generate(self, ids, n=GEN_TOKENS):
        torch = self.torch
        cache = self.model.new_kv_cache()
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = []
        with torch.no_grad():
            for _ in range(n):
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
                    logits, _, _, _ = self.model(x, kv_cache=cache)
                nid = int(logits[0, -1, : self.real_vocab].argmax())
                if nid == self.eot:
                    break
                out.append(nid)
                x = torch.tensor([[nid]], dtype=torch.long, device=self.device)
        return self.tok.decode(out)

    def token_losses(self, ids):
        torch = self.torch
        x = torch.tensor([ids[:-1]], dtype=torch.long, device=self.device)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device)
        with torch.no_grad(), torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            logits, _, _, _ = self.model(x)
        logits = logits[0]
        out = []
        for i in range(0, logits.shape[0], 2048):  # chunked so the softmax temp stays small at 16k
            out.append(torch.nn.functional.cross_entropy(logits[i:i + 2048, : self.real_vocab].float(), y[i:i + 2048], reduction="none"))
        return torch.cat(out).cpu().numpy()


class Haystack:
    """Filler token stream from held-out docs (EOT-free), with sentence-boundary insertion points."""

    def __init__(self, ctx, docs):
        self.ctx = ctx
        self.stream = np.concatenate([d for d in docs]).astype(np.int64)
        self.stream = self.stream[self.stream != ctx.eot]
        self.boundary = {}

    def filler(self, rng, n):
        start = rng.randrange(0, len(self.stream) - n - 1)
        return self.stream[start:start + n].tolist()

    def snap(self, ids, pos):
        """Move pos to just after the nearest token ending a sentence ('.', '!', '?', newline)."""
        for off in range(0, 300):
            for p in (pos + off, pos - off):
                if 0 < p < len(ids):
                    t = ids[p - 1]
                    if t not in self.boundary:
                        s = self.ctx.tok.decode([t])
                        self.boundary[t] = bool(re.search(r"[.!?\n]\s*$", s))
                    if self.boundary[t]:
                        return p
        return pos


def build(ctx, hay, rng, total_len, inserts, question):
    """inserts: list of (depth, text). Returns token ids of exactly <= total_len tokens."""
    ins = [(d, ctx.enc(" " + t.strip() + " ")) for d, t in inserts]
    q = ctx.enc("\n\n" + question)
    n_fill = total_len - len(q) - sum(len(t) for _, t in ins)
    ids = hay.filler(rng, n_fill)
    for d, t in sorted(ins, key=lambda x: -x[0]):  # insert deepest first so earlier positions don't shift
        p = hay.snap(ids, int(d * len(ids)))
        ids[p:p] = t
    return ids + q


def run_passkey(ctx, hay, rng, n_trials):
    rows = []
    for L in LENGTHS:
        if L > ctx.max_len:
            continue
        for d in DEPTHS:
            for _ in range(n_trials):
                key = str(rng.randrange(10000, 99999))
                needle = f"The pass key is {key}. Remember it. {key} is the pass key."
                ids = build(ctx, hay, rng, L - GEN_TOKENS, [(d, needle)],
                            "What is the pass key? The pass key is")
                out = ctx.generate(ids)
                rows.append({"len": L, "depth": d, "key": key, "out": out.strip()[:40], "correct": key in re.sub(r"\s", "", out)})
    return rows


def run_multihop(ctx, hay, rng, n_trials):
    rows = []
    for L in LENGTHS:
        if L > ctx.max_len:
            continue
        for d1, d2 in HOP_DEPTHS:
            for _ in range(n_trials):
                names = set()
                while len(names) < 12:
                    names.add(fake_name(rng))
                names = list(names)
                people, towns, bakeries = names[0:4], names[4:8], names[8:12]
                facts = [(d1, f"{people[0]} lives in the town of {towns[0]}."),
                         (d2, f"The only bakery in {towns[0]} is called {bakeries[0]}.")]
                for k in range(1, 4):  # distractors at random depths
                    facts.append((rng.uniform(0.05, 0.95), f"{people[k]} lives in the town of {towns[k]}."))
                    facts.append((rng.uniform(0.05, 0.95), f"The only bakery in {towns[k]} is called {bakeries[k]}."))
                for kind, q in (("2hop", f"Question: What is the bakery called in the town where {people[0]} lives?\nAnswer: The bakery is called"),
                                ("1hop", f"Question: What is the bakery in {towns[0]} called?\nAnswer: The bakery is called")):
                    ids = build(ctx, hay, rng, L - GEN_TOKENS, facts, q)
                    out = ctx.generate(ids)
                    listed = [b for b in bakeries if b.lower() in out.lower()]
                    rows.append({"len": L, "kind": kind, "d_person": d1, "d_bakery": d2, "gold": bakeries[0],
                                 "out": out.strip()[:40], "correct": bakeries[0].lower() in out.lower(),
                                 "picked_listed": bool(listed)})
    return rows


def run_posloss(ctx, docs):
    sums = {b: [0.0, 0] for b in BUCKETS}
    n_docs = 0
    for d in docs:
        ids = d[: ctx.max_len + 1].astype(np.int64).tolist()
        if len(ids) < 2:
            continue
        losses = ctx.token_losses(ids)  # losses[i] = loss predicting token i+1, i.e. at position i
        for lo, hi in BUCKETS:
            seg = losses[lo:hi]
            if len(seg):
                sums[(lo, hi)][0] += float(seg.sum())
                sums[(lo, hi)][1] += len(seg)
        n_docs += 1
    return {"n_docs": n_docs,
            "buckets": {f"{lo}-{hi}": (round(s / n, 4) if n else None) for (lo, hi), (s, n) in sums.items()},
            "tokens": {f"{lo}-{hi}": n for (lo, hi), (s, n) in sums.items()}}


def acc(rows, **match):
    sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return (round(sum(r["correct"] for r in sel) / len(sel), 3), len(sel)) if sel else (None, 0)


def evaluate(a):
    t0 = time.time()
    ctx = Ctx(a)
    z = np.load(a.docs, allow_pickle=True)
    docs = list(z["docs"])
    hay = Haystack(ctx, docs)
    rng = random.Random(a.seed)
    res = {"checkpoint": a.checkpoint, "step": ctx.step, "config": a.config, "max_seq_len": ctx.max_len,
           "rope_theta": ctx.cfg.model.attention.rope_theta, "seed": a.seed}
    lens = [L for L in LENGTHS if L <= ctx.max_len]
    if "passkey" in a.tests:
        rows = run_passkey(ctx, hay, rng, a.trials)
        res["passkey"] = {"by_len": {L: acc(rows, len=L) for L in lens},
                          "by_len_depth": {f"{L}@{d}": acc(rows, len=L, depth=d)[0] for L in lens for d in DEPTHS},
                          "rows": rows}
        print("passkey:", res["passkey"]["by_len"])
    if "multihop" in a.tests:
        rows = run_multihop(ctx, hay, rng, a.trials)
        res["multihop"] = {"2hop_by_len": {L: acc(rows, len=L, kind="2hop") for L in lens},
                           "1hop_by_len": {L: acc(rows, len=L, kind="1hop") for L in lens},
                           "2hop_picked_listed": round(np.mean([r["picked_listed"] for r in rows if r["kind"] == "2hop"]), 3),
                           "rows": rows}
        print("multihop 2-hop:", res["multihop"]["2hop_by_len"], " 1-hop:", res["multihop"]["1hop_by_len"])
    if "posloss" in a.tests:
        long_docs = [d for d in docs if len(d) > 4096]
        long_docs = long_docs[:: max(1, len(long_docs) // a.posloss_docs)][: a.posloss_docs]  # spread over all sources
        res["posloss"] = run_posloss(ctx, long_docs)
        print("posloss:", res["posloss"]["buckets"])
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {a.output} ({res['elapsed_s']} s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make_docs")
    m.add_argument("--pools", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--min_tokens", type=int, default=8192)
    m.add_argument("--max_tokens", type=int, default=16385)
    m.add_argument("--per_pool", type=int, default=40)
    m.add_argument("--seq_len", type=int, default=16384, help="training seq_len (sets the hold-out rounding)")
    e = sub.add_parser("eval")
    e.add_argument("--config", default="configs/phase25b.yaml")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--docs", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--tests", default="passkey,multihop,posloss")
    e.add_argument("--trials", type=int, default=8, help="trials per (length, depth) cell")
    e.add_argument("--posloss_docs", type=int, default=48)
    e.add_argument("--max_seq_len", type=int, default=None, help="override, e.g. 16384 for a 2k-trained baseline")
    e.add_argument("--rope_theta", type=float, default=None, help="override (normally leave as the config's)")
    e.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    make_docs(a) if a.cmd == "make_docs" else evaluate(a)


if __name__ == "__main__":
    main()
