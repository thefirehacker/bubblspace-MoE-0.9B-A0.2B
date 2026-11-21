"""Per-checkpoint Phase 2 eval suite (see doc/eval-plan-phase2.md), ~10-15 min on the A10.

All examples come from the HELD-OUT tail of each pool -- the exact slice
MultiPoolTokenDataset excludes from training (same _Pool split, same
val_tokens_per_pool as configs/phase2.yaml), so nothing here was trained on.

  fim        single-line infill exact match on unseen code (PSM prompt, greedy to EOL)
  agentic    tool-call turns from held-out tool-use episodes: emits <tool_call>, closes
             it, valid JSON with name+arguments, name present in the episode's <tools>,
             name == gold, argument keys == gold, ends the turn with <|im_end|>
  reasoning  held-out reasoning episodes: opens <think>, closes </think> within budget
  greedy     fixed prompts (binding, facts, code) saved verbatim for side-by-side review

Usage:
    python scripts/eval_phase2_quick.py --checkpoint ckpt.pt --pools_yaml configs/pools_a10.yaml \
        --out benchmark/phase2/quick_step1000.json
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import _Pool
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

GREEDY_PROMPTS = [
    "The marathon took longer to finish than the sprint because the marathon was",
    "The marathon took longer to finish than the sprint because the sprint was",
    "The wardrobe would not go through the doorway because the doorway was",
    "The wardrobe would not go through the doorway because the wardrobe was",
    "The capital of India is",
    "The capital of the United States is",
    "The Riemann hypothesis is a mathematical conjecture about",
    "In Sanskrit, the word Kama means",
    "def fibonacci(n):\n",
    "import json\n\ndef load_config(path):\n",
    "SELECT name, email FROM users WHERE",
    "<|im_start|>system\n<tools>\n[{\"name\": \"get_weather\", \"description\": \"Get current weather for a city\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}]\n</tools>\n<|im_end|>\n<|im_start|>user\nWhat's the weather in Paris right now?\n<|im_end|>\n<|im_start|>assistant\n",
]

NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


class Gen:
    def __init__(self, model, tok, device, max_seq_len):
        self.model, self.tok, self.device, self.max_seq_len = model, tok, device, max_seq_len
        self.rv = len(tok)

    @torch.no_grad()
    def greedy(self, ids, max_new, stop_ids=(), stop_text=None, use_cache=True):
        """Greedy decode. use_cache=True: one prefill pass over the prompt, then
        one token per step against the KV cache (O(n) instead of O(n^2) work).
        use_cache=False re-runs the full sequence each step (reference path)."""
        ids = list(ids)[-(self.max_seq_len - max_new):]
        out = []
        cache = self.model.new_kv_cache() if use_cache else None
        step_input = ids
        for _ in range(max_new):
            x = torch.tensor([step_input if use_cache else ids + out], device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, *_ = self.model(x, kv_cache=cache)
            nxt = int(logits[0, -1, : self.rv].argmax())
            if nxt in stop_ids:
                return out, nxt
            out.append(nxt)
            step_input = [nxt]
            if stop_text is not None and stop_text in self.tok.decode(out):
                return out, None
        return out, None


def heldout_docs(pool: _Pool, tok, eot_id, max_windows=64):
    """Decode the held-out tail into individual documents (split on EOT)."""
    shard = pool.shards[-1]
    base, L = int(pool.train_lengths[-1]), pool.seq_len
    docs, n = [], 0
    for s in range(base, base + pool.val_tokens - L, L):
        if n >= max_windows:
            break
        w = shard[s : s + L].astype(np.int64).tolist()
        cur = []
        for t in w:
            if t == eot_id:
                if cur:
                    docs.append(cur)
                cur = []
            else:
                cur.append(t)
        n += 1
    return docs


def eval_fim(gen, pools, tok, ids, n_examples, rng):
    pre, mid, suf, eot = ids["pre"], ids["mid"], ids["suf"], ids["eot"]
    cands = []
    for p in pools:
        for d in heldout_docs(p, tok, eot, max_windows=24):
            text = tok.decode(d)
            lines = text.split("\n")
            if len(lines) < 8:
                continue
            ok = [i for i in range(2, len(lines) - 2) if 8 <= len(lines[i].strip()) <= 100]
            if ok:
                cands.append((p.name, lines, rng.choice(ok)))
    rng.shuffle(cands)
    results = []
    for pool_name, lines, i in cands[:n_examples]:
        prefix = "\n".join(lines[:i]) + "\n"
        suffix = "\n" + "\n".join(lines[i + 1 :])
        gold = lines[i]
        prompt = [pre] + tok.encode(prefix, add_special_tokens=False)[-1024:] + [suf] + tok.encode(suffix, add_special_tokens=False)[:384] + [mid]
        out, _ = gen.greedy(prompt, 64, stop_ids=(eot, ids["pad"]), stop_text="\n")
        pred = tok.decode(out).split("\n")[0]
        results.append({"pool": pool_name, "gold": gold, "pred": pred, "exact": pred.strip() == gold.strip()})
    em = float(np.mean([r["exact"] for r in results])) if results else 0.0
    return {"n": len(results), "exact_match": em, "examples": results[:20]}


def parse_call(text):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {"name": obj["name"], "args": args if isinstance(args, dict) else {}}


def eval_agentic(gen, pools, tok, ids, n_examples):
    header = "<|im_start|>assistant\n"
    cases = []
    for p in pools:
        per_pool = []
        for d in heldout_docs(p, tok, ids["eot"], max_windows=64):
            ep = tok.decode([t for t in d if t != ids["pad"]])
            pos = 0
            while True:
                k = ep.find(header, pos)
                if k < 0:
                    break
                start = k + len(header)
                end = ep.find("<|im_end|>", start)
                gold_turn = ep[start:end if end > 0 else None]
                if "<tool_call>" in gold_turn:
                    gold = parse_call(gold_turn)
                    if gold:
                        tools_m = re.search(r"<tools>(.*?)</tools>", ep[:start], re.S)
                        tool_names = set(NAME_RE.findall(tools_m.group(1))) if tools_m else set()
                        per_pool.append((p.name, ep[:start], gold, tool_names))
                        break  # first tool-call turn per episode
                pos = start
        cases.extend(per_pool[: max(1, n_examples // len(pools))])
    res = []
    for pool_name, prompt_text, gold, tool_names in cases:
        prompt = tok.encode(prompt_text, add_special_tokens=False)
        if len(prompt) > 1800:
            continue
        out, stop = gen.greedy(prompt, 200, stop_ids=(ids["im_end"], ids["eot"]))
        text = tok.decode(out)
        call = parse_call(text)
        res.append({
            "pool": pool_name,
            "emits_tool_call": "<tool_call>" in text,
            "closes_tool_call": "</tool_call>" in text,
            "valid_json_call": call is not None,
            "name_in_tools": bool(call and tool_names and call["name"] in tool_names),
            "name_eq_gold": bool(call and call["name"] == gold["name"]),
            "argkeys_eq_gold": bool(call and set(call["args"]) == set(gold["args"])),
            "ends_turn": stop == ids["im_end"],
            "gold": gold,
            "pred_text": text[:400],
        })
    keys = ["emits_tool_call", "closes_tool_call", "valid_json_call", "name_in_tools", "name_eq_gold", "argkeys_eq_gold", "ends_turn"]
    summary = {k: float(np.mean([r[k] for r in res])) if res else 0.0 for k in keys}
    return {"n": len(res), **summary, "examples": res[:12]}


def eval_reasoning(gen, pools, tok, ids, n_examples, budget):
    header = "<|im_start|>assistant\n"
    prompts = []
    for p in pools:
        for d in heldout_docs(p, tok, ids["eot"], max_windows=32):
            ep = tok.decode([t for t in d if t != ids["pad"]])
            k = ep.find(header)
            if k >= 0 and "<think>" in ep[k:]:
                prompts.append((p.name, ep[: k + len(header)]))
    prompts = prompts[:n_examples]
    res = []
    for pool_name, ptxt in prompts:
        ids_p = tok.encode(ptxt, add_special_tokens=False)
        if len(ids_p) > 2048 - budget - 8:
            continue
        out, stop = gen.greedy(ids_p, budget, stop_ids=(ids["im_end"], ids["eot"]))
        text = tok.decode(out)
        res.append({"pool": pool_name, "opens_think": text.lstrip().startswith("<think>") or "<think>" in text[:40],
                    "closes_think": "</think>" in text, "n_tokens": len(out), "head": text[:300]})
    summary = {k: float(np.mean([r[k] for r in res])) if res else 0.0 for k in ("opens_think", "closes_think")}
    return {"n": len(res), "budget_tokens": budget, **summary, "examples": res[:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--phase2_config", default="configs/phase2.yaml", help="for val_tokens_per_pool (held-out split)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pools_yaml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fim_n", type=int, default=60)
    ap.add_argument("--agentic_n", type=int, default=45)
    ap.add_argument("--reasoning_n", type=int, default=8)
    ap.add_argument("--reasoning_budget", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    val_tokens = int(yaml.safe_load(open(args.phase2_config))["phase2"]["val_tokens_per_pool"])
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    ids = {
        "eot": tok.eos_token_id,
        "pad": tok.convert_tokens_to_ids("<|fim_pad|>"),
        "pre": tok.convert_tokens_to_ids("<|fim_prefix|>"),
        "mid": tok.convert_tokens_to_ids("<|fim_middle|>"),
        "suf": tok.convert_tokens_to_ids("<|fim_suffix|>"),
        "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
    }
    model = Transformer(cfg.model).to("cuda").eval()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    step = ckpt.get("step")
    gen = Gen(model, tok, "cuda", cfg.model.max_seq_len)
    rng = random.Random(args.seed)

    pools_cfg = yaml.safe_load(open(args.pools_yaml))["pools"]
    pools = {n: _Pool(n, s, cfg.training.seq_len, val_tokens) for n, s in pools_cfg.items()}
    code = [p for p in pools.values() if p.group == "code"]
    tool = [pools[n] for n in ("agentic_glaive_fc_v2", "agentic_hermes", "agentic_toolmind") if n in pools]
    reas = [pools[n] for n in ("agentic_openthoughts", "agentic_bespoke_stratos") if n in pools]

    result = {"checkpoint": args.checkpoint, "step": step}
    result["fim"] = eval_fim(gen, code, tok, ids, args.fim_n, rng)
    print(f"[fim] n={result['fim']['n']} exact_match={result['fim']['exact_match']:.3f}")
    result["agentic"] = eval_agentic(gen, tool, tok, ids, args.agentic_n)
    a = result["agentic"]
    print(f"[agentic] n={a['n']} emits={a['emits_tool_call']:.2f} closes={a['closes_tool_call']:.2f} json={a['valid_json_call']:.2f} "
          f"name_in_tools={a['name_in_tools']:.2f} name=gold={a['name_eq_gold']:.2f} argkeys=gold={a['argkeys_eq_gold']:.2f} ends_turn={a['ends_turn']:.2f}")
    result["reasoning"] = eval_reasoning(gen, reas, tok, ids, args.reasoning_n, args.reasoning_budget)
    r = result["reasoning"]
    print(f"[reasoning] n={r['n']} opens_think={r['opens_think']:.2f} closes_think(<= {r['budget_tokens']} tok)={r['closes_think']:.2f}")
    result["greedy"] = []
    for p in GREEDY_PROMPTS:
        out, _ = gen.greedy(tok.encode(p, add_special_tokens=False), 60, stop_ids=(ids["eot"], ids["im_end"]))
        result["greedy"].append({"prompt": p, "completion": tok.decode(out)})
    result["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[quick] step {step} done in {result['elapsed_s']}s -> {args.out}")


if __name__ == "__main__":
    main()
