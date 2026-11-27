"""Test the RL weight sync (src/rl/vllm_engine.py): does pushing weights into a running vLLM engine give exactly what
loading those weights from disk gives?

  --mode fresh   load model_dir B from disk, save greedy outputs + prompt log-probs  (run in its own process)
  --mode sync    load model_dir A from disk, then:
                   1. sync B's weights from a training-code Transformer (fp32 master weights, as the trainer holds them)
                      -> compare with the `fresh` results of B (should match exactly)
                   2. sleep (level 1) / wake / sync A again -> compare with A's own first outputs (round trip)
                   3. time the sync, and sampling with n=8 (log-probs returned per token)

    python scripts/vllm_sync_test.py --mode fresh --model_dir vllm_models/phase3-step1000 --out /tmp/fresh_1000.json
    python scripts/vllm_sync_test.py --mode sync --model_dir vllm_models/phase3-step1500 --other_weights .../step1000/model.safetensors \
        --config configs/phase3.yaml --fresh /tmp/fresh_1000.json --out benchmark/vllm/sync_test.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

PROMPTS = ["What is the capital of India?", "Write a Python function that reverses a string.",
           "Tom has 3 boxes with 8 apples in each box. He eats 5 apples. How many apples are left?",
           "Explain photosynthesis in two sentences.", "List three fruits, one per line, nothing else.",
           "Who painted the Mona Lisa?"]


def probe(eng):
    from vllm import SamplingParams
    chats = [eng.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    g = eng.llm.generate(chats, SamplingParams(max_tokens=96, temperature=0, stop_token_ids=eng.stop_ids,
                                               skip_special_tokens=False), use_tqdm=False)
    lp = eng.llm.generate(chats, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0), use_tqdm=False)
    return {"texts": [o.outputs[0].text for o in g],
            "prompt_logprobs": [[None if d is None else next(iter(d.values())).logprob for d in o.prompt_logprobs] for o in lp]}


def compare(a, b):
    diffs = [abs(x - y) for pa, pb in zip(a["prompt_logprobs"], b["prompt_logprobs"]) for x, y in zip(pa, pb) if x is not None]
    return {"texts_identical": sum(x == y for x, y in zip(a["texts"], b["texts"])), "n": len(a["texts"]),
            "max_abs_dlogprob": max(diffs), "mean_abs_dlogprob": sum(diffs) / len(diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fresh", "sync"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--other_weights")
    ap.add_argument("--config")
    ap.add_argument("--fresh")
    ap.add_argument("--gpu_mem", type=float, default=0.35)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.rl.vllm_engine import VLLMEngine
    eng = VLLMEngine(a.model_dir, gpu_memory_utilization=a.gpu_mem, max_model_len=4096)
    if a.mode == "fresh":
        Path(a.out).write_text(json.dumps(probe(eng)))
        print("[fresh] saved", a.out)
        return

    res = {}
    own = probe(eng)  # A from disk
    # the trainer's view of B: a training-code Transformer with fp32 weights on the GPU
    from safetensors.torch import load_file
    from src.config import load_config
    from src.model.transformer import Transformer
    cfg = load_config(a.config)
    cfg.model.moe.backend = "fallback"
    policy = Transformer(cfg.model).cuda()
    policy.load_state_dict(load_file(a.other_weights))
    t = eng.sync_weights(policy)
    res["sync_seconds_first"] = round(t, 2)
    res["synced_B_vs_fresh_B"] = compare(probe(eng), json.loads(Path(a.fresh).read_text()))
    print("[sync] B pushed in %.2fs; vs fresh B: %s" % (t, res["synced_B_vs_fresh_B"]), flush=True)

    # sleep (level 1) / wake round trip back to A
    free0 = torch.cuda.mem_get_info()[0]
    eng.sleep()
    free1 = torch.cuda.mem_get_info()[0]
    res["sleep_freed_GB"] = round((free1 - free0) / 1e9, 2)
    eng.wake()
    sd_a = load_file(str(Path(a.model_dir) / "model.safetensors"), device="cuda")
    t = eng.sync_weights(sd_a)
    res["sync_seconds_after_wake"] = round(t, 2)
    res["round_trip_A_vs_A"] = compare(probe(eng), own)
    print("[sync] sleep freed %.2f GB; A pushed back in %.2fs; vs original A: %s"
          % (res["sleep_freed_GB"], t, res["round_trip_A_vs_A"]), flush=True)

    # repeated syncs (as in training) + RL-style sampling with per-token log-probs
    ts = [eng.sync_weights(policy) for _ in range(3)]
    res["sync_seconds_repeat"] = [round(x, 2) for x in ts]
    t0 = time.time()
    outs = eng.generate([PROMPTS[i % len(PROMPTS)] for i in range(64)], n=8, max_tokens=256, temperature=1.0, seed=0)
    dt = time.time() - t0
    n_tok = sum(len(r.completion_ids) for g in outs for r in g)
    res["sampling"] = {"prompts": 64, "n": 8, "tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / dt),
                       "logprobs_per_token": all(len(r.completion_logprobs) == len(r.completion_ids) for g in outs for r in g),
                       "finished_frac": sum(r.finished for g in outs for r in g) / (64 * 8)}
    print("[sample]", json.dumps(res["sampling"]), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
