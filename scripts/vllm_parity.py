"""Check the vLLM plugin against the training-code model, then measure vLLM generation speed.

1. Teacher-forced: for each test text, the log-prob of every actual next token and the top-1 prediction, from our
   PyTorch model (src/model, bf16 autocast) and from vLLM (prompt_logprobs). Reports mean |dlogprob| and top-1 agreement.
2. Greedy continuations of the 23 chat-smoke conversations: identical texts, and the common-prefix length otherwise
   (bf16 + near-tied router choices make some drift expected; see src/eval/determinism.py).
3. Throughput: N prompts x M new tokens, sampled (temperature 1), as in RL rollouts.

Run in the vLLM env with the plugin installed:
    python scripts/vllm_parity.py --model_dir vllm_models/phase3-step1500 --config configs/phase3.yaml
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

TEXTS = [
    "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n\nprint(fibonacci(10))",
    "<|im_start|>user\nWhat is 47 + 38?\n<|im_end|><|im_start|>assistant\n47 + 38 = 85.\n<|im_end|>",
    "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.",
    "<|im_start|>system\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"dict\", "
    "\"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}]\n</tools>\n<|im_end|><|im_start|>user\nWeather in Paris?"
    "\n<|im_end|><|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n<|im_end|>",
]


def reference(cfg_path, weights, tok, texts, chats, max_new):
    from safetensors.torch import load_file
    from chat_repl import generate
    from src.config import load_config
    from src.model.moe import probe_grouped_mm
    from src.model.transformer import Transformer
    cfg = load_config(cfg_path)
    cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    model = Transformer(cfg.model).cuda().eval()
    model.load_state_dict(load_file(weights))
    tf = []
    with torch.no_grad():
        for t in texts:
            ids = tok.encode(t, add_special_tokens=False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(torch.tensor([ids], device="cuda"))[0][0].float()
            lp = torch.log_softmax(logits[:, :len(tok)], -1)
            tf.append({"ids": ids, "lp": [lp[i, ids[i + 1]].item() for i in range(len(ids) - 1)],
                       "top1": lp[:-1].argmax(-1).tolist()})
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    gens = [generate(model, tok, "cuda", len(tok), stop, cfg.model.max_seq_len, c, max_new, 1.0, 1.0, True) for c in chats]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return tf, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_new", type=int, default=160)
    ap.add_argument("--bench_prompts", type=int, default=512)
    ap.add_argument("--bench_tokens", type=int, default=256)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from phase3_chat_smoke import CASES, render
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    chats = [render(t) for _, _, t in CASES]

    t0 = time.time()
    tf_ref, gen_ref = reference(a.config, str(Path(a.model_dir) / "model.safetensors"), tok, TEXTS, chats, a.max_new)
    print(f"[ref] done in {time.time() - t0:.0f}s", flush=True)

    from vllm import LLM, SamplingParams, TokensPrompt
    llm = LLM(model=a.model_dir, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=16384, seed=0)
    res = {}

    # 1. teacher-forced
    outs = llm.generate([TokensPrompt(prompt_token_ids=r["ids"]) for r in tf_ref],
                        SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
    diffs, agree, n = [], 0, 0
    for r, o in zip(tf_ref, outs):
        for i, pl in enumerate(o.prompt_logprobs[1:]):  # position i+1: logprob of token ids[i+1] given ids[:i+1]
            tid = r["ids"][i + 1]
            diffs.append(abs(pl[tid].logprob - r["lp"][i]))
            v_top = max(pl.items(), key=lambda kv: kv[1].logprob)[0]
            agree += v_top == r["top1"][i] or (len(pl) == 1 and tid == r["top1"][i])
            n += 1
    res["teacher_forced"] = {"positions": n, "mean_abs_dlogprob": sum(diffs) / n, "max_abs_dlogprob": max(diffs),
                             "top1_agreement": agree / n}
    print("[parity] teacher-forced:", json.dumps(res["teacher_forced"]), flush=True)

    # 2. greedy continuations
    sp = SamplingParams(max_tokens=a.max_new, temperature=0, skip_special_tokens=False, stop_token_ids=[tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id])
    gen_v = [o.outputs[0].text for o in llm.generate(chats, sp)]
    same, prefix = 0, []
    for (cat, name, _), r, v in zip(CASES, gen_ref, gen_v):
        k = next((i for i in range(min(len(r), len(v))) if r[i] != v[i]), min(len(r), len(v)))
        same += r == v
        prefix.append(k / max(1, len(r)))
        if r != v:
            print(f"  [{name}] diverges at char {k}/{len(r)}: ref {r[k:k + 40]!r} | vllm {v[k:k + 40]!r}")
    res["greedy"] = {"cases": len(CASES), "identical": same, "mean_common_prefix_frac": sum(prefix) / len(prefix)}
    print("[parity] greedy:", json.dumps(res["greedy"]), flush=True)
    res["examples"] = [{"case": n, "vllm": v[:300]} for (_, n, _), v in zip(CASES[:6], gen_v[:6])]

    # 3. throughput (RL-style: many prompts, sampled, fixed length)
    prompts = [chats[i % len(chats)] for i in range(a.bench_prompts)]
    sp = SamplingParams(max_tokens=a.bench_tokens, min_tokens=a.bench_tokens, temperature=1.0, seed=0)
    llm.generate(prompts[:16], sp)  # warm-up
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    n_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    res["throughput"] = {"prompts": len(prompts), "generated_tokens": n_tok, "seconds": round(dt, 1), "tok_per_s": round(n_tok / dt)}
    print("[bench]", json.dumps(res["throughput"]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
