"""Run the standard base-LM benchmark suite (lm-evaluation-harness) against
a training checkpoint, without needing a transformers-native architecture.

Usage:
    python scripts/run_benchmarks.py \\
        --config configs/base.yaml \\
        --checkpoint /path/to/step_6000.pt \\
        --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \\
        --output results/step_6000.json

Accepts either a raw training checkpoint (.pt, has a "model" state_dict key
and optimizer/scheduler state we ignore) or a weights-only .safetensors file
(as pushed to the HF repo) — detected from the file extension.

Tasks default to the standard base-model loglikelihood-scored set (no
instruction-following/chat tasks — this model has no SFT yet, so those
would not be meaningful).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.eval.lm_eval_adapter import BubblSpaceLM
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer

DEFAULT_TASKS = "hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai"


def load_weights(model: Transformer, checkpoint_path: str) -> int | None:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
        return None
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return ckpt.get("step")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--debug", action="store_true", help="merge configs/debug.yaml, for a quick smoke test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=None, help="defaults to cfg.tokenizer.path")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--limit", type=int, default=None, help="cap examples per task, for a quick smoke test")
    parser.add_argument("--output", default=None, help="path to save results JSON")
    args = parser.parse_args()

    import lm_eval  # deferred: only needed for this script, not training

    cfg = load_config(args.config, debug=args.debug)
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    print(f"[moe] backend resolved to: {cfg.model.moe.backend}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_path = args.tokenizer or cfg.tokenizer.path
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    model = Transformer(cfg.model).to(device)
    step = load_weights(model, args.checkpoint)
    model.eval()
    print(f"[ckpt] loaded {args.checkpoint}" + (f" (step {step})" if step is not None else ""))

    lm = BubblSpaceLM(model, tok, cfg, device)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[eval] running tasks: {task_list}" + (f" (limit={args.limit})" if args.limit else ""))

    results = lm_eval.simple_evaluate(model=lm, tasks=task_list, limit=args.limit)

    summary = results["results"]
    print(json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "step": step,
                    "tasks": task_list,
                    "limit": args.limit,
                    "results": summary,
                },
                f,
                indent=2,
            )
        print(f"[eval] saved results to {out_path}")


if __name__ == "__main__":
    main()
