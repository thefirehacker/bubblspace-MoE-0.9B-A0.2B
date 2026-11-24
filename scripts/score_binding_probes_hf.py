"""Score any Hugging Face causal LM on the BubblSpace entity-binding probes (v2 / v3 / v4).

Same rule as scripts/probe_entity_binding.py (used for all BubblSpace results): for each minimal pair,
score both completions by summed log-probability after each context (as lm-evaluation-harness scores
WinoGrande). A case is correct only if the preference FLIPS with the entity: ctxA prefers c1 AND ctxB
prefers c2. Random guessing scores ~25%.

    pip install torch transformers datasets
    python score_binding_probes_hf.py --model HuggingFaceTB/SmolLM2-360M --split v4
"""

import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def logprob(model, tok, ctx, cont, device):
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"] or [tok.eos_token_id]
    cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ctx_ids + cont_ids], device=device)
    logits = model(ids[:, :-1]).logits[0, -len(cont_ids):].float()
    return F.log_softmax(logits, -1).gather(1, ids[0, -len(cont_ids):, None]).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="v4", choices=["v2", "v3", "v4"])
    ap.add_argument("--dataset", default="TheFireHacker/bubblspace-contrastive-binding")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()

    cases = load_dataset(args.dataset, "probes", split=args.split)
    n, by_cat = 0, defaultdict(lambda: [0, 0])
    for c in cases:
        a1, a2 = logprob(model, tok, c["ctxA"], c["c1"], device), logprob(model, tok, c["ctxA"], c["c2"], device)
        b1, b2 = logprob(model, tok, c["ctxB"], c["c1"], device), logprob(model, tok, c["ctxB"], c["c2"], device)
        ok = a1 > a2 and b2 > b1
        n += ok
        by_cat[c["category"]][0] += ok
        by_cat[c["category"]][1] += 1
    print(f"{args.model} probe {args.split}: {n}/{len(cases)} correct (chance ~25%)")
    for cat, (k, t) in sorted(by_cat.items()):
        print(f"  {cat:<26} {k}/{t}")


if __name__ == "__main__":
    main()
