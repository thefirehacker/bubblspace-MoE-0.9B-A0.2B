"""Three-level binding evaluation (L1 single row / L2 pair / L3 writing) for BubblSpace checkpoints.

Same suites, prompts and scoring as scripts/binding_levels_hf.py (the cross-model study in
doc/binding-study-01-cross-model.md), so BubblSpace results sit in the same table as Pythia / OLMo-2 / Qwen2.5.
Only the model wrapper differs: this loads our own Transformer from a step_N.pt / .safetensors checkpoint and
exposes the small interface the HF script's run_probes() / run_winogrande() use
(tok, eos, ctx_ids, cont_ids, logprob, greedy).

Tokenisation follows src/eval/lm_eval_adapter.py (no special tokens; an empty context becomes the doc separator),
so L1 here matches how our lm-eval numbers are scored.

    python scripts/binding_levels_bubbl.py --checkpoint step_22888.pt --winogrande \
        --probes v4=benchmark/data/.../v4.json v5=benchmark/data/.../v5.json --out binding_levels_step22888.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer  # noqa: E402

import binding_levels_hf as bl  # noqa: E402
from scripts.run_benchmarks import load_weights  # noqa: E402
from src.config import load_config  # noqa: E402
from src.eval.determinism import make_deterministic  # noqa: E402
from src.model.moe import probe_grouped_mm  # noqa: E402
from src.model.transformer import Transformer  # noqa: E402


class BubblLM:
    def __init__(self, config, checkpoint):
        cfg = load_config(config)
        if cfg.model.moe.backend == "auto":
            cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
        self.model = Transformer(cfg.model).to(self.device)
        self.step = load_weights(self.model, checkpoint)
        self.model.eval()
        self.vocab = len(self.tok)
        self.max_len = cfg.model.max_seq_len
        self.eos = self.tok.convert_tokens_to_ids(cfg.tokenizer.doc_separator)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.weights_format = "bubblspace-checkpoint"

    def _fwd(self, x, **kw):
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            return self.model(x, **kw)[0]

    def ctx_ids(self, text):
        return self.tok.encode(text, add_special_tokens=False) or [self.eos]

    def cont_ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    @torch.no_grad()
    def logprob(self, ctx, cont):
        full = (ctx + cont)[-self.max_len:]
        x = torch.tensor([full], device=self.device)
        logits = self._fwd(x[:, :-1])[0, -len(cont):, : self.vocab].float()
        return F.log_softmax(logits, -1).gather(1, x[0, -len(cont):, None]).sum().item()

    @torch.no_grad()
    def greedy(self, ctx, n):
        cache = self.model.new_kv_cache()
        x = torch.tensor([ctx[-(self.max_len - n):]], device=self.device)
        out = []
        for _ in range(n):
            nid = int(self._fwd(x, kv_cache=cache)[0, -1, : self.vocab].argmax())
            if nid == self.eos:
                break
            out.append(nid)
            x = torch.tensor([[nid]], device=self.device)
        return out


def main():
    make_deterministic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probes", nargs="*", default=[], help="name=path pairs")
    ap.add_argument("--winogrande", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(0)
    lm = BubblLM(a.config, a.checkpoint)
    res = {"model": "bubblspace-moe0.9b-a0.2b", "checkpoint": a.checkpoint, "step": lm.step, "n_params": lm.n_params,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "weights_format": lm.weights_format, "suites": {}}
    for spec in a.probes:
        name, path = spec.split("=", 1)
        res["suites"][name] = s = bl.run_probes(lm, path)
        print(f"{name}: L1 {s['L1']:.3f}  L2 {s['L2']:.3f}  L3 {s['L3']:.3f}  (n={s['n_pairs']})", flush=True)
    if a.winogrande:
        res["suites"]["winogrande"] = s = bl.run_winogrande(lm)
        print(f"winogrande: L1 {s['L1']:.3f} (all rows {s['l1_all_rows']:.3f})  L2 {s['L2']:.3f}  L3 {s['L3']:.3f}  "
              f"answered_with_option {s['answered_with_option']:.3f}  prior_free {s['prior_free_pairs']:.3f}  shortcuts {s['shortcuts']}", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out} ({res['elapsed_s']} s)")


if __name__ == "__main__":
    main()
