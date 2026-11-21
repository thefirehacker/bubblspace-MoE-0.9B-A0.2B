"""Profile where Phase 2 step time goes, to decide what (if anything) to optimize.

Measures, per micro-batch (16 per optimizer step at the base config):
  - data:    MultiPoolTokenDataset.get_batch (CPU: pool pick, window read, FIM)
  - mask:    build_doc_block_mask (create_block_mask)
  - fwd+bwd: compiled model forward+backward WITH the doc mask (flex_attention)
  - fwd+bwd: same batch WITHOUT the mask (Phase 1 causal SDPA path), for comparison
  - opt:     Muon + AdamW step
and reports torch.compile recompilations. Random weights are fine for timing.

Usage: TORCH_LOGS=recompiles python scripts/profile_phase2.py configs/phase2_smoke.yaml
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.config import load_config
from src.data.dataset import IGNORE_INDEX, MultiPoolTokenDataset
from src.model.attention import build_doc_block_mask
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer
from src.optim.param_groups import build_optimizers


def timed(fn, n, sync=True):
    times = []
    for _ in range(n):
        if sync:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2]  # median


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/phase2_smoke.yaml")
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    p2 = cfg.raw["phase2"]
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer.path)
    eot = tok.eos_token_id
    pad = tok.convert_tokens_to_ids("<|fim_pad|>")
    fim_ids = tuple(tok.convert_tokens_to_ids(t) for t in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"))
    ds = MultiPoolTokenDataset(p2, cfg.training.seq_len, pad, eot, fim_ids)
    mb = cfg.training.micro_batch
    rv = ds.meta["real_vocab_size"]

    model = torch.compile(Transformer(cfg.model).to(device))
    muon, adamw = build_optimizers(model, cfg.optimizer.muon, cfg.optimizer.adamw)

    # 1) data loading (pure CPU, includes the H2D copy)
    t_data = timed(lambda: ds.get_batch(mb, device, step=0), 40)
    t_data_anneal = timed(lambda: ds.get_batch(mb, device, step=99), 40)

    x, y = ds.get_batch(mb, device, step=0)
    rows_with_eot = int((x == eot).any(dim=1).sum())

    # 2) mask construction
    t_mask = timed(lambda: build_doc_block_mask(x, eot), 40)

    def fwd_bwd(masked):
        bm = build_doc_block_mask(x, eot) if masked else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, lb, z, _ = model(x, None, bm)
            loss = F.cross_entropy(logits[..., :rv].reshape(-1, rv), y.view(-1), ignore_index=IGNORE_INDEX)
            loss = loss + 0.01 * lb + 0.001 * z
        loss.backward()

    # 3) compute: masked (Phase 2) vs unmasked (Phase 1 path); warmup compiles first
    t0 = time.perf_counter()
    for _ in range(3):
        fwd_bwd(True)
    torch.cuda.synchronize()
    compile_masked = time.perf_counter() - t0
    # vary the batch so a per-batch recompile would show up in TORCH_LOGS
    for _ in range(5):
        x, y = ds.get_batch(mb, device, step=0)
        fwd_bwd(True)
    t_fb_masked_incl_mask = timed(lambda: fwd_bwd(True), 20)

    for _ in range(3):
        fwd_bwd(False)
    t_fb_unmasked = timed(lambda: fwd_bwd(False), 20)

    # 4) optimizer step
    def opt_step():
        muon.step()
        adamw.step()
    t_opt = timed(opt_step, 5)

    grad_accum = cfg.training.grad_accum_steps
    serial_step = grad_accum * (t_data + t_fb_masked_incl_mask) + t_opt
    phase1_like = grad_accum * t_fb_unmasked + t_opt
    gbt = cfg.training.global_batch_tokens

    print("\n=== Phase 2 profile (medians) ===")
    print(f"micro_batch={mb} seq_len={cfg.training.seq_len} grad_accum={grad_accum}")
    print(f"rows in batch containing an EOT: {rows_with_eot}/{mb}")
    print(f"data   get_batch (stage 2A mix): {t_data*1000:8.1f} ms / micro-batch")
    print(f"data   get_batch (stage 2B mix): {t_data_anneal*1000:8.1f} ms / micro-batch")
    print(f"mask   build_doc_block_mask:     {t_mask*1000:8.1f} ms / micro-batch")
    print(f"fwd+bwd masked (incl. mask):     {t_fb_masked_incl_mask*1000:8.1f} ms / micro-batch")
    print(f"fwd+bwd UNmasked (Phase 1 path): {t_fb_unmasked*1000:8.1f} ms / micro-batch")
    print(f"optimizer step (Muon+AdamW):     {t_opt*1000:8.1f} ms / step")
    print(f"first 3 masked steps (compile):  {compile_masked:8.1f} s")
    print(f"\nestimated step time, serial loader:  {serial_step:6.2f} s -> {gbt/serial_step:,.0f} tok/s")
    print(f"if data were fully overlapped:       {serial_step - grad_accum*t_data:6.2f} s -> {gbt/(serial_step - grad_accum*t_data):,.0f} tok/s")
    print(f"Phase-1-equivalent (no mask, no data):{phase1_like:6.2f} s -> {gbt/phase1_like:,.0f} tok/s")


if __name__ == "__main__":
    main()
