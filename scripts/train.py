import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RunConfig, load_config
from src.data.dataset import ShardedTokenDataset
from src.model.moe import probe_grouped_mm
from src.model.transformer import Transformer
from src.optim.param_groups import build_optimizers
from src.optim.schedule import build_scheduler


def get_raw_model(model):
    # torch.compile wraps the module in an OptimizedModule exposing the
    # original under _orig_mod; unwrap so attribute access (make_load_tracker,
    # rope buffers, blocks) reaches the real Transformer.
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def resolve_backend(cfg: RunConfig) -> None:
    if cfg.model.moe.backend == "auto":
        cfg.model.moe.backend = "grouped_mm" if probe_grouped_mm() else "fallback"
    print(f"[moe] backend resolved to: {cfg.model.moe.backend}")


def save_checkpoint(path: Path, model, muon_opt, adamw_opt, muon_sched, adamw_sched, step: int, cfg: RunConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": get_raw_model(model).state_dict(),
            "muon_opt": muon_opt.state_dict(),
            "adamw_opt": adamw_opt.state_dict(),
            "muon_sched": muon_sched.state_dict(),
            "adamw_sched": adamw_sched.state_dict(),
            "step": step,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "config": cfg.raw,
        },
        path,
    )
    print(f"[ckpt] saved {path}")


def load_checkpoint(path: Path, model, muon_opt, adamw_opt, muon_sched, adamw_sched) -> int:
    ckpt = torch.load(path, map_location="cpu")
    get_raw_model(model).load_state_dict(ckpt["model"])
    muon_opt.load_state_dict(ckpt["muon_opt"])
    adamw_opt.load_state_dict(ckpt["adamw_opt"])
    muon_sched.load_state_dict(ckpt["muon_sched"])
    adamw_sched.load_state_dict(ckpt["adamw_sched"])
    torch.set_rng_state(ckpt["rng_state"])
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
    print(f"[ckpt] resumed from {path} at step {ckpt['step']}")
    return ckpt["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/base.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    load_dotenv()

    cfg = load_config(args.config, debug=args.debug)
    resolve_backend(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    model = Transformer(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} total params")

    if cfg.training.compile:
        model = torch.compile(model)

    muon_opt, adamw_opt = build_optimizers(model, cfg.optimizer.muon, cfg.optimizer.adamw)

    total_steps = cfg.training.total_tokens // cfg.training.global_batch_tokens
    muon_sched = build_scheduler(
        muon_opt, cfg.schedule.warmup_steps, total_steps, cfg.schedule.decay, cfg.schedule.min_lr_ratio, cfg.schedule.wsd_decay_frac
    )
    adamw_sched = build_scheduler(
        adamw_opt, cfg.schedule.warmup_steps, total_steps, cfg.schedule.decay, cfg.schedule.min_lr_ratio, cfg.schedule.wsd_decay_frac
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(Path(args.resume), model, muon_opt, adamw_opt, muon_sched, adamw_sched)

    dataset = ShardedTokenDataset(cfg.data.data_dir, cfg.training.seq_len)
    load_tracker = get_raw_model(model).make_load_tracker(device)

    # vocab_size is padded to a multiple of 128 for hardware efficiency; the
    # tokenizer never actually produces ids in the padded tail, so those
    # output rows are sliced out of the loss entirely (never just relying on
    # cross-entropy's target-matching to ignore them, since they'd still
    # participate in the softmax normalization otherwise).
    real_vocab_size = dataset.meta["real_vocab_size"]
    assert real_vocab_size <= cfg.model.vocab_size, (
        f"tokenizer real_vocab_size ({real_vocab_size}) exceeds model.vocab_size ({cfg.model.vocab_size})"
    )
    print(f"[data] real_vocab_size={real_vocab_size}, model.vocab_size={cfg.model.vocab_size} (loss masks the padded tail)")

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.run_name, config=cfg.raw, resume="allow")

    grad_accum = cfg.training.grad_accum_steps
    n_experts = cfg.model.moe.n_experts
    autocast_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float32

    for step in range(start_step, total_steps):
        t0 = time.time()
        muon_opt.zero_grad(set_to_none=True)
        adamw_opt.zero_grad(set_to_none=True)
        if load_tracker is not None:
            load_tracker.reset()

        total_loss = 0.0
        total_lb = 0.0
        total_z = 0.0
        last_x = None

        for _ in range(grad_accum):
            x, y = dataset.get_batch(cfg.training.micro_batch, device)
            last_x = x
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=autocast_dtype):
                logits, lb_loss, z_loss, _ = model(x, load_tracker)
                real_logits = logits[..., :real_vocab_size]
                ce_loss = F.cross_entropy(real_logits.reshape(-1, real_vocab_size), y.view(-1))
                loss = (
                    ce_loss
                    + cfg.model.moe.load_balance_coef * lb_loss
                    + cfg.model.moe.router_z_loss_coef * z_loss
                )
            (loss / grad_accum).backward()
            total_loss += ce_loss.item() / grad_accum
            total_lb += lb_loss.item() / grad_accum
            total_z += z_loss.item() / grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)

        muon_opt.step()
        adamw_opt.step()
        muon_sched.step()
        adamw_sched.step()

        step_time = time.time() - t0
        tokens_per_sec = cfg.training.global_batch_tokens / step_time

        if step % cfg.training.log_every == 0:
            print(
                f"step {step}/{total_steps} loss {total_loss:.4f} lb {total_lb:.4f} "
                f"z {total_z:.4f} grad_norm {grad_norm:.3f} tok/s {tokens_per_sec:,.0f}"
            )
            wandb.log(
                {
                    "loss": total_loss,
                    "lb_loss": total_lb,
                    "z_loss": total_z,
                    "grad_norm": grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                    "lr": muon_sched.get_last_lr()[0],
                },
                step=step,
            )

        if step % cfg.training.expert_hist_every == 0 and load_tracker is not None:
            counts = load_tracker.counts.sum(dim=0).cpu().numpy()  # sum over layers
            wandb.log({"tokens_per_expert": wandb.Histogram(np_histogram=(counts, list(range(n_experts + 1))))}, step=step)

        if step % cfg.training.attn_logit_every == 0 and last_x is not None:
            raw_model = get_raw_model(model)
            with torch.no_grad():
                seq_len = last_x.shape[1]
                cos = raw_model.rope_cos[:seq_len].to(device)
                sin = raw_model.rope_sin[:seq_len].to(device)
                h = raw_model.embed(last_x)
                max_logit = raw_model.blocks[0].attn.max_attn_logit(raw_model.blocks[0].norm1(h), cos, sin)
            wandb.log({"max_attn_logit": max_logit}, step=step)

        if step > 0 and step % cfg.training.checkpoint_every == 0:
            ckpt_path = Path(cfg.data.checkpoint_dir) / cfg.run_name / f"step_{step}.pt"
            save_checkpoint(ckpt_path, model, muon_opt, adamw_opt, muon_sched, adamw_sched, step, cfg)

    final_path = Path(cfg.data.checkpoint_dir) / cfg.run_name / f"step_{total_steps}.pt"
    save_checkpoint(final_path, model, muon_opt, adamw_opt, muon_sched, adamw_sched, total_steps, cfg)


if __name__ == "__main__":
    main()
