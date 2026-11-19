from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class AttentionConfig:
    n_heads: int
    head_dim: int
    n_kv_heads: int
    qk_norm: bool
    rope_theta: float
    output_gate: bool
    bias: bool


@dataclass
class MoEConfig:
    n_experts: int
    top_k: int
    expert_hidden: int
    activation: str
    router_init_std: float
    backend: str
    load_balance_coef: float
    router_z_loss_coef: float


@dataclass
class InitConfig:
    std: float


@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    vocab_size: int
    tie_embeddings: bool
    max_seq_len: int
    norm_eps: float
    attention: AttentionConfig
    moe: MoEConfig
    init: InitConfig


@dataclass
class MuonConfig:
    lr: float
    momentum: float
    nesterov: bool
    ns_steps: int
    weight_decay: float


@dataclass
class AdamWConfig:
    lr: float
    betas: list
    eps: float
    weight_decay: float


@dataclass
class OptimizerConfig:
    muon: MuonConfig
    adamw: AdamWConfig
    grad_clip: float


@dataclass
class ScheduleConfig:
    warmup_steps: int
    decay: str
    min_lr_ratio: float
    wsd_decay_frac: float


@dataclass
class TrainingConfig:
    dataset: str
    dataset_config: str
    total_tokens: int
    seq_len: int
    global_batch_tokens: int
    micro_batch: int
    precision: str
    compile: bool
    checkpoint_every: int
    log_every: int
    expert_hist_every: int
    attn_logit_every: int

    @property
    def grad_accum_steps(self) -> int:
        micro_tokens = self.micro_batch * self.seq_len
        assert self.global_batch_tokens % micro_tokens == 0, (
            f"global_batch_tokens ({self.global_batch_tokens}) must be a multiple "
            f"of micro_batch*seq_len ({micro_tokens})"
        )
        return self.global_batch_tokens // micro_tokens


@dataclass
class DataConfig:
    data_dir: str
    checkpoint_dir: str


@dataclass
class TokenizerConfig:
    path: str
    doc_separator: str


@dataclass
class WandbConfig:
    project: str
    entity: Optional[str] = None


@dataclass
class RunConfig:
    run_name: str
    model: ModelConfig
    optimizer: OptimizerConfig
    schedule: ScheduleConfig
    training: TrainingConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    wandb: WandbConfig
    raw: dict = field(default_factory=dict, repr=False)


def _build_run_config(d: dict) -> RunConfig:
    m = d["model"]
    attn = AttentionConfig(**m["attention"])
    moe = MoEConfig(**m["moe"])
    init = InitConfig(**m["init"])
    model = ModelConfig(
        n_layers=m["n_layers"],
        d_model=m["d_model"],
        vocab_size=m["vocab_size"],
        tie_embeddings=m["tie_embeddings"],
        max_seq_len=m["max_seq_len"],
        norm_eps=m["norm_eps"],
        attention=attn,
        moe=moe,
        init=init,
    )

    o = d["optimizer"]
    optimizer = OptimizerConfig(
        muon=MuonConfig(**o["muon"]),
        adamw=AdamWConfig(**o["adamw"]),
        grad_clip=o["grad_clip"],
    )

    schedule = ScheduleConfig(**d["schedule"])
    training = TrainingConfig(**d["training"])
    data = DataConfig(**d["data"])
    tokenizer = TokenizerConfig(**d["tokenizer"])
    wandb = WandbConfig(**d["wandb"])

    return RunConfig(
        run_name=d["run_name"],
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        training=training,
        data=data,
        tokenizer=tokenizer,
        wandb=wandb,
        raw=d,
    )


def load_config(base_path: str | Path, debug: bool = False, debug_path: str | Path | None = None) -> RunConfig:
    base_path = Path(base_path)
    with open(base_path) as f:
        merged = yaml.safe_load(f)

    if debug:
        debug_path = Path(debug_path) if debug_path else base_path.parent / "debug.yaml"
        with open(debug_path) as f:
            override = yaml.safe_load(f)
        merged = _deep_merge(merged, override)

    return _build_run_config(merged)
