import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data.fim import apply_window_fim

IGNORE_INDEX = -100


class ShardedTokenDataset:
    """Random-window reader over flat uint16 token shards written by prepare.py.

    Each sampled sequence is drawn from within a single shard (a standard
    nanoGPT/llm.c-style simplification — the tiny fraction of sequences that
    would otherwise straddle a shard boundary is simply not sampled).
    """

    def __init__(self, data_dir: str | Path, seq_len: int):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        with open(self.data_dir / "meta.json") as f:
            self.meta = json.load(f)

        shard_paths = sorted(self.data_dir.glob("shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.bin files found in {self.data_dir}")

        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        self.shard_lengths = np.array([len(s) for s in self.shards])
        self.shard_weights = self.shard_lengths / self.shard_lengths.sum()

    def get_batch(self, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        shard_choices = np.random.choice(len(self.shards), size=batch_size, p=self.shard_weights)
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)

        for i, shard_idx in enumerate(shard_choices):
            shard = self.shards[shard_idx]
            max_start = len(shard) - self.seq_len - 1
            start = random.randint(0, max_start)
            window = shard[start : start + self.seq_len + 1].astype(np.int64)
            x[i] = window[:-1]
            y[i] = window[1:]

        x_t = torch.from_numpy(x).to(device, non_blocking=True)
        y_t = torch.from_numpy(y).to(device, non_blocking=True)
        return x_t, y_t


class _Pool:
    """One named token pool: flat uint16 shards (possibly in subdirectories).

    kind="flat":  random windows anywhere (web text / code).
    kind="block": windows start only at multiples of seq_len (agentic/reasoning
                  data packed by scripts/prepare_agentic.py into exact
                  seq_len-token blocks, tail-padded with pad_id), so every
                  sample begins at an episode start. Padding targets and the
                  cross-block last target are masked to IGNORE_INDEX.

    The final `val_tokens` of the last shard are held out from training and
    served deterministically by val_batch().
    """

    def __init__(self, name: str, spec: dict, seq_len: int, val_tokens: int):
        self.name = name
        self.path = Path(spec["path"])
        self.kind = spec.get("kind", "flat")
        self.group = spec["group"]
        self.fim = bool(spec.get("fim", False))
        self.weight = spec.get("weight")  # optional within-group weight override
        self.seq_len = seq_len

        # recursive=False for a directory whose subfolders are OTHER pools
        # (e.g. Phase 1's FineWeb shards sit next to phase2/ and debug/).
        pattern = "**/" if spec.get("recursive", True) else ""
        shard_paths = sorted(self.path.glob(f"{pattern}shard_*.bin"))
        if not shard_paths:
            raise FileNotFoundError(f"[{name}] no shard_*.bin under {self.path}")
        metas = sorted(self.path.glob(f"{pattern}meta.json"))
        self.meta = json.loads(metas[0].read_text()) if metas else {}

        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in shard_paths]
        lengths = np.array([len(s) for s in self.shards], dtype=np.int64)

        if self.kind == "block":
            val_tokens = (val_tokens // seq_len) * seq_len
        # hold out the tail of the last shard (shrink if the shard is tiny)
        val_tokens = min(val_tokens, int(lengths[-1]) // 4 // seq_len * seq_len)
        self.val_tokens = val_tokens
        self.train_lengths = lengths.copy()
        self.train_lengths[-1] -= val_tokens
        self.unique_train_tokens = int(self.train_lengths.sum())
        self.shard_weights = self.train_lengths / self.train_lengths.sum()

    def _window(self, shard: np.ndarray, start: int) -> np.ndarray:
        return shard[start : start + self.seq_len + 1].astype(np.int64)

    def sample(self, rng: random.Random, pad_id: int, fim_ids, eot_id: int, fim_rate: float):
        shard_idx = np.random.choice(len(self.shards), p=self.shard_weights)
        shard, train_len = self.shards[shard_idx], int(self.train_lengths[shard_idx])
        L = self.seq_len

        if self.kind == "block":
            n_blocks = train_len // L
            k = rng.randrange(n_blocks)
            block = shard[k * L : (k + 1) * L].astype(np.int64)
            x = block
            y = np.empty(L, dtype=np.int64)
            y[:-1] = block[1:]
            y[-1] = IGNORE_INDEX  # next token belongs to a different block
            y[y == pad_id] = IGNORE_INDEX
            return x, y

        start = rng.randint(0, train_len - L - 1)
        window = self._window(shard, start)
        if self.fim and fim_ids is not None and rng.random() < fim_rate:
            window = apply_window_fim(window, eot_id, *fim_ids, rng)
        return window[:-1], window[1:].copy()

    def val_batch(self, n: int, pad_id: int):
        """Deterministic, evenly spaced windows from the held-out tail (no FIM)."""
        shard = self.shards[-1]
        base = int(self.train_lengths[-1])
        L = self.seq_len
        xs, ys = [], []
        if self.kind == "block":
            n_blocks = self.val_tokens // L
            for i in range(min(n, n_blocks)):
                k = i * n_blocks // max(1, min(n, n_blocks))
                block = shard[base + k * L : base + (k + 1) * L].astype(np.int64)
                y = np.empty(L, dtype=np.int64)
                y[:-1] = block[1:]
                y[-1] = IGNORE_INDEX
                y[y == pad_id] = IGNORE_INDEX
                xs.append(block)
                ys.append(y)
        else:
            span = self.val_tokens - L - 1
            if span <= 0:
                return np.empty((0, L), dtype=np.int64), np.empty((0, L), dtype=np.int64)
            for i in range(n):
                s = base + (i * span // max(1, n - 1) if n > 1 else 0)
                w = self._window(shard, s)
                xs.append(w[:-1])
                ys.append(w[1:].copy())
        return np.stack(xs), np.stack(ys)


class MultiPoolTokenDataset:
    """Phase 2 multi-pool sampler with a step-dependent mixture.

    Config (phase2.pools / phase2.stages):
      pools:  {name: {path, group, kind: flat|block, fim: bool, weight?}}
      stages: [{until_step, groups: {group: frac}, boost?: {pool: mult}}]

    Within a group, pools are weighted by `weight` if given, else by their
    unique training-token count, then multiplied by any per-stage `boost`.
    Each row of a micro-batch is drawn independently, so one micro-batch can
    mix English, code and agentic rows (the doc mask isolates each row's
    documents regardless).
    """

    def __init__(self, phase2_cfg: dict, seq_len: int, pad_id: int, eot_id: int, fim_ids, seed: int = 0):
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.eot_id = eot_id
        self.fim_ids = fim_ids
        self.fim_rate = float(phase2_cfg.get("fim_rate", 0.5))
        self.rng = random.Random(seed)
        val_tokens = int(phase2_cfg.get("val_tokens_per_pool", 2_097_152))

        self.pools = {
            name: _Pool(name, spec, seq_len, val_tokens) for name, spec in phase2_cfg["pools"].items()
        }
        self.stages = sorted(phase2_cfg["stages"], key=lambda s: s["until_step"])
        self._cache_step_stage = None
        self._cache_probs = None

        metas = [p.meta for p in self.pools.values() if p.meta.get("real_vocab_size")]
        self.meta = {"real_vocab_size": metas[0]["real_vocab_size"]} if metas else {}

    def stage_for(self, step: int) -> int:
        for i, s in enumerate(self.stages):
            if step < s["until_step"]:
                return i
        return len(self.stages) - 1

    def pool_probs(self, stage_idx: int) -> dict[str, float]:
        stage = self.stages[stage_idx]
        groups = stage["groups"]
        boost = stage.get("boost", {})
        probs = {}
        for group, gfrac in groups.items():
            members = [p for p in self.pools.values() if p.group == group]
            if not members or gfrac <= 0:
                continue
            raw = {
                p.name: float(p.weight if p.weight is not None else p.unique_train_tokens) * float(boost.get(p.name, 1.0))
                for p in members
            }
            total = sum(raw.values())
            if total <= 0:  # every pool in this group boosted to 0 this stage
                continue
            for name, w in raw.items():
                probs[name] = gfrac * w / total
        z = sum(probs.values())
        return {k: v / z for k, v in probs.items()}

    def get_batch(self, batch_size: int, device, step: int = 0):
        stage_idx = self.stage_for(step)
        if self._cache_step_stage != stage_idx:
            self._cache_probs = self.pool_probs(stage_idx)
            self._cache_step_stage = stage_idx
        names = list(self._cache_probs.keys())
        weights = np.array([self._cache_probs[n] for n in names])
        picks = np.random.choice(len(names), size=batch_size, p=weights)

        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        for i, pi in enumerate(picks):
            x[i], y[i] = self.pools[names[pi]].sample(self.rng, self.pad_id, self.fim_ids, self.eot_id, self.fim_rate)

        return (
            torch.from_numpy(x).to(device, non_blocking=True),
            torch.from_numpy(y).to(device, non_blocking=True),
        )

    def val_batches(self, n_per_pool: int, device):
        for name, pool in self.pools.items():
            xv, yv = pool.val_batch(n_per_pool, self.pad_id)
            yield name, pool.group, torch.from_numpy(xv).to(device), torch.from_numpy(yv).to(device)
