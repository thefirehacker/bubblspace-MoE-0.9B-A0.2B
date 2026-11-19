import json
import random
from pathlib import Path

import numpy as np
import torch


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
