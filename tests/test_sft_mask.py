"""SFT loss mask (scripts/phase3_build_sft.py + src/data/dataset.py): targets survive only where the mask marks
assistant tokens; everything else (system/user turns, role headers, padding, the block's last position) is ignored."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import IGNORE_INDEX, _Pool  # noqa: E402

PAD, EOT = 99, 0


def _make_pool(tmp: Path, block: int, n_blocks: int):
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 90, size=block * n_blocks).astype(np.uint16)
    mask = rng.integers(0, 2, size=block * n_blocks).astype(np.uint8)
    ids[block - 3:block] = PAD  # padded tail in the first block
    mask[block - 3:block] = 0
    ids.tofile(tmp / "shard_00000.bin")
    mask.tofile(tmp / "mask_00000.bin")
    (tmp / "meta.json").write_text(json.dumps({"seq_len": block, "loss_mask": True, "eot_token": EOT}))
    return ids.astype(np.int64), mask


def _expected(ids_block, mask_block):
    y = np.full(len(ids_block), IGNORE_INDEX, dtype=np.int64)
    for j in range(len(ids_block) - 1):
        if mask_block[j + 1] == 1 and ids_block[j + 1] != PAD:
            y[j] = ids_block[j + 1]
    return y


def test_mask_native_block(tmp_path):
    B = 16
    ids, mask = _make_pool(tmp_path, B, 8)
    pool = _Pool("sft", {"path": str(tmp_path), "group": "g", "kind": "block"}, seq_len=B, val_tokens=0)
    assert pool.masks is not None
    for k in range(4):  # the blocks _block_xy sees are exactly shard blocks; check against a hand-built target
        x, y = pool._block_xy(ids[k * B:(k + 1) * B], PAD, mask[k * B:(k + 1) * B])
        assert np.array_equal(y, _expected(ids[k * B:(k + 1) * B], mask[k * B:(k + 1) * B]))
    random.seed(0)
    np.random.seed(0)
    x, y = pool.sample(random.Random(0), PAD, None, EOT, 0.0)
    kept = y != IGNORE_INDEX
    assert kept.any() and (~kept).any()


def test_mask_joined_blocks(tmp_path):
    B = 16
    ids, mask = _make_pool(tmp_path, B, 8)
    pool = _Pool("sft", {"path": str(tmp_path), "group": "g", "kind": "block"}, seq_len=4 * B, val_tokens=0)
    x, y = pool.sample(random.Random(1), PAD, None, EOT, 0.0)
    assert x.shape == y.shape == (4 * B,)
    # every surviving target must be a real token that some block's mask marked as a target
    assert set(np.unique(y[y != IGNORE_INDEX])) <= set(np.unique(ids[mask == 1]))
    assert (x != PAD).all()  # pads in joined inputs become document boundaries (EOT)


def test_no_mask_pool_unchanged(tmp_path):
    B = 16
    ids, _ = _make_pool(tmp_path, B, 4)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["loss_mask"] = False
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    pool = _Pool("plain", {"path": str(tmp_path), "group": "g", "kind": "block"}, seq_len=B, val_tokens=0)
    assert pool.masks is None
    _, y = pool._block_xy(ids[:B], PAD)
    assert (y[:-1][ids[1:B] != PAD] != IGNORE_INDEX).all()
