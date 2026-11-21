"""Phase 2 correctness tests: doc-mask isolation, window FIM invariants,
block-aligned agentic sampling + pad masking, Phase 2 LR schedule, and that
the Phase 1 code path (no block mask) is unchanged."""

import json
import random

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import IGNORE_INDEX, MultiPoolTokenDataset
from src.data.fim import apply_window_fim
from src.model.attention import build_doc_block_mask
from src.model.transformer import Transformer
from src.optim.schedule import phase2_lr_multiplier

EOT, PAD, PRE, MID, SUF = 0, 49155, 49152, 49153, 49154
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_model():
    cfg = load_config("configs/base.yaml", debug=True)
    cfg.model.moe.backend = "fallback"
    torch.manual_seed(0)
    return Transformer(cfg.model).to(DEVICE).eval(), cfg


@pytest.mark.skipif(DEVICE != "cuda", reason="flex_attention block masks need CUDA here")
def test_doc_mask_isolates_documents():
    model, cfg = tiny_model()
    T = 128
    x = torch.randint(10, 1000, (1, T), device=DEVICE)
    x[0, 63] = EOT  # doc A = [0..63], doc B = [64..127]
    x2 = x.clone()
    x2[0, :63] = torch.randint(10, 1000, (63,), device=DEVICE)  # perturb doc A only

    with torch.no_grad():
        la, *_ = model(x, None, build_doc_block_mask(x, EOT))
        lb, *_ = model(x2, None, build_doc_block_mask(x2, EOT))
    # doc B's logits must be bit-for-bit unaffected by doc A's content
    assert torch.allclose(la[0, 64:], lb[0, 64:], atol=1e-5), (la[0, 64:] - lb[0, 64:]).abs().max()
    # sanity: without the mask, doc B DOES see doc A (so the test can fail)
    with torch.no_grad():
        ua, *_ = model(x)
        ub, *_ = model(x2)
    assert not torch.allclose(ua[0, 64:], ub[0, 64:], atol=1e-5)


@pytest.mark.skipif(DEVICE != "cuda", reason="flex_attention block masks need CUDA here")
def test_doc_mask_single_doc_matches_causal_sdpa():
    model, _ = tiny_model()
    x = torch.randint(10, 1000, (2, 128), device=DEVICE)  # no EOT -> one doc per row
    with torch.no_grad():
        masked, *_ = model(x, None, build_doc_block_mask(x, EOT))
        causal, *_ = model(x)  # Phase 1 path
    assert torch.allclose(masked, causal, atol=2e-4), (masked - causal).abs().max()


def test_window_fim_invariants():
    rng = random.Random(0)
    seen_psm = seen_spm = False
    for _ in range(300):
        doc1 = np.random.randint(10, 1000, 40)
        doc2 = np.random.randint(10, 1000, 120)
        window = np.concatenate([doc1, [EOT], doc2, [EOT], np.random.randint(10, 1000, 20)]).astype(np.int64)
        out = apply_window_fim(window, EOT, PRE, MID, SUF, rng)
        assert len(out) == len(window)
        assert (out == PRE).sum() == 1 and (out == SUF).sum() <= 1 and (out == MID).sum() <= 1
        p = int(np.flatnonzero(out == PRE)[0])
        # rewritten span is the longest doc (doc2), starting right after the first EOT
        assert p == 41
        if out[p + 1] == SUF:
            seen_spm = True
        else:
            seen_psm = True
    assert seen_psm and seen_spm


def test_window_fim_no_long_span_is_noop():
    rng = random.Random(0)
    window = np.array([5, 6, EOT, 7, 8, EOT, 9], dtype=np.int64)
    assert np.array_equal(apply_window_fim(window, EOT, PRE, MID, SUF, rng), window)


def _write_pool(d, tokens):
    d.mkdir(parents=True)
    np.asarray(tokens, dtype=np.uint16).tofile(d / "shard_00000.bin")
    (d / "meta.json").write_text(json.dumps({"real_vocab_size": 49164}))


def test_block_pool_alignment_and_pad_masking(tmp_path):
    L = 16
    blocks = []
    for _ in range(400):  # each block: an episode then pad tail
        ep = list(np.random.randint(10, 1000, 10)) + [EOT]
        blocks.extend(ep + [PAD] * (L - len(ep)))
    _write_pool(tmp_path / "ag", blocks)
    _write_pool(tmp_path / "en", np.random.randint(10, 1000, 50_000))
    cfg = {
        "fim_rate": 0.0,
        "val_tokens_per_pool": 64 * L,
        "pools": {"ag": {"path": str(tmp_path / "ag"), "group": "agentic", "kind": "block"},
                  "en": {"path": str(tmp_path / "en"), "group": "english"}},
        "stages": [{"until_step": 10, "groups": {"agentic": 1.0, "english": 0.0}}],
    }
    ds = MultiPoolTokenDataset(cfg, L, PAD, EOT, (PRE, MID, SUF))
    x, y = ds.get_batch(64, "cpu", step=0)
    assert (x[:, 11:] == PAD).all(), "block start must be aligned: pad tail at fixed offset"
    assert (y[:, 10:] == IGNORE_INDEX).all(), "pad targets and cross-block target must be masked"
    assert (y[:, :10] != IGNORE_INDEX).all()
    # val split is disjoint from training
    pool = ds.pools["ag"]
    assert pool.train_lengths[-1] + pool.val_tokens == len(pool.shards[-1])


def test_stage_mixture(tmp_path):
    L = 16
    for n in ("a", "b"):
        _write_pool(tmp_path / n, np.random.randint(10, 1000, 20_000))
    cfg = {
        "pools": {"a": {"path": str(tmp_path / "a"), "group": "g1"}, "b": {"path": str(tmp_path / "b"), "group": "g2"}},
        "stages": [{"until_step": 5, "groups": {"g1": 0.9, "g2": 0.1}},
                   {"until_step": 10, "groups": {"g1": 0.2, "g2": 0.8}, "boost": {"a": 0.0}}],
    }
    ds = MultiPoolTokenDataset(cfg, L, PAD, EOT, None)
    assert ds.pool_probs(0) == pytest.approx({"a": 0.9, "b": 0.1})
    assert ds.pool_probs(1)["b"] == pytest.approx(1.0)  # boost 0 removes a entirely
    assert ds.stage_for(4) == 0 and ds.stage_for(5) == 1 and ds.stage_for(99) == 1


def test_phase2_schedule():
    f = lambda s: phase2_lr_multiplier(s, 300, 19073, 22888, 0.25, 0.025)
    assert f(0) == pytest.approx(0.25 + 0.75 / 300)
    assert f(299) == pytest.approx(1.0)
    assert f(300) == 1.0 and f(19072) == 1.0
    assert f(19073) == pytest.approx(1.0)
    assert f(22888) == pytest.approx(0.025)
    assert all(f(s) >= f(s + 1) for s in range(19073, 22888, 37))


def test_prefetcher_matches_inline_order(tmp_path):
    from src.data.dataset import BatchPrefetcher
    L = 16
    for n in ("a", "b"):
        _write_pool(tmp_path / n, np.random.randint(10, 1000, 20_000))
    cfg = {
        "pools": {"a": {"path": str(tmp_path / "a"), "group": "g1"}, "b": {"path": str(tmp_path / "b"), "group": "g2"}},
        "stages": [{"until_step": 3, "groups": {"g1": 1.0, "g2": 0.0}},
                   {"until_step": 6, "groups": {"g1": 0.0, "g2": 1.0}}],
    }
    np.random.seed(7)
    inline_ds = MultiPoolTokenDataset(cfg, L, PAD, EOT, None, seed=1)
    inline = [inline_ds.get_batch(4, "cpu", step=s) for s in range(6) for _ in range(2)]

    np.random.seed(7)
    pf_ds = MultiPoolTokenDataset(cfg, L, PAD, EOT, None, seed=1)
    pf = BatchPrefetcher(pf_ds, micro_batch=4, grad_accum=2, start_step=0, total_steps=6, depth=3)
    fetched = [pf.next("cpu") for _ in range(12)]
    pf.close()

    for (xi, yi), (xp, yp) in zip(inline, fetched):
        assert torch.equal(xi, xp) and torch.equal(yi, yp)
    with pytest.raises(StopIteration):
        BatchPrefetcher(pf_ds, 4, 2, 0, 0).next("cpu")


def test_kv_cache_matches_full_forward():
    model, _ = tiny_model()
    torch.manual_seed(1)
    T, P = 40, 17  # total length, prefill length
    x = torch.randint(10, 1000, (1, T), device=DEVICE)
    with torch.no_grad():
        full, *_ = model(x)  # reference: no cache, whole sequence at once
        cache = model.new_kv_cache()
        pre, *_ = model(x[:, :P], kv_cache=cache)
        assert torch.allclose(pre[0], full[0, :P], atol=1e-4), (pre[0] - full[0, :P]).abs().max()
        for t in range(P, T):  # single-token decode steps
            step, *_ = model(x[:, t : t + 1], kv_cache=cache)
            assert torch.allclose(step[0, -1], full[0, t], atol=1e-4), (t, (step[0, -1] - full[0, t]).abs().max())
        assert cache.seq_len == T
