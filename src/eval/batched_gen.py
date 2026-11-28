"""Batched generation with a static KV cache, for evals and (Phase 4) RL rollouts.

The model's own KV-cache path (Transformer.forward(kv_cache=...)) handles one sequence. This runs many at once with
the same weights and the same math, without touching the training code:

  * prompts are LEFT-padded to a common length; each row's RoPE positions start at 0 at its first real token
  * one prefill over the padded prompts, then one token per step for every still-running row
  * the attention mask hides pad keys (and keeps causality in the prefill); a pad query attends to itself only,
    so no row of the softmax is empty
  * the KV cache is preallocated (layers x [B, kv_heads, P + max_new, head_dim]); rows that hit a stop token or
    their budget are dropped from the batch (the cache is index-selected), so long tails don't slow short rows

Results match the one-sequence generator up to bf16/MoE-routing noise: near-tied top-8 router choices can flip
between batch shapes, so a greedy continuation can diverge after many tokens (see src/eval/determinism.py).
Pad tokens still pass through the MoE (harmless at inference: nothing attends to them).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.rope import apply_rope


def _attn_step(attn, x, cos, sin, kc, vc, t0, mask):
    """One attention call writing new k/v at cache positions [t0, t0+L). cos/sin: (B, L, hd/2). mask: (B,1,L,T)."""
    B, L, _ = x.shape
    q = attn.q_norm(attn.q_proj(x).view(B, L, attn.n_heads, attn.head_dim))
    k = attn.k_norm(attn.k_proj(x).view(B, L, attn.n_kv_heads, attn.head_dim))
    v = attn.v_proj(x).view(B, L, attn.n_kv_heads, attn.head_dim)
    c, s = cos[:, None], sin[:, None]  # broadcast over heads
    q = apply_rope(q.transpose(1, 2), c, s)
    k = apply_rope(k.transpose(1, 2), c, s)
    kc[:, :, t0:t0 + L] = k.to(kc.dtype)
    vc[:, :, t0:t0 + L] = v.transpose(1, 2).to(vc.dtype)
    T = t0 + L
    out = F.scaled_dot_product_attention(q, kc[:, :, :T].to(q.dtype), vc[:, :, :T].to(q.dtype), attn_mask=mask[..., :T],
                                         enable_gqa=(attn.n_heads != attn.n_kv_heads))
    out = out.transpose(1, 2).reshape(B, L, attn.n_heads * attn.head_dim)
    if attn.output_gate:
        out = out * torch.sigmoid(attn.gate_proj(x))
    return attn.o_proj(out)


def _forward(model, ids, pos, caches, t0, mask):
    x = model.embed(ids)
    cos = model.rope_cos.to(x.device)[pos]
    sin = model.rope_sin.to(x.device)[pos]
    for i, blk in enumerate(model.blocks):
        kc, vc = caches[i]
        x = x + _attn_step(blk.attn, blk.norm1(x), cos, sin, kc, vc, t0, mask)
        moe_out, _, _, _ = blk.moe(blk.norm2(x), None, i)
        x = x + moe_out
    x = model.final_norm(x[:, -1])
    return F.linear(x, model.embed.weight)


@torch.no_grad()
def batched_generate(model, prompts: list[list[int]], max_new_tokens: int, stop_ids, pad_id: int,
                     real_vocab_size: int | None = None, temperature: float = 0.0, top_p: float = 1.0,
                     generator: torch.Generator | None = None, device: str = "cuda"):
    """Generate for a list of token-id prompts. temperature 0 = greedy.

    Returns a list of (generated_ids, stopped) per prompt, in input order. generated_ids exclude the stop token;
    stopped is True when a stop token ended the row, False when it ran out of budget.
    """
    B = len(prompts)
    if B == 0:
        return []
    stop = torch.tensor(sorted(stop_ids), device=device)
    max_len = model.cfg.max_seq_len
    prompts = [p[-(max_len - max_new_tokens):] for p in prompts]
    P = max(len(p) for p in prompts)
    T = P + max_new_tokens
    ids = torch.full((B, P), pad_id, dtype=torch.long, device=device)
    n_pad = torch.empty(B, dtype=torch.long, device=device)
    for b, p in enumerate(prompts):
        ids[b, P - len(p):] = torch.tensor(p, device=device)
        n_pad[b] = P - len(p)

    a = model.blocks[0].attn
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    caches = [(torch.zeros(B, a.n_kv_heads, T, a.head_dim, device=device, dtype=dtype),
               torch.zeros(B, a.n_kv_heads, T, a.head_dim, device=device, dtype=dtype)) for _ in model.blocks]

    ar = torch.arange(T, device=device)
    key_ok = ar[None, :] >= n_pad[:, None]  # (B, T): real tokens (prompt pads are False; generated slots True)
    pos_all = (ar[None, :] - n_pad[:, None]).clamp(min=0)  # RoPE position of each cache slot, per row
    q_idx = ar[:P]
    causal = (q_idx[None, :, None] >= ar[None, None, :])  # (1, P, T)
    self_only = (q_idx[None, :, None] == ar[None, None, :])
    mask = ((causal & key_ok[:, None, :]) | self_only)[:, None]  # (B,1,P,T)

    out = [[] for _ in range(B)]
    stopped = [False] * B
    rows = torch.arange(B, device=device)  # original index of each live row
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = _forward(model, ids, pos_all[:, :P], caches, 0, mask)
        for step in range(max_new_tokens):
            lg = logits.float()
            if real_vocab_size:
                lg = lg[:, :real_vocab_size]
            if temperature <= 0:
                nxt = lg.argmax(-1)
            else:
                probs = F.softmax(lg / temperature, -1)
                if top_p < 1.0:
                    sp, si = probs.sort(-1, descending=True)
                    keep = (sp.cumsum(-1) - sp) < top_p
                    probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            is_stop = torch.isin(nxt, stop)
            nxt_l, stop_l, rows_l = nxt.tolist(), is_stop.tolist(), rows.tolist()
            for j, r in enumerate(rows_l):
                if stop_l[j]:
                    stopped[r] = True
                else:
                    out[r].append(nxt_l[j])
            live = ~is_stop
            if step == max_new_tokens - 1 or not live.any():
                break
            if not live.all():  # drop finished rows from the batch
                keep_idx = live.nonzero().squeeze(-1)
                rows, nxt = rows[keep_idx], nxt[keep_idx]
                caches = [(k[keep_idx], v[keep_idx]) for k, v in caches]
                key_ok, pos_all = key_ok[keep_idx], pos_all[keep_idx]
            t = P + step
            m = key_ok[:, None, None, :].clone()
            m[..., t + 1:] = False
            logits = _forward(model, nxt[:, None], pos_all[:, t:t + 1], caches, t, m)
    return [(out[b], stopped[b]) for b in range(B)]


def generate_many(model, prompts, max_new_tokens, stop_ids, pad_id, batch_size=32, sort_by_len=True, **kw):
    """batched_generate over a long list, in chunks; similar-length prompts are batched together to limit padding."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i])) if sort_by_len else list(range(len(prompts)))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        for i, r in zip(chunk, batched_generate(model, [prompts[i] for i in chunk], max_new_tokens, stop_ids, pad_id, **kw)):
            res[i] = r
    return res
