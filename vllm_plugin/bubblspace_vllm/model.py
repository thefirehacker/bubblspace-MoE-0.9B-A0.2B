"""vLLM implementation of BubblSpace MoE-0.9B-A0.2B (out-of-tree model, registered by bubblspace_vllm.register).

Same math as src/model/ (the training code), on vLLM's fused kernels:

  * attention: q/k/v projections (GQA 16/4, head_dim 64), per-head QK-norm BEFORE RoPE (NeoX-style rotate-half RoPE),
    then the output gate: attn_out * sigmoid(gate_proj(x)) before o_proj (x = the normed block input)
  * norms: ZeroCenteredRMSNorm, y = x / rms(x) * (1 + w) in fp32 == vLLM's GemmaRMSNorm, so the checkpoint's `w`
    is loaded unchanged
  * MoE: router Linear(d, 64), softmax -> top-8 -> renormalise, SwiGLU experts, no shared expert
    our W_gate/W_up (E, d, I) and W_down (E, I, d)  ->  vLLM w13 (E, 2I, d) = [W_gate^T ; W_up^T], w2 (E, d, I) = W_down^T
  * tied embeddings

Weights load straight from our own safetensors names (blocks.N.attn.q_proj.weight, blocks.N.moe.W_gate, ...), so the
model.safetensors published on Hugging Face is used as is. Tensor parallel size 1 only (a 0.9B model fits one GPU).
"""
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors


class BubblSpaceAttention(nn.Module):
    def __init__(self, config, cache_config, prefix: str):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        d = config.hidden_size
        self.qkv_proj = QKVParallelLinear(d, self.head_dim, self.num_heads, self.num_kv_heads, bias=False,
                                          prefix=f"{prefix}.qkv_proj")
        self.gate_proj = ReplicatedLinear(d, self.q_size, bias=False, prefix=f"{prefix}.gate_proj")
        self.o_proj = RowParallelLinear(self.q_size, d, bias=False, prefix=f"{prefix}.o_proj")
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                   is_neox_style=True, rope_parameters=config.rope_parameters)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5, num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config, prefix=f"{prefix}.attn")

    def forward(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(*q.shape[:-1], self.num_heads, self.head_dim)).view(q.shape)
        k = self.k_norm(k.view(*k.shape[:-1], self.num_kv_heads, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        out = self.attn(q, k, v)
        gate, _ = self.gate_proj(x)
        out, _ = self.o_proj(out * torch.sigmoid(gate))
        return out


class BubblSpaceMoE(nn.Module):
    def __init__(self, config, prefix: str):
        super().__init__()
        self.gate = GateLinear(config.hidden_size, config.num_experts, prefix=f"{prefix}.gate")
        self.experts = FusedMoEFactory(
            gate=self.gate, num_experts=config.num_experts, top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size, intermediate_size=config.moe_intermediate_size,
            renormalize=True, scoring_func="softmax", activation="silu", prefix=f"{prefix}.experts")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.experts(hidden_states=x, router_logits=x)


class BubblSpaceDecoderLayer(nn.Module):
    def __init__(self, config, cache_config, prefix: str):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BubblSpaceAttention(config, cache_config, prefix=f"{prefix}.self_attn")
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BubblSpaceMoE(config, prefix=f"{prefix}.mlp")

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(hidden_states), residual


class BubblSpaceMoeForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        assert get_tensor_model_parallel_world_size() == 1, "BubblSpace vLLM model supports tensor_parallel_size=1"
        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}embed_tokens")
        self.layers = nn.ModuleList([BubblSpaceDecoderLayer(config, cache_config, prefix=f"{prefix}layers.{i}")
                                     for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=f"{prefix}lm_head")
        self.lm_head = self.lm_head.tie_weights(self.embed_tokens)
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @torch.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load our training-code names directly. Returns the set of vLLM parameter names that were filled."""
        params = dict(self.named_parameters())
        loaded, pending = set(), {}

        def resolve(name):
            """vLLM may nest expert weights deeper (e.g. mlp.experts.<runner>.w13_weight): match by prefix + leaf."""
            if name in params:
                return name
            head, leaf = name.rsplit(".", 1)
            hits = [n for n in params if n.startswith(head + ".") and n.endswith("." + leaf)]
            assert len(hits) == 1, f"{name}: {len(hits)} matching parameters {hits[:4]}"
            return hits[0]

        # Every tensor goes through the parameter's own weight_loader. vLLM's in-place reload (used by the RL weight
        # sync) moves layers to the meta device and wraps these loaders; a direct param.data.copy_ would be lost.
        def put(name, tensor):
            name = resolve(name)
            p = params[name]
            assert p.shape == tensor.shape, f"{name}: checkpoint {tuple(tensor.shape)} vs model {tuple(p.shape)}"
            getattr(p, "weight_loader", default_weight_loader)(p, tensor)
            loaded.add(name)

        def put_experts(L, gate, up, down):
            """Per-expert loads through FusedMoE's loader. Ours: W_gate/W_up (E, d, I), W_down (E, I, d); vLLM wants
            w1/w3 (I, d) and w2 (d, I) per expert, i.e. the transposes."""
            n13, n2 = resolve(f"{L}.mlp.experts.w13_weight"), resolve(f"{L}.mlp.experts.w2_weight")
            p13, p2 = params[n13], params[n2]
            for e in range(gate.shape[0]):
                p13.weight_loader(p13, gate[e].t(), f"{L}.mlp.experts.{e}.gate_proj.weight", shard_id="w1", expert_id=e)
                p13.weight_loader(p13, up[e].t(), f"{L}.mlp.experts.{e}.up_proj.weight", shard_id="w3", expert_id=e)
                p2.weight_loader(p2, down[e].t(), f"{L}.mlp.experts.{e}.down_proj.weight", shard_id="w2", expert_id=e)
            loaded.update((n13, n2))

        for name, w in weights:
            if name == "embed.weight":
                put("embed_tokens.weight", w)
                continue
            if name == "final_norm.w":
                put("norm.weight", w)
                continue
            if not name.startswith("blocks."):
                raise KeyError(f"unexpected weight {name}")
            _, i, rest = name.split(".", 2)
            L = f"layers.{i}"
            if rest in ("attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight"):
                shard = rest.split(".")[1][0]  # q / k / v
                p = params[f"{L}.self_attn.qkv_proj.weight"]
                p.weight_loader(p, w, shard)
                loaded.add(f"{L}.self_attn.qkv_proj.weight")
            elif rest == "attn.gate_proj.weight":
                put(f"{L}.self_attn.gate_proj.weight", w)
            elif rest == "attn.o_proj.weight":
                put(f"{L}.self_attn.o_proj.weight", w)
            elif rest == "attn.q_norm.w":
                put(f"{L}.self_attn.q_norm.weight", w)
            elif rest == "attn.k_norm.w":
                put(f"{L}.self_attn.k_norm.weight", w)
            elif rest == "norm1.w":
                put(f"{L}.input_layernorm.weight", w)
            elif rest == "norm2.w":
                put(f"{L}.post_attention_layernorm.weight", w)
            elif rest == "moe.router.weight":
                put(f"{L}.mlp.gate.weight", w)
            elif rest in ("moe.W_gate", "moe.W_up", "moe.W_down"):
                pending.setdefault(i, {})[rest] = w
                if len(pending[i]) == 3:
                    trio = pending.pop(i)
                    put_experts(L, trio["moe.W_gate"], trio["moe.W_up"], trio["moe.W_down"])
            else:
                raise KeyError(f"unexpected weight {name}")
        assert not pending, f"incomplete expert weights for layers {list(pending)}"
        missing = {n for n in params if n not in loaded and not n.startswith("lm_head")}
        assert not missing, f"weights not loaded: {sorted(missing)[:10]}"
        return loaded | {n for n in params if n.startswith("lm_head")}
