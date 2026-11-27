"""Make a vLLM model directory for BubblSpace MoE-0.9B-A0.2B (served by the plugin in vllm_plugin/).

    <out>/config.json             architecture BubblSpaceMoeForCausalLM; model_type qwen3_moe so transformers can parse
                                  it (the shapes/fields match; the plugin, not Qwen3-MoE code, runs the model)
    <out>/generation_config.json  stop at <|im_end|> or <|endoftext|>
    <out>/model.safetensors       our weights under our own names (linked, or exported from a training .pt)
    <out>/tokenizer.json, tokenizer_config.json (with the chat template)

    python scripts/make_vllm_model_dir.py --config configs/phase3.yaml --weights model.safetensors --tokenizer tok/ --out vllm_models/phase3-step1500
    python scripts/make_vllm_model_dir.py --config configs/phase3.yaml --checkpoint step_1500.pt --tokenizer tok/ --out ...
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402

CHAT_TEMPLATE = ('{% for message in messages %}<|im_start|>{{ message["role"] }}\n{{ message["content"] }}\n<|im_end|>{% endfor %}'
                 '{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training config of the checkpoint (shapes, RoPE base, context)")
    ap.add_argument("--weights", help="model.safetensors with our names (e.g. from the Hugging Face repo)")
    ap.add_argument("--checkpoint", help="or a training .pt checkpoint: its 'model' state dict is exported")
    ap.add_argument("--tokenizer", required=True, help="tokenizer dir (tokenizer.json + tokenizer_config.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert bool(a.weights) != bool(a.checkpoint), "give --weights or --checkpoint"
    m = load_config(a.config).model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos, im_end = tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")
    cfg = {
        "architectures": ["BubblSpaceMoeForCausalLM"], "model_type": "qwen3_moe",
        "vocab_size": m.vocab_size, "hidden_size": m.d_model, "num_hidden_layers": m.n_layers,
        "num_attention_heads": m.attention.n_heads, "num_key_value_heads": m.attention.n_kv_heads,
        "head_dim": m.attention.head_dim, "hidden_act": "silu", "rms_norm_eps": m.norm_eps,
        "max_position_embeddings": m.max_seq_len, "rope_theta": float(m.attention.rope_theta),
        "rope_parameters": {"rope_type": "default", "rope_theta": float(m.attention.rope_theta)},
        "num_experts": m.moe.n_experts, "num_experts_per_tok": m.moe.top_k, "moe_intermediate_size": m.moe.expert_hidden,
        "intermediate_size": m.moe.expert_hidden, "norm_topk_prob": True, "decoder_sparse_step": 1, "mlp_only_layers": [],
        "tie_word_embeddings": True, "attention_bias": False, "torch_dtype": "bfloat16", "dtype": "bfloat16",
        "bos_token_id": eos, "eos_token_id": [im_end, eos],
        "bubblspace": {"attn_output_gate": True, "norm": "zero_centered_rms (1 + w)", "qk_norm_before_rope": True,
                       "real_vocab_size": len(tok)},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [im_end, eos], "bos_token_id": eos}, indent=2))

    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (Path(a.tokenizer) / f).exists():
            shutil.copy(Path(a.tokenizer) / f, out / f)
    tc = json.loads((out / "tokenizer_config.json").read_text())
    tc.setdefault("chat_template", CHAT_TEMPLATE)
    (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2, ensure_ascii=False))

    dst = out / "model.safetensors"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if a.weights:
        os.symlink(Path(a.weights).resolve(), dst)
    else:
        import torch
        from safetensors.torch import save_file
        sd = torch.load(a.checkpoint, map_location="cpu")["model"]
        save_file({k.removeprefix("_orig_mod."): v.contiguous() for k, v in sd.items()}, str(dst))
    print("wrote", out, sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
