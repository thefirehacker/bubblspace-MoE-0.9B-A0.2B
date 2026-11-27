"""vLLM plugin for BubblSpace MoE-0.9B-A0.2B.

Installed with `pip install -e vllm_plugin`, vLLM finds this through the `vllm.general_plugins` entry point and calls
register() in every process it starts (engine core, workers), so `vllm serve <model_dir>` and `LLM(<model_dir>)` work
for a model directory whose config.json lists the architecture "BubblSpaceMoeForCausalLM"
(made by scripts/make_vllm_model_dir.py).
"""


def register():
    from vllm import ModelRegistry

    if "BubblSpaceMoeForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("BubblSpaceMoeForCausalLM", "bubblspace_vllm.model:BubblSpaceMoeForCausalLM")
