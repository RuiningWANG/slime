import re

from ..hf_to_megatron.common import strip_mcore_wrappers
from .deepseekv3 import convert_deepseekv3_to_hf


def convert_glm5_next_to_hf(args, name, param):
    name = strip_mcore_wrappers(name)
    if name == "model.visual.merger.gate_up_proj.weight":
        gate, up = param.chunk(2, dim=0)
        return [("model.visual.merger.gate_proj.weight", gate),
                ("model.visual.merger.up_proj.weight", up)]
    if name.startswith("model.visual."):
        return [(name, param)]
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match:
        layer, rest = match.groups()
        if rest.startswith("self_attention."):
            return [(f"model.language_model.layers.{layer}.self_attn.{rest.removeprefix('self_attention.')}", param)]
        if rest.startswith("hc_"):
            return [(f"model.language_model.layers.{layer}.{rest}", param)]
    return [
        (key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key, value)
        for key, value in convert_deepseekv3_to_hf(args, f"module.module.{name}", param)
    ]
