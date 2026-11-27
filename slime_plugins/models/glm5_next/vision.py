"""Checkpoint-compatible CogViT using the Transformers GLM-OCR building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from slime.utils import accelerator
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.glm_ocr.modeling_glm_ocr import (
    GlmOcrVisionBlock,
    GlmOcrVisionMlp,
    GlmOcrVisionModel,
    GlmOcrVisionPatchEmbed,
    GlmOcrVisionPatchMerger,
)

from .attention import RMSNorm
from .ops import swiglu_with_limit


def vision_swiglu_with_limit(x, limit):
    if x.device.type == "npu":
        # Inference disables torch.compile on NPU. Match its eager rounding
        # at each operation, including the BF16 SiLU intermediate.
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    # Preserve the FP32 intermediates used by compiled inference elsewhere.
    return swiglu_with_limit(x, limit)


class Glm5NextVisionRMSNorm(RMSNorm):
    def forward(self, x):
        if x.device.type == "npu":
            import torch_npu

            return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]
        return super().forward(x)


class Glm5NextVisionBlock(GlmOcrVisionBlock):
    def __init__(self, config):
        super().__init__(config)
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        # NPU Q/K normalization uses the model epsilon; GPU uses 1e-6.
        qk_norm_eps = config.rms_norm_eps if accelerator.device_type() == "npu" else 1e-6
        if accelerator.device_type() == "npu":
            from .vision_attention import Glm5NextVisionAttention

            self.attn = Glm5NextVisionAttention(config)
        self.attn.q_norm = Glm5NextVisionRMSNorm(config.hidden_size // config.num_heads, qk_norm_eps)
        self.attn.k_norm = Glm5NextVisionRMSNorm(config.hidden_size // config.num_heads, qk_norm_eps)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        attention = self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, **kwargs
        )
        if hidden_states.device.type == "npu":
            import torch_npu

            normalized, _, residual = torch_npu.npu_add_rms_norm(
                hidden_states, attention, self.norm2.weight, self.norm2.eps
            )
            return residual + self.mlp(normalized)
        # Match the fused residual-add/RMSNorm: normalize the FP32 sum,
        # while the residual branch itself is stored in the model dtype.
        residual = hidden_states.float() + attention.float()
        normalized = self.norm2(residual).to(hidden_states.dtype)
        return residual.to(hidden_states.dtype) + self.mlp(normalized)


class Glm5NextVisionMLP(GlmOcrVisionMlp):
    def __init__(self, config):
        super().__init__(config, bias=True)
        self.limit = config.swiglu_limit

    def forward(self, x):
        if x.device.type == "npu":
            # Match inference's packed gate/up GEMM shape and BF16 rounding.
            # Concatenation preserves checkpoint keys and autograd to both
            # parameters; no detached or duplicated trainable weights.
            weight = torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0)
            bias = torch.cat((self.gate_proj.bias, self.up_proj.bias), dim=0)
            gate_up = F.linear(x, weight, bias)
        else:
            gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(vision_swiglu_with_limit(gate_up, self.limit))


class Glm5NextVisionLayerNorm(nn.LayerNorm):
    """Match inference LayerNorm precision while preserving checkpoint keys."""

    def forward(self, x):
        return F.layer_norm(
            x.float(), self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(x.dtype)


class Glm5NextVisionMerger(GlmOcrVisionPatchMerger):
    def __init__(self, config):
        super().__init__(config.out_hidden_size, config.projection_intermediate_size, config.hidden_act)
        self.post_projection_norm = Glm5NextVisionLayerNorm(
            self.post_projection_norm.normalized_shape, eps=1e-6
        )
        self.limit = config.swiglu_limit
        # Store one trainable matrix, in inference's [gate; up] order. No
        # weight concatenation or duplicated parameters in the forward path.
        assert self.gate_proj.bias is None and self.up_proj.bias is None
        self.gate_up_proj = nn.Linear(
            self.gate_proj.in_features, 2 * self.gate_proj.out_features, bias=False,
            device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype,
        )
        with torch.no_grad():
            self.gate_up_proj.weight.copy_(torch.cat((self.gate_proj.weight, self.up_proj.weight), dim=0))
        del self.gate_proj, self.up_proj

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Accept legacy plain PyTorch/HF state dicts as well as new packed
        # state dicts. The streaming HF loader has the same conversion.
        packed = prefix + "gate_up_proj.weight"
        gate, up = prefix + "gate_proj.weight", prefix + "up_proj.weight"
        if packed not in state_dict and gate in state_dict and up in state_dict:
            state_dict[packed] = torch.cat((state_dict.pop(gate), state_dict.pop(up)), dim=0)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        x = self.act1(self.post_projection_norm(self.proj(x)))
        gate_up = self.gate_up_proj(x)
        return self.down_proj(vision_swiglu_with_limit(gate_up, self.limit))


class Glm5NextPatchEmbed(GlmOcrVisionPatchEmbed):
    def forward(self, hidden_states):
        # Every input is exactly one non-overlapping Conv3d patch. GEMM
        # implements the same contraction without cuDNN's BF16 Conv3d fallback.
        patches = hidden_states.reshape(-1, self.proj.weight[0].numel()).to(self.proj.weight.dtype)
        return F.linear(patches, self.proj.weight.flatten(1), self.proj.bias)


class Glm5NextVisionModel(GlmOcrVisionModel):
    def __init__(self, config):
        if accelerator.device_type() == "npu":
            config._attn_implementation = "sdpa"
        super().__init__(config)
        self.patch_embed = Glm5NextPatchEmbed(config)
        self.merger = Glm5NextVisionMerger(config)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.post_layernorm = Glm5NextVisionRMSNorm(config.hidden_size, config.rms_norm_eps)
        if accelerator.device_type() == "npu":
            # Use one FP32 frequency table regardless of the construction device.
            rotary = self.rotary_pos_emb
            with torch.device("cpu"):
                inv_freq, _ = rotary.compute_axial_rope_parameters(rotary.config, device="cpu")
            rotary.inv_freq = inv_freq.to(rotary.inv_freq.device)
            rotary.original_inv_freq = rotary.inv_freq.clone()

    def forward(self, hidden_states, grid_thw, **kwargs):
        # Keep GEMM shapes independent of rollout batching and training packing.
        if hidden_states.device.type != "npu" or len(grid_thw) <= 1:
            return super().forward(hidden_states, grid_thw, **kwargs)
        counts = grid_thw.prod(dim=-1).tolist()
        outputs = [
            super(Glm5NextVisionModel, self).forward(pixels, grid_thw[i : i + 1], **kwargs)
            for i, pixels in enumerate(hidden_states.split(counts))
        ]
        return BaseModelOutputWithPooling(
            last_hidden_state=torch.cat([x.last_hidden_state for x in outputs]),
            pooler_output=torch.cat([x.pooler_output for x in outputs]),
        )

    def _apply(self, fn, recurse=True):
        # Frequencies are not checkpoint parameters. Preserve their original
        # FP32 values when Megatron casts the model weights to BF16.
        rotary = self.rotary_pos_emb
        frequencies = {name: getattr(rotary, name) for name in ("inv_freq", "original_inv_freq")}
        super()._apply(fn, recurse=recurse)
        for name, frequency in frequencies.items():
            setattr(rotary, name, frequency.to(device=getattr(rotary, name).device))
        return self
