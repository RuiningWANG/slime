"""Reproduce unquantized serving TP8 projection arithmetic on training TP1."""

import torch

from .npu_backend import ServingValue


@torch.no_grad()
def local_eight_shard_projection(x, weight):
    from sglang.srt.hardware_backend.npu.glm53_parity import ordered_shard_sum, projection_shard

    if x.shape[-1] != weight.shape[-1] or x.shape[-1] % 8:
        raise ValueError(f"Invalid GLM5-next projection layout: {x.shape} x {weight.shape}")
    return ordered_shard_sum(tuple(
        projection_shard(part, shard)
        for part, shard in zip(x.chunk(8, dim=-1), weight.chunk(8, dim=-1), strict=True)
    ))


def training_projection(linear, x):
    if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.numel() or linear.bias is not None:
        return linear(x)
    aligned = local_eight_shard_projection(x, linear.weight)
    if torch.is_grad_enabled() and (x.requires_grad or linear.weight.requires_grad):
        return ServingValue.apply(linear(x), aligned)
    return aligned
