"""Transformer Engine row-linear with aligned GLM5-next forward values."""

import torch
from megatron.core.extensions.transformer_engine import TERowParallelLinear

from .npu_backend import ServingValue
from .projection_parity import local_eight_shard_projection


class AlignedTERowParallelLinear(TERowParallelLinear):
    def forward(self, x, *args, **kwargs):
        output, bias = super().forward(x, *args, **kwargs)
        if x.device.type != "npu" or x.dtype != torch.bfloat16 or not x.numel():
            return output, bias
        if bias is not None or self.use_bias:
            raise RuntimeError("GLM5-next aligned dense projection requires bias=False")
        aligned = local_eight_shard_projection(x, self.weight)
        if output.requires_grad:
            aligned = ServingValue.apply(output, aligned)
        return aligned, bias
