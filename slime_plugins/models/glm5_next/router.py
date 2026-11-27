"""GLM5-next router projection aligned with the default NPU inference path."""

from types import MethodType

import torch
import torch.nn.functional as F


def inference_aligned_gating(self, hidden_states):
    # Inference rounds the linear projection to the model dtype, then promotes
    # logits for FP32 sigmoid, correction bias, and top-k weight normalization.
    # Cast views only: keep parameter storage and optimizer gradients intact.
    weight = self.weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
    bias = self.bias
    if bias is not None:
        bias = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat, weight, bias)
    return logits.reshape(*hidden_states.shape[:-1], -1).float()


def install_inference_aligned_router(router):
    router.gating = MethodType(inference_aligned_gating, router)
