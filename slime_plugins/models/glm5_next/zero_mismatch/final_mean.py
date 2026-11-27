"""Use the serving BI reduction for the final mHC stream contraction."""
import torch
from sglang.srt.hardware_backend.npu.batch_invariant_ops.npu_batch_invariant_ops import npu_mean_batch_invariant
from .autograd import ServingValue


def final_stream_mean(hidden_states, streams):
    if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
        return hidden_states.unflatten(-1, (streams, -1)).float().mean(-2).to(hidden_states.dtype)
    with torch.no_grad():
        aligned = npu_mean_batch_invariant(
            hidden_states.unflatten(-1, (streams, -1)), [-2],
        ).to(hidden_states.dtype)
    if torch.is_grad_enabled() and hidden_states.requires_grad:
        original = hidden_states.unflatten(-1, (streams, -1)).float().mean(-2).to(hidden_states.dtype)
        return ServingValue.apply(original, aligned)
    return aligned

