"""Serving KDA values with the established native training backward."""
import torch
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from ..npu_backend import chunk_kda as native_chunk_kda, FusedRMSNormGated as NativeFusedRMSNormGated, _boundaries
from .autograd import ServingValue

def shared_kda(q, k, v, *, g, beta, raw_beta, cu_seqlens, cp_context=None, **kwargs):
    if q.device.type != 'npu' or q.shape[0] != 1 or cp_context is not None:
        raise RuntimeError('Shared KDA requires an NPU packed batch with CP=1')
    cpu_bounds = kwargs.get('cu_seqlens_cpu')
    bounds = _boundaries(cpu_bounds if cpu_bounds is not None else cu_seqlens, q.shape[1])
    if bounds[0] != 0 or bounds[-1] != q.shape[1]:
        raise ValueError('Packed KDA boundaries must cover the complete input')
    a_log = kwargs['A_log']
    dt_bias = kwargs['dt_bias']
    with torch.no_grad():
        kernel = TritonKDAKernel()
        state = torch.zeros((2, v.shape[2], v.shape[3], k.shape[3]), device=q.device, dtype=torch.float32)
        slot = torch.tensor([1], device=q.device, dtype=torch.int32)
        pieces = []
        for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
            if begin == end:
                continue
            state.zero_()
            pieces.append(kernel.decode(q=q[:, begin:end].contiguous(), k=k[:, begin:end].contiguous(), v=v[:, begin:end].contiguous(), a=g[:, begin:end].contiguous(), b=raw_beta[:, begin:end].contiguous(), A_log=a_log, dt_bias=dt_bias, ssm_states=state, cache_indices=slot, query_start_loc=torch.tensor([0, end - begin], device=q.device, dtype=torch.int32), lower_bound=kwargs['lower_bound']))
        serving = torch.cat(pieces, dim=1) if pieces else v.new_empty(v.shape)
        if serving.shape != v.shape or serving.dtype != v.dtype:
            raise RuntimeError(f'Unexpected shared KDA output {serving.shape} {serving.dtype}')
    if torch.is_grad_enabled() and any((x.requires_grad for x in (q, k, v, g, beta, a_log, dt_bias))):
        native, _ = native_chunk_kda(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens, cp_context=cp_context, **kwargs)
        serving = ServingValue.apply(native, serving)
    return (serving, None)

class SharedGatedNorm(NativeFusedRMSNormGated):

    def forward(self, x, gate):
        if x.device.type != 'npu':
            return super().forward(x, gate)
        with torch.no_grad():
            serving = rms_norm_gated(x, gate, self.weight, None, activation='sigmoid', eps=self.eps)
        if torch.is_grad_enabled() and any((t.requires_grad for t in (x, gate, self.weight))):
            return ServingValue.apply(super().forward(x, gate), serving)
        return serving
