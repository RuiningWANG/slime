"""Align training KDA short convolution with the SGLang forward."""
import torch
from .autograd import ServingValue

def project_and_convolve(module, hidden, cu, cu_cpu, cp_context):
    names = ('q', 'k', 'v')
    convolutions = [getattr(module, f'{name}_conv1d') for name in names]
    projections = [getattr(module, f'{name}_proj') for name in names]
    if cp_context is not None or hidden.device.type != 'npu' or hidden.dtype != torch.bfloat16:
        raise RuntimeError('Shared KDA inputs require BF16 NPU training with CP=1')
    qkv = [projection(hidden) for projection in projections]
    beta = module.b_proj(hidden)
    forget = module.f_b_proj(module.f_a_proj(hidden))
    norm_gate = module.g_b_proj(module.g_a_proj(hidden))
    if torch.is_grad_enabled() and any((x.requires_grad or conv.weight.requires_grad for x, conv in zip(qkv, convolutions, strict=True))):
        native = [conv(x, cu_seqlens=cu)[0] for x, conv in zip(qkv, convolutions, strict=True)]
    else:
        native = None
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_native
    from slime_plugins.models.glm5_next.npu_backend import _boundaries
    bounds = _boundaries(cu_cpu if cu_cpu is not None else cu, hidden.shape[1])
    with torch.no_grad():
        weight = torch.cat([conv.weight.squeeze(1) for conv in convolutions])
        mixed = torch.cat(qkv, dim=-1)
        outputs = [causal_conv1d_fn_native(mixed[:, begin:end].transpose(1, 2).contiguous(), weight, activation='silu')[0].transpose(1, 2) for begin, end in zip(bounds[:-1], bounds[1:], strict=True) if begin < end]
        convolved = torch.cat(outputs, dim=1).split([x.shape[-1] for x in qkv], dim=-1)
    if native is not None:
        if torch.is_grad_enabled():
            convolved = [ServingValue.apply(original, output.contiguous()) for original, output in zip(native, convolved, strict=True)]
    qkv = [x.unflatten(-1, (module.heads, module.head_dim)) for x in convolved]
    return (qkv, forget.unflatten(-1, (module.heads, module.head_dim)), beta, norm_gate)
