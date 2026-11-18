"""Ascend backend for the GLM5-next model.

KDA and causal convolution use AscendC forward/backward bridges. Gated
normalization uses PyTorch NPU operations; sparse MLA uses CANN AscendC.
CP exchanges sequence shards for head shards when the head count is divisible
by the CP size, and restores the local sequence layout after computation.
"""

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region


def _boundaries(cu, length):
    boundaries = [0, length] if cu is None else (cu.tolist() if hasattr(cu, "tolist") else list(cu))
    if length <= 0:
        raise ValueError("Packed input must contain at least one token")
    if len(boundaries) < 2 or any(not isinstance(value, int) for value in boundaries):
        raise ValueError("Packed boundaries must contain at least two integer offsets")
    if boundaries[0] != 0 or boundaries[-1] != length:
        raise ValueError(f"Packed boundaries do not cover {length} tokens: {boundaries}")
    if any(end < start for start, end in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("Packed boundaries must be nondecreasing")
    return boundaries


def _gather(x, context):
    if context is None:
        return x
    return gather_from_sequence_parallel_region(x.transpose(0, 1).contiguous(), group=context.group).transpose(0, 1)


def _local(x, context):
    if context is None:
        return x
    return x.chunk(context.group.size(), dim=1)[context.group.rank()].contiguous()


def _can_shard_heads(x, context):
    return context is not None and x.shape[0] == 1 and x.shape[2] % context.group.size() == 0


def _to_head_parallel(x, context):
    parts = x.chunk(context.group.size(), dim=2)
    return all_to_all(context.group, torch.cat(parts, dim=1).squeeze(0)).unsqueeze(0)


def _to_sequence_parallel(x, context):
    exchanged = all_to_all(context.group, x.squeeze(0)).unsqueeze(0)
    return torch.cat(exchanged.chunk(context.group.size(), dim=1), dim=2).contiguous()


def _head_slice(x, context):
    return x.chunk(context.group.size(), dim=0)[context.group.rank()].contiguous()


def build_cp_context(cu_seqlens, group, conv1d_kernel_size):
    return SimpleNamespace(cu_seqlens=cu_seqlens, group=group)


def _convolve(x, weight, activation, cu):
    # Dispatch Conv1d independently of KDA availability. The torch backend
    # can be selected explicitly through the environment setting.
    backend = os.environ.get("GLM53_CONV_BACKEND", "ascendc")
    if backend not in ("ascendc", "torch"):
        raise ValueError(f"Unsupported GLM53_CONV_BACKEND: {backend}")
    if backend == "ascendc" and x.device.type == "npu" and x.dtype in (torch.bfloat16, torch.float16):
        from .ascendc_conv1d import causal_conv1d_ascendc

        return causal_conv1d_ascendc(x, weight, activation, cu)
    return _convolve_torch(x, weight, activation, cu)


def _convolve_torch(x, weight, activation, cu):
    outputs = []
    for start, end in zip((bounds := _boundaries(cu, x.shape[1]))[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        part = x[:, start:end].transpose(1, 2)
        with torch.profiler.record_function("glm53::conv_fallback"):
            part = F.pad(part, (weight.shape[-1] - 1, 0))
            part = F.conv1d(part, weight[:, None], groups=weight.shape[0]).transpose(1, 2)
        if activation == "silu":
            part = F.silu(part)
        elif activation is not None:
            raise ValueError(f"Unsupported convolution activation: {activation}")
        outputs.append(part)
    return torch.cat(outputs, dim=1)


class ShortConvolution(torch.nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu"):
        super().__init__()
        if bias:
            raise ValueError("GLM5-next NPU convolution expects bias=False")
        self.kernel_size = (kernel_size,)
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x, cu_seqlens=None):
        return _convolve(x, self.weight.squeeze(1), self.activation, cu_seqlens), None


def causal_conv1d(x, weight, activation, cp_context):
    if _can_shard_heads(x, cp_context):
        out = _convolve(
            _to_head_parallel(x, cp_context), _head_slice(weight, cp_context), activation, cp_context.cu_seqlens
        )
        return _to_sequence_parallel(out, cp_context), None
    cu_seqlens = cp_context.cu_seqlens if cp_context is not None else None
    out = _convolve(_gather(x, cp_context), weight, activation, cu_seqlens)
    return _local(out, cp_context), None


class FusedRMSNormGated(torch.nn.Module):
    """Same parameter contract; use autograd-supported NPU operations."""

    def __init__(self, hidden_size, eps, activation="sigmoid"):
        super().__init__()
        if activation != "sigmoid":
            raise ValueError("GLM5-next gated norm expects sigmoid")
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, gate):
        if os.getenv("GLM53_TRAIN_ROLLOUT_PARITY") != "1":
            return self._native_forward(x, gate)

        from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated

        with torch.no_grad():
            serving = rms_norm_gated(x, gate, self.weight, None, activation="sigmoid", eps=self.eps)
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, gate, self.weight)):
            return ServingValue.apply(self._native_forward(x, gate), serving)
        return serving

    def _native_forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * gate.float().sigmoid()).to(x.dtype)


def chunk_kda(q, k, v, *, g, beta, cu_seqlens, cp_context=None, **kwargs):
    from .ascendc_kda import chunk_kda_ascendc

    shard_heads = _can_shard_heads(q, cp_context)
    if shard_heads:
        heads = q.shape[2]
        if kwargs.get("A_log") is not None:
            kwargs["A_log"] = _head_slice(kwargs["A_log"], cp_context)
        if kwargs.get("dt_bias") is not None:
            kwargs["dt_bias"] = _head_slice(kwargs["dt_bias"].reshape(heads, -1), cp_context).flatten()
        q, k, v, g, beta = [_to_head_parallel(x, cp_context) for x in (q, k, v, g, beta)]
    else:
        q, k, v, g, beta = [_gather(x, cp_context).contiguous() for x in (q, k, v, g, beta)]

    def local_output(out):
        return _to_sequence_parallel(out, cp_context) if shard_heads else _local(out, cp_context)

    cpu_bounds = kwargs.pop("cu_seqlens_cpu", None)
    bounds = _boundaries(cpu_bounds if cpu_bounds is not None else cu_seqlens, q.shape[1])
    mode = os.getenv("GLM53_KDA_SEQUENCE_MODE", "loop")
    if mode not in ("loop", "packed"):
        raise ValueError(f"Unsupported GLM53_KDA_SEQUENCE_MODE: {mode}")
    if mode == "packed":
        # Remove empty sequences without changing token positions.
        bounds = list(dict.fromkeys(bounds))
        with torch.profiler.record_function("glm53::kda_packed"):
            out, _ = chunk_kda_ascendc(q, k, v, g=g, beta=beta, cu_seqlens=bounds, **kwargs)
        return local_output(out), None
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        if start == end:
            continue
        out, _ = chunk_kda_ascendc(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            **kwargs,
        )
        outputs.append(out)
    return local_output(torch.cat(outputs, dim=1)), None


class SparseMLA:
    @staticmethod
    def apply(q, kv, indices, scaling):
        # Keep CANN SMLA loading lazy for rollout/driver-only imports.
        from .ascendc_smla import SparseMLA as AscendCSparseMLA

        return AscendCSparseMLA.apply(q, kv, indices, scaling)

class ServingValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, serving):
        return serving

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def project_and_convolve_for_serving(module, hidden, cu, cu_cpu, cp_context):
    names = ("q", "k", "v")
    convolutions = [getattr(module, f"{name}_conv1d") for name in names]
    projections = [getattr(module, f"{name}_proj") for name in names]
    if cp_context is not None or hidden.device.type != "npu" or hidden.dtype != torch.bfloat16:
        raise RuntimeError("Shared KDA inputs require BF16 NPU training with CP=1")

    qkv = [projection(hidden) for projection in projections]
    beta = module.b_proj(hidden)
    forget = module.f_b_proj(module.f_a_proj(hidden))
    norm_gate = module.g_b_proj(module.g_a_proj(hidden))

    if torch.is_grad_enabled() and any(
        x.requires_grad or conv.weight.requires_grad for x, conv in zip(qkv, convolutions, strict=True)
    ):
        native = [conv(x, cu_seqlens=cu)[0] for x, conv in zip(qkv, convolutions, strict=True)]
    else:
        native = None

    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_native

    bounds = _boundaries(cu_cpu if cu_cpu is not None else cu, hidden.shape[1])
    with torch.no_grad():
        weight = torch.cat([conv.weight.squeeze(1) for conv in convolutions])
        mixed = torch.cat(qkv, dim=-1)
        outputs = [
            causal_conv1d_fn_native(
                mixed[:, begin:end].transpose(1, 2).contiguous(), weight, activation="silu"
            )[0].transpose(1, 2)
            for begin, end in zip(bounds[:-1], bounds[1:], strict=True) if begin < end
        ]
        convolved = torch.cat(outputs, dim=1).split([x.shape[-1] for x in qkv], dim=-1)
    if native is not None:
        convolved = [
            ServingValue.apply(original, output.contiguous())
            for original, output in zip(native, convolved, strict=True)
        ]

    qkv = [x.unflatten(-1, (module.heads, module.head_dim)) for x in convolved]
    return qkv, forget.unflatten(-1, (module.heads, module.head_dim)), beta, norm_gate


def serving_chunk_kda(q, k, v, *, g, beta, raw_beta, cu_seqlens, cp_context=None, **kwargs):
    if q.device.type != "npu" or q.shape[0] != 1 or cp_context is not None:
        raise RuntimeError("Shared KDA requires an NPU packed batch with CP=1")
    cpu_bounds = kwargs.get("cu_seqlens_cpu")
    bounds = _boundaries(cpu_bounds if cpu_bounds is not None else cu_seqlens, q.shape[1])
    if bounds[0] != 0 or bounds[-1] != q.shape[1]:
        raise ValueError("Packed KDA boundaries must cover the complete input")
    a_log = kwargs["A_log"]
    dt_bias = kwargs["dt_bias"]
    from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

    with torch.no_grad():
        kernel = TritonKDAKernel()
        if os.getenv("GLM53_KDA_GROUPED_RECURRENCE") == "1":
            count = len(bounds) - 1
            state = torch.zeros((count + 1, v.shape[2], v.shape[3], k.shape[3]),
                                device=q.device, dtype=torch.float32)
            serving = kernel.decode(
                q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
                a=g.contiguous(), b=raw_beta.contiguous(), A_log=a_log,
                dt_bias=dt_bias, ssm_states=state,
                cache_indices=torch.arange(1, count + 1, device=q.device, dtype=torch.int32),
                query_start_loc=torch.tensor(bounds, device=q.device, dtype=torch.int32),
                lower_bound=kwargs["lower_bound"],
            ) if q.shape[1] else v.new_empty(v.shape)
        else:
            state = torch.zeros((2, v.shape[2], v.shape[3], k.shape[3]),
                                device=q.device, dtype=torch.float32)
            slot = torch.tensor([1], device=q.device, dtype=torch.int32)
            pieces = []
            for begin, end in zip(bounds[:-1], bounds[1:], strict=True):
                if begin == end:
                    continue
                state.zero_()
                pieces.append(kernel.decode(
                    q=q[:, begin:end].contiguous(), k=k[:, begin:end].contiguous(),
                    v=v[:, begin:end].contiguous(), a=g[:, begin:end].contiguous(),
                    b=raw_beta[:, begin:end].contiguous(), A_log=a_log,
                    dt_bias=dt_bias, ssm_states=state, cache_indices=slot,
                    query_start_loc=torch.tensor([0, end - begin], device=q.device, dtype=torch.int32),
                    lower_bound=kwargs["lower_bound"],
                ))
            serving = torch.cat(pieces, dim=1) if pieces else v.new_empty(v.shape)
        if serving.shape != v.shape or serving.dtype != v.dtype:
            raise RuntimeError(f"Unexpected shared KDA output {serving.shape} {serving.dtype}")

    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v, g, beta, a_log, dt_bias)):
        native, _ = chunk_kda(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens, cp_context=cp_context, **kwargs)
        serving = ServingValue.apply(native, serving)
    return serving, None
