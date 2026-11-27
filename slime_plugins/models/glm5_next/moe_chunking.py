"""Bound native GLM MoE workspaces without replacing its math kernels.

A chunk size of zero selects the unchunked forward pass.
Each complete MoE chunk is checkpointed with full replay: the mutable dispatcher is
reconstructed immediately before its backward, not reused from another chunk.
"""
import os
from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _FixedRoutes:
    def __init__(self, routes):
        self.routes = routes

    def pop_forward(self):
        return self.routes

    def pop_backward(self):
        return self.routes


@contextmanager
def _bind_routes(router, routes):
    if routes is None:
        yield
        return
    from slime.utils import routing_replay as rr

    original = router.routing_replay
    previous_global = rr.ROUTING_REPLAY
    router.routing_replay = _FixedRoutes(routes)
    try:
        yield
    finally:
        router.routing_replay = original
        rr.set_routing_replay(previous_global)


def _packed_routes(module, length, padded_length):
    stage = os.environ.get("ROUTING_REPLAY_STAGE", "fallthrough")
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") != "1" or stage == "fallthrough":
        return None
    if stage not in ("replay_forward", "replay_backward"):
        raise ValueError("Chunked MoE supports rollout replay or fallthrough, not record mode")
    replay = module.router.routing_replay
    routes = replay.pop_forward() if stage == "replay_forward" else replay.pop_backward()
    if routes.shape != (length, module.config.moe_router_topk):
        raise ValueError(f"Packed routes {routes.shape} do not match {length} tokens")
    if padded_length > length:
        dummy = torch.arange(routes.shape[1], device=routes.device, dtype=routes.dtype)
        routes = torch.cat((routes, dummy.expand(padded_length - length, -1)))
    return routes


def _checkpoint(fn, *args):
    # Outer layer recomputation runs the initial forward under no_grad. During
    # its backward recompute this inner checkpoint bounds expert activations.
    if torch.is_grad_enabled() and any(x.requires_grad for x in args):
        return checkpoint(fn, *args, use_reentrant=False, early_stop=False, preserve_rng_state=True)
    return fn(*args)


def _moe_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
    size = self._glm53_dispatch_chunk
    if not size:
        return self._glm53_original_forward(hidden_states, intermediate_tensors, padding_mask)
    if intermediate_tensors is not None:
        raise ValueError("Partial MoE execution is incompatible with chunking")
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, 1, shape[-1])
    length = x.shape[0]
    count = torch.tensor([length], dtype=torch.int64, device=x.device)
    if torch.distributed.get_world_size(self.ep_group) > 1:
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX, group=self.ep_group)
    padded_length = max(1, int(count.item()))
    routes = _packed_routes(self, length, padded_length)
    if padded_length > length:
        x = F.pad(x, (0, 0, 0, 0, 0, padded_length - length))
    # Native MoELayer accepts [batch, sequence] masks and transposes internally.
    mask = None if padding_mask is None else padding_mask.transpose(0, 1).reshape(1, length)
    if padded_length > length and mask is not None:
        mask = F.pad(mask, (0, padded_length - length), value=False)
    outputs = []
    for start in range(0, padded_length, size):
        end = min(start + size, padded_length)
        chunk_routes = None if routes is None else routes[start:end]
        chunk_mask = None if mask is None else mask[:, start:end]

        def run(part, chunk_routes=chunk_routes, chunk_mask=chunk_mask):
            with _bind_routes(self.router, chunk_routes):
                output, bias = self._glm53_original_forward(part, padding_mask=chunk_mask)
            if bias is not None:
                raise ValueError("Chunked GLM MoE requires bias-free experts")
            return output

        outputs.append(_checkpoint(run, x[start:end]))
    return torch.cat(outputs, dim=0)[:length].reshape(shape), None


def _expert_forward(self, x, counts, probs, output_buffer=None, grad_input_buffer=None):
    size = self._glm53_expert_chunk
    if not size:
        return self._glm53_original_forward(x, counts, probs, output_buffer, grad_input_buffer)
    # One D2H sync instead of counts.max().item() followed by counts.tolist().
    lengths = counts.tolist()
    if max(lengths) <= size:
        return self._glm53_original_forward(x, counts, probs, output_buffer, grad_input_buffer)
    if output_buffer is not None or grad_input_buffer is not None:
        raise ValueError("Chunked experts do not support zero-copy output buffers")
    x_parts = x.split(lengths, dim=0)
    p_parts = probs.split(lengths, dim=0)
    outputs = [[] for _ in lengths]
    for start in range(0, max(lengths), size):
        sizes = [max(0, min(size, n - start)) for n in lengths]
        chunk_counts = counts.new_tensor(sizes)
        chunk_x = torch.cat([v[start:start + n] for v, n in zip(x_parts, sizes)], dim=0)
        chunk_p = torch.cat([v[start:start + n] for v, n in zip(p_parts, sizes)], dim=0)

        def run(a, p, chunk_counts=chunk_counts):
            output, bias = self._glm53_original_forward(a, chunk_counts, p)
            if bias is not None:
                raise ValueError("Chunked GLM experts require bias-free output")
            return output

        result = _checkpoint(run, chunk_x, chunk_p)
        for parts, value in zip(outputs, result.split(sizes, dim=0)):
            parts.append(value)
    return torch.cat([torch.cat(parts, dim=0) for parts in outputs], dim=0), None


def install_moe_chunking(module, dispatch_chunk=0, expert_chunk=0):
    """Install on an existing native MoELayer; parameter names remain unchanged."""
    if dispatch_chunk < 0 or expert_chunk < 0:
        raise ValueError("Chunk sizes must be nonnegative")
    if not dispatch_chunk and not expert_chunk:
        return
    if hasattr(module, "_glm53_dispatch_chunk"):
        raise ValueError("MoE chunking is already installed")
    config = module.config
    for name in ("fp8", "fp4", "moe_shared_expert_overlap", "moe_expert_capacity_factor",
                 "moe_pad_expert_input_to_capacity", "moe_router_enable_expert_bias_update",
                 "overlap_dispatch_backward_with_experts_wgrad", "delay_wgrad_compute"):
        if getattr(config, name, False):
            raise ValueError(f"Chunked MoE does not support {name}")
    for name in ("moe_aux_loss_coeff", "moe_z_loss_coeff", "moe_router_bias_update_rate"):
        if getattr(config, name, 0):
            raise ValueError(f"Chunked MoE requires {name}=0")
    if config.tensor_model_parallel_size != 1 or config.expert_tensor_parallel_size != 1:
        raise ValueError("Chunked native GLM currently requires TP=ETP=1")
    if module.moe_layer_recompute or module.experts.activation_recompute:
        raise ValueError("Use the recipe's full layer recompute, not selective MoE recompute")
    module._glm53_original_forward = module.forward
    module._glm53_dispatch_chunk = dispatch_chunk
    module.forward = MethodType(_moe_forward, module)
    module.experts._glm53_original_forward = module.experts.forward
    module.experts._glm53_expert_chunk = expert_chunk
    module.experts.forward = MethodType(_expert_forward, module.experts)


def _aligned_moe_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
    size = self._glm53_dispatch_chunk
    if not size:
        return self._native_chunk_forward(hidden_states, intermediate_tensors, padding_mask)
    if intermediate_tensors is not None:
        raise ValueError("Partial MoE execution is incompatible with chunking")
    shape = hidden_states.shape
    x = hidden_states.reshape(-1, 1, shape[-1])
    length = x.shape[0]
    count = torch.tensor([length], dtype=torch.int64, device=x.device)
    if torch.distributed.get_world_size(self.ep_group) > 1:
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.MAX, group=self.ep_group)
    padded_length = max(1, int(count.item()))
    routes = _packed_routes(self, length, padded_length)
    if padded_length > length:
        x = F.pad(x, (0, 0, 0, 0, 0, padded_length - length))
    # Native MoELayer accepts [batch, sequence] masks and transposes internally.
    mask = None if padding_mask is None else padding_mask.transpose(0, 1).reshape(1, length)
    if padded_length > length and mask is not None:
        mask = F.pad(mask, (0, padded_length - length), value=False)
    outputs = []
    for start in range(0, padded_length, size):
        end = min(start + size, padded_length)
        chunk_routes = None if routes is None else routes[start:end]
        chunk_mask = None if mask is None else mask[:, start:end]

        def run(part, chunk_routes=chunk_routes, chunk_mask=chunk_mask):
            with _bind_routes(self.router, chunk_routes):
                output, bias = self._native_chunk_forward(part, padding_mask=chunk_mask)
            if bias is not None:
                raise ValueError("Chunked GLM MoE requires bias-free experts")
            return output

        outputs.append(_checkpoint(run, x[start:end]))
    return torch.cat(outputs, dim=0)[:length].reshape(shape), None


def _aligned_expert_forward(self, x, counts, probs, output_buffer=None, grad_input_buffer=None):
    size = self._glm53_expert_chunk
    if not size:
        return self._native_chunk_forward(x, counts, probs, output_buffer, grad_input_buffer)
    # One D2H sync instead of counts.max().item() followed by counts.tolist().
    lengths = counts.tolist()
    if max(lengths) <= size:
        return self._native_chunk_forward(x, counts, probs, output_buffer, grad_input_buffer)
    if output_buffer is not None or grad_input_buffer is not None:
        raise ValueError("Chunked experts do not support zero-copy output buffers")
    x_parts = x.split(lengths, dim=0)
    p_parts = probs.split(lengths, dim=0)
    outputs = [[] for _ in lengths]
    for start in range(0, max(lengths), size):
        sizes = [max(0, min(size, n - start)) for n in lengths]
        chunk_counts = counts.new_tensor(sizes)
        chunk_x = torch.cat([v[start:start + n] for v, n in zip(x_parts, sizes)], dim=0)
        chunk_p = torch.cat([v[start:start + n] for v, n in zip(p_parts, sizes)], dim=0)

        def run(a, p, chunk_counts=chunk_counts):
            output, bias = self._native_chunk_forward(a, chunk_counts, p)
            if bias is not None:
                raise ValueError("Chunked GLM experts require bias-free output")
            return output

        result = _checkpoint(run, chunk_x, chunk_p)
        for parts, value in zip(outputs, result.split(sizes, dim=0)):
            parts.append(value)
    return torch.cat([torch.cat(parts, dim=0) for parts in outputs], dim=0), None


from functools import partial
from dataclasses import replace
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import TEGroupedMLP
from .zero_mismatch.autograd import ServingValue
from .zero_mismatch.ep16 import serving_forward_ep16


class ChunkedTEGroupedMLP(TEGroupedMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glm53_expert_chunk = int(os.environ.get("GLM53_EXPERT_TOKEN_CHUNK_SIZE", "0"))
        if self._glm53_expert_chunk < 0:
            raise ValueError("Expert chunk size must be nonnegative")

    def _native_chunk_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, counts, probs, output_buffer=None, grad_input_buffer=None):
        return _aligned_expert_forward(self, x, counts, probs, output_buffer, grad_input_buffer)


class AlignedMoELayer(MoELayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glm53_dispatch_chunk = int(os.environ.get("GLM53_MOE_DISPATCH_CHUNK_SIZE", "0"))
        if self._glm53_dispatch_chunk < 0:
            raise ValueError("Dispatch chunk size must be nonnegative")
        config = self.config
        for name in ("fp8", "fp4", "moe_shared_expert_overlap", "moe_expert_capacity_factor",
                     "moe_pad_expert_input_to_capacity", "moe_router_enable_expert_bias_update",
                     "overlap_dispatch_backward_with_experts_wgrad", "delay_wgrad_compute"):
            if getattr(config, name, False):
                raise ValueError(f"Chunked MoE does not support {name}")
        for name in ("moe_aux_loss_coeff", "moe_z_loss_coeff", "moe_router_bias_update_rate"):
            if getattr(config, name, 0):
                raise ValueError(f"Chunked MoE requires {name}=0")
        if config.tensor_model_parallel_size != 1 or config.expert_tensor_parallel_size != 1:
            raise ValueError("Chunked GLM requires TP=ETP=1")
        if self.moe_layer_recompute or self.experts.activation_recompute:
            raise ValueError("Use full layer recompute")

    def _native_chunk_forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
            return super().forward(hidden_states, intermediate_tensors, padding_mask)
        if intermediate_tensors is not None:
            raise RuntimeError("Aligned MoE requires a complete eager training forward")
        serving = serving_forward_ep16(self, hidden_states)
        native, bias = super().forward(hidden_states, intermediate_tensors, padding_mask)
        if bias is not None:
            raise RuntimeError("Aligned MoE requires bias-free experts")
        return (ServingValue.apply(native, serving) if native.requires_grad else serving), None

    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        return _aligned_moe_forward(self, hidden_states, intermediate_tensors, padding_mask)


def aligned_moe_builder(builder):
    """Select source-defined classes through Megatron's normal builder API."""
    if not isinstance(builder, partial) or builder.func is not MoELayer:
        raise TypeError("Expected the pinned Megatron MoELayer builder")
    kw = dict(builder.keywords)
    submodules = kw["submodules"]
    experts = submodules.experts
    if not isinstance(experts, partial) or experts.func is not TEGroupedMLP:
        raise TypeError("Expected the pinned TEGroupedMLP builder")
    expert_builder = partial(ChunkedTEGroupedMLP, *experts.args, **experts.keywords)
    kw["submodules"] = replace(submodules, experts=expert_builder)
    return partial(AlignedMoELayer, *builder.args, **kw)
