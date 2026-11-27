"""Experiment for the BF16 routed and shared expert combination with EP16."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluWithLimit
from sglang.srt.hardware_backend.npu.moe.routing import npu_moe_init_routing, npu_moe_finalize_routing
from sglang.srt.hardware_backend.npu.moe.matmul import GroupedMatmul


@dataclass
class LocalExpertWeights:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor


def local_routed(hidden, expert_ids, probabilities, weights, *, ep_rank, limit):
    """Apply the Ascend standard dispatcher mask and the BF16 expert kernels."""
    nlocal = weights.w13_weight.shape[0]
    local_ids = expert_ids.to(torch.int32) - ep_rank * nlocal
    own = (local_ids >= 0) & (local_ids < nlocal)
    local_ids = torch.where(own, local_ids, -1)
    routes = npu_moe_init_routing(
        hidden, local_ids, nlocal, active_num=local_ids.numel()
    )
    matmul = GroupedMatmul()
    gate_up = matmul.forward(weights, "w13", routes.hidden_states, routes.expert_tokens, hidden.dtype, 1, False)
    activated, _ = NPUSwigluWithLimit(limit)._apply_activation(gate_up)
    expert_output = matmul.forward(weights, "w2", activated, routes.expert_tokens, hidden.dtype, 1, False)
    return npu_moe_finalize_routing(
        expert_output, routes.expanded_row_idx, probabilities.to(hidden.dtype), local_ids
    )


def local_shared(hidden, weight_in, weight_out, limit):
    gate_up = F.linear(hidden, weight_in)
    act, _ = NPUSwigluWithLimit(limit)._apply_activation(gate_up)
    return F.linear(act, weight_out)


def local_contribution(routed, shared, scale):
    return routed * scale + shared


def reduce_token_partials(partials, sizes, group):
    """Use the same equal/uneven HCCL branches as SGLang GroupCoordinator."""
    if len(sizes)!=group.size() or sum(sizes)!=partials.shape[0]:
        raise ValueError('Token splits do not match the EP rank group')
    local=partials.new_empty((sizes[group.rank()],*partials.shape[1:]))
    if all(n==sizes[0] for n in sizes):
        dist.reduce_scatter_tensor(local,partials,group=group)
    else:
        import torch_npu

        torch_npu.distributed.reduce_scatter_tensor_uneven(
            local,partials,input_split_sizes=sizes,group=group)
    return local
