"""Use SGLang's BF16 NPU MoE forward with live partitioned trainer weights."""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from megatron.core.transformer.moe.moe_layer import MoELayer
from sglang.srt.hardware_backend.npu.moe.finalize_routing import NPUFinalizeRouting
from sglang.srt.hardware_backend.npu.moe.init_routing import NPUMoEInitRouting_v2

from .npu_backend import ServingValue


def all_expert_weights(linear, ep_group, local_experts, dtype):
    parts = linear._get_weight_tensors()
    if len(parts) == 1 and parts[0].ndim == 3:
        local = parts[0].to(dtype).contiguous()
    else:
        local = torch.stack(parts).to(dtype)
    if local.ndim != 3 or local.shape[0] != local_experts:
        raise ValueError(f"Unexpected expert weight shape {local.shape}")
    size = ep_group.size()
    if size == 1:
        return local
    result = local.new_empty((size * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(result, local, group=ep_group)
    return result


def grouped_linear(x, weight, counts):
    return torch.ops.npu.npu_grouped_matmul(
        x=[x], weight=[weight.transpose(1, 2)], split_item=2, group_list_type=1,
        group_type=0, group_list=counts, output_dtype=x.dtype,
    )[0]


def swiglu(x, limit):
    gate, up = x.chunk(2, dim=-1)
    return torch_npu.npu_swiglu(torch.cat((gate.clamp(max=limit), up.clamp(-limit, limit)), dim=-1))


def eight_shard_moe(
    layer, x, permuted, expanded, counts, probabilities, ids,
    *, expert_weights, grouped_linear, swiglu,
):
    from sglang.srt.hardware_backend.npu.glm53_parity import ordered_shard_sum, projection_shard

    experts = layer.experts
    ep = layer.ep_group
    num_local_experts = layer.num_local_experts
    w1 = expert_weights(experts.linear_fc1, ep, num_local_experts, x.dtype)
    w2 = expert_weights(experts.linear_fc2, ep, num_local_experts, x.dtype)
    gate, up = w1.chunk(2, dim=1)

    shared = layer.shared_experts
    shared_gate, shared_up = shared.linear_fc1.weight.chunk(2, dim=0)
    shared_down = shared.linear_fc2.weight
    if gate.shape[1] != w2.shape[2] or shared_gate.shape[0] != shared_down.shape[1]:
        raise ValueError("The MoE gate/up weights and down projection have different widths")
    if gate.shape[1] % 8 or shared_down.shape[1] % 8:
        raise ValueError("The MoE intermediate width must divide evenly into eight ranks")

    limit = layer.config.activation_func_clamp_value
    scale = layer.config.moe_router_topk_scaling_factor
    finalize = NPUFinalizeRouting(drop_pad_mode=2)
    wide_activation = None
    if (
        os.getenv("GLM53_TUNE_MOE_UP") == "1"
        and permuted.device.type == "npu"
        and permuted.dtype == torch.bfloat16
        and 0 < permuted.shape[0] <= int(os.getenv("GLM53_TUNE_MOE_UP_ROWS", "16384"))
    ):
        wide_activation = swiglu(grouped_linear(permuted, w1, counts), limit).chunk(8, dim=-1)

    mixtures = []
    shards = zip(
        gate.chunk(8, dim=1), up.chunk(8, dim=1), w2.chunk(8, dim=2),
        shared_gate.chunk(8, dim=0), shared_up.chunk(8, dim=0), shared_down.chunk(8, dim=1),
        strict=True,
    )
    for rank, (g, u, w, sg, su, sd) in enumerate(shards):
        if wide_activation is None:
            gate_up = grouped_linear(permuted, torch.cat((g, u), dim=1), counts)
            activated = swiglu(gate_up, limit)
        else:
            activated = wide_activation[rank].contiguous()
        expert_out = grouped_linear(activated, w.contiguous(), counts)
        routed = finalize._finalize_routing(
            expert_out, probabilities.to(x.dtype), expanded, ids,
        )
        shared_up_out = projection_shard(x, torch.cat((sg, su), dim=0))
        shared_out = projection_shard(swiglu(shared_up_out, limit), sd)
        mixtures.append(routed * scale + shared_out)
    return ordered_shard_sum(tuple(mixtures))


def serving_ep2_forward(layer, x, ids, probabilities):
    """Apply each serving rank's BF16 scale and shared expert before reducing."""
    ep = layer.ep_group
    local_experts = layer.num_local_experts
    config = layer.config
    if ep.size() != 2 or config.num_moe_experts != 2 * local_experts:
        raise RuntimeError("The serving EP2 forward expects two contiguous expert partitions")

    expert_fc1 = all_expert_weights(layer.experts.linear_fc1, ep, local_experts, x.dtype)
    expert_fc2 = all_expert_weights(layer.experts.linear_fc2, ep, local_experts, x.dtype)
    shared = layer.shared_experts
    gate, up = shared.linear_fc1.weight.chunk(2, dim=0)
    gate_shards = gate.chunk(2, dim=0)
    up_shards = up.chunk(2, dim=0)
    down_shards = shared.linear_fc2.weight.chunk(2, dim=1)
    init = NPUMoEInitRouting_v2()
    finalize = NPUFinalizeRouting(drop_pad_mode=2)
    limit = config.activation_func_clamp_value
    scale = config.moe_router_topk_scaling_factor
    partials = []
    for rank in range(2):
        begin = rank * local_experts
        end = begin + local_experts
        local_ids = torch.where((ids >= begin) & (ids < end), ids - begin, -1)
        permuted, expanded, counts, _ = init._init_routing(
            x, local_ids, local_experts, config.moe_router_topk,
        )
        gate_up = grouped_linear(permuted, expert_fc1[begin:end], counts)
        expert_output = grouped_linear(swiglu(gate_up, limit), expert_fc2[begin:end], counts)
        routed = finalize._finalize_routing(expert_output, probabilities, expanded, local_ids)
        shared_gate_up = F.linear(x, torch.cat((gate_shards[rank], up_shards[rank]), dim=0))
        shared_output = F.linear(swiglu(shared_gate_up, limit), down_shards[rank].contiguous())
        partials.append(routed * scale + shared_output)
    return partials[0] + partials[1]


@torch.no_grad()
def serving_forward(layer, hidden_states):
    config = layer.config
    router = layer.router
    ep = layer.ep_group
    local_experts = layer.num_local_experts
    expected = list(range(ep.rank() * local_experts, (ep.rank() + 1) * local_experts))
    if list(layer.local_expert_indices) != expected or config.moe_mlp_glu_interleave_size is not None:
        raise RuntimeError("The shared NPU MoE expects contiguous expert ownership and split gate/up weights")
    if config.moe_router_score_function != "sigmoid" or layer.shared_experts is None:
        raise RuntimeError("The shared NPU MoE expects sigmoid routing and a shared expert")
    x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    logits = F.linear(x, router.weight.to(x.dtype), None).float()
    k = config.moe_router_topk
    probabilities, ids, _ = torch.ops.npu.npu_moe_gating_top_k(
        logits, k=k, bias=router.expert_bias.float(), k_group=1, group_count=1,
        group_select_mode=1, renorm=True, norm_type=1, routed_scaling_factor=1.0,
        eps=1e-20,
    )
    ids = ids.to(torch.int32)
    if os.getenv("GLM53_REPLAY_SERVING_MOE_EP2") == "1":
        return serving_ep2_forward(layer, x, ids, probabilities.to(x.dtype)).reshape_as(hidden_states)

    permuted, expanded, counts, _ = NPUMoEInitRouting_v2()._init_routing(
        x, ids, config.num_moe_experts, k
    )
    if os.getenv("GLM53_STRICT_MOE_PROJECT") == "1":
        return eight_shard_moe(
            layer, x, permuted, expanded, counts, probabilities, ids,
            expert_weights=all_expert_weights, grouped_linear=grouped_linear, swiglu=swiglu,
        ).reshape_as(hidden_states)

    experts = layer.experts
    gate_up = grouped_linear(
        permuted, all_expert_weights(experts.linear_fc1, ep, local_experts, x.dtype), counts
    )
    activated = swiglu(gate_up, config.activation_func_clamp_value)
    expert_output = grouped_linear(
        activated, all_expert_weights(experts.linear_fc2, ep, local_experts, x.dtype), counts
    )
    routed = NPUFinalizeRouting(drop_pad_mode=2)._finalize_routing(
        expert_output, probabilities.to(x.dtype), expanded, ids
    )
    shared = layer.shared_experts
    shared_gate_up = F.linear(x, shared.linear_fc1.weight)
    shared_output = F.linear(swiglu(shared_gate_up, config.activation_func_clamp_value),
                             shared.linear_fc2.weight)
    return (routed * config.moe_router_topk_scaling_factor + shared_output).reshape_as(hidden_states)


class Glm5NextAlignedMoELayer(MoELayer):
    def forward(self, hidden_states, intermediate_tensors=None, padding_mask=None):
        if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
            return super().forward(hidden_states, intermediate_tensors, padding_mask)
        if intermediate_tensors is not None:
            raise RuntimeError("The shared NPU MoE expects a complete eager training forward")
        serving = serving_forward(self, hidden_states)
        original, bias = super().forward(hidden_states, intermediate_tensors, padding_mask)
        if bias is not None:
            raise RuntimeError("The shared NPU MoE expects bias-free expert projections")
        if original.requires_grad:
            serving = ServingValue.apply(original, serving)
        return serving, None
