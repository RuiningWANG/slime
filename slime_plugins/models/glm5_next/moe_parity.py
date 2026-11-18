"""Use SGLang's BF16 NPU MoE forward with live partitioned trainer weights."""

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
    permuted, expanded, counts, _ = NPUMoEInitRouting_v2()._init_routing(
        x, ids, config.num_moe_experts, k
    )
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
