"""Training candidate for 288-expert BF16 EP16 with SGLang-style partial sums.

Activation exchange replaces the previous all-gather of every expert weight.
The existing shared_training_moe wrapper still runs the original training
forward and uses it as the backward source.
"""
import os

import torch
import torch.distributed as dist



def stacked_local_weights(linear, local_experts, dtype):
    parts = linear._get_weight_tensors()
    if len(parts) == 1 and parts[0].ndim == 3:
        result = parts[0].to(dtype).contiguous()
    else:
        result = torch.stack(parts).to(dtype)
    if result.ndim != 3 or result.shape[0] != local_experts:
        raise RuntimeError(f'Unexpected local expert weight layout: {tuple(result.shape)}')
    return result


def gather_train_tokens(hidden, group):
    """Gather the per-rank packed rows and return the rank splits."""
    local_count = torch.tensor([hidden.shape[0]],dtype=torch.int64,device=hidden.device)
    all_counts = local_count.new_empty(group.size())
    dist.all_gather_into_tensor(all_counts,local_count,group=group)
    counts = all_counts.tolist()
    max_count = max(counts)
    if max_count==0:
        return hidden,counts
    padded = hidden.new_zeros((max_count,hidden.shape[-1]))
    padded[:hidden.shape[0]].copy_(hidden)
    gathered = padded.new_empty((max_count*group.size(),hidden.shape[-1]))
    dist.all_gather_into_tensor(gathered,padded,group=group)
    # DP lane rows stay grouped, in rank order. Padding is excluded from expert counts.
    active = torch.cat([gathered[i*max_count:i*max_count+n] for i,n in enumerate(counts)])
    return active,counts


@torch.no_grad()
def serving_forward_ep16(layer, hidden_states):
    from .ep16_local import LocalExpertWeights, local_contribution, local_routed, local_shared, reduce_token_partials

    cfg = layer.config
    group = layer.ep_group
    rank = group.rank()
    count = layer.num_local_experts
    if group.size()!=16 or count!=18 or cfg.num_moe_experts!=288:
        raise RuntimeError('This candidate expects the eight-node 16-rank, 288-expert layout')
    if list(layer.local_expert_indices)!=list(range(rank*count,(rank+1)*count)):
        raise RuntimeError('Expert ownership differs from the serving rank layout')
    if cfg.moe_mlp_glu_interleave_size is not None or layer.shared_experts is None:
        raise RuntimeError('Expected separate gate/up weights and one unfused shared expert')
    if cfg.moe_router_score_function!='sigmoid' or cfg.moe_router_topk!=8:
        raise RuntimeError('Expected the full GLM-5.3 sigmoid TopK 8 gate')

    local = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    x, counts = gather_train_tokens(local,group)
    if x.shape[0]==0:
        return hidden_states

    logits = torch.nn.functional.linear(x,layer.router.weight.to(x.dtype),None).float()
    probs,ids,_ = torch.ops.npu.npu_moe_gating_top_k(
        logits,k=cfg.moe_router_topk,bias=layer.router.expert_bias.float(),
        k_group=1,group_count=1,group_select_mode=1,renorm=True,norm_type=1,
        routed_scaling_factor=1.0,eps=1e-20)
    experts = layer.experts
    weights = LocalExpertWeights(
        stacked_local_weights(experts.linear_fc1,count,x.dtype),
        stacked_local_weights(experts.linear_fc2,count,x.dtype))
    routed = local_routed(x,ids,probs,weights,ep_rank=rank,limit=cfg.activation_func_clamp_value)

    if os.environ.get("SGLANG_SHARED_EXPERT_TP1", "0") == "1":
        # GLM5Next uses ffn_boundary, not MoeRowTransport. Its replicated TP1
        # shared output is added on TP rank 0 before the BF16 partial sum.
        # Keep that addition before the ordered reduction, including its rounding.
        result = routed * cfg.moe_router_topk_scaling_factor
        if rank == 0:
            shared = local_shared(
                x, layer.shared_experts.linear_fc1.weight,
                layer.shared_experts.linear_fc2.weight, cfg.activation_func_clamp_value,
            )
            result = result + shared
        from sglang.srt.zero_mismatch.ordered_moe import reduce_rank_order

        return reduce_rank_order(result, counts, group).reshape_as(hidden_states)

    gate,up = layer.shared_experts.linear_fc1.weight.chunk(2,dim=0)
    in_weight = torch.cat((gate.chunk(16,dim=0)[rank],up.chunk(16,dim=0)[rank])).contiguous()
    out_weight = layer.shared_experts.linear_fc2.weight.chunk(16,dim=1)[rank].contiguous()
    shared = local_shared(x,in_weight,out_weight,cfg.activation_func_clamp_value)
    result = local_contribution(routed,shared,cfg.moe_router_topk_scaling_factor)
    from sglang.srt.zero_mismatch.ordered_moe import reduce_rank_order
    result = reduce_rank_order(result, counts, group)
    return result.reshape_as(hidden_states)

