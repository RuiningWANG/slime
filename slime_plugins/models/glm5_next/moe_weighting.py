"""Apply GLM routed-expert weights at combine, matching NPU inference."""

import copy
import math
from types import MethodType

import torch


def weighted_sum_fp32(expert_output, token_indices, probabilities, restore_shape):
    # Inference stores probabilities in the model dtype, but accumulates the
    # weighted expert outputs in FP32 before rounding the combined token once.
    weights = probabilities.to(expert_output.dtype).float()
    weighted = expert_output.float() * weights.unsqueeze(-1)
    result = torch.zeros(restore_shape, device=expert_output.device, dtype=torch.float32)
    return result.index_add(0, token_indices.long(), weighted)


def _experts_without_weighting(self, x, counts, probs, output_buffer=None, grad_input_buffer=None):
    return self._glm53_unweighted_original_forward(
        x, counts, torch.ones_like(probs), output_buffer, grad_input_buffer
    )


def _alltoall_combine_postprocess(self, expert_output):
    probabilities = self.probs.T.contiguous().masked_select(self.routing_map.T.contiguous())
    # NPU fused permute returns an inverse map over token-expert assignments,
    # not source token IDs. Rebuild IDs in the expert-major output order.
    token_indices = torch.arange(
        self.routing_map.shape[0], device=expert_output.device,
    ).expand(self.routing_map.shape[1], -1).masked_select(self.routing_map.T.contiguous())
    output = weighted_sum_fp32(
        expert_output, token_indices,
        probabilities, self.hidden_shape_before_permute,
    ).to(expert_output.dtype)
    output = output * self._glm53_routed_scaling_factor
    return output.view(self.hidden_shape)


def _allgather_combine_preprocess(self, expert_output):
    self._glm53_combined_dtype = expert_output.dtype
    return weighted_sum_fp32(
        expert_output, self.reversed_local_input_permutation_mapping,
        self.local_probs, self.hidden_shape_before_permute,
    )


def _allgather_combine_postprocess(self, output):
    # The existing ReduceScatter receives FP32 partial sums from preprocess.
    return (output.to(self._glm53_combined_dtype) * self._glm53_routed_scaling_factor).view(self.hidden_shape)


def install_inference_aligned_weighting(module, scaling_factor):
    """Install before optional chunking, preserving modules and checkpoint keys."""
    config = module.config
    if hasattr(module.experts, '_glm53_unweighted_original_forward'):
        raise ValueError('GLM inference-aligned MoE weighting is already installed')
    if config.moe_token_dispatcher_type not in ('alltoall', 'allgather'):
        return
    if config.tensor_model_parallel_size != 1 or config.expert_tensor_parallel_size != 1:
        return
    for name in ('fp8', 'fp4', 'moe_apply_probs_on_input', 'moe_shared_expert_overlap',
                 'moe_expert_capacity_factor', 'moe_pad_expert_input_to_capacity',
                 'moe_router_padding_for_quantization'):
        if getattr(config, name, False):
            return
    scale = float(scaling_factor)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Routed scaling factor must be finite and positive')
    # Router config is shared by other modules. Copy it before moving the
    # scaling factor out of route probabilities and into the combined output.
    module.router.config = copy.copy(module.router.config)
    module.router.config.moe_router_topk_scaling_factor = 1.0
    module.experts._glm53_unweighted_original_forward = module.experts.forward
    module.experts.forward = MethodType(_experts_without_weighting, module.experts)
    dispatcher = module.token_dispatcher
    dispatcher._glm53_routed_scaling_factor = scale
    if config.moe_token_dispatcher_type == 'alltoall':
        dispatcher.combine_postprocess = MethodType(_alltoall_combine_postprocess, dispatcher)
    else:
        dispatcher.combine_preprocess = MethodType(_allgather_combine_preprocess, dispatcher)
        dispatcher.combine_postprocess = MethodType(_allgather_combine_postprocess, dispatcher)
