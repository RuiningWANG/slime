"""Match serving's paged SFA forward while retaining the native training VJP."""

import torch
import torch.nn.functional as F
import torch_npu

from .npu_backend import ServingValue, _boundaries
from .ascendc_smla import compact_sparse_indices


@torch.no_grad()
def serving_sparse_forward(q, kv, indices, scaling, boundaries, index_width):
    pieces = []
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        length = end - begin
        if length == 0:
            continue
        local_indices = indices[begin:end, ..., :index_width]
        local_indices = torch.where(local_indices >= 0, local_indices - begin, -1)
        selected = compact_sparse_indices(local_indices, length)
        page_size = 64
        capacity = max(length, index_width + 1)
        pages = (capacity + page_size - 1) // page_size
        cache = F.pad(kv[begin:end], (0, 0, 0, 0, 0, pages * page_size - length))
        cache = cache.reshape(pages, page_size, 1, kv.shape[-1]).contiguous()
        query = q[begin:end].contiguous()
        q_rope = query.new_zeros((*query.shape[:-1], 64))
        k_rope = cache.new_zeros((*cache.shape[:-1], 64))
        lengths = torch.tensor([length], device=q.device, dtype=torch.int32)
        table = torch.arange(pages, device=q.device, dtype=torch.int32).view(1, -1)
        output, _, _ = torch_npu.npu_sparse_flash_attention(
            query=query, key=cache, value=cache, query_rope=q_rope, key_rope=k_rope,
            sparse_indices=selected, scale_value=float(scaling),
            actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
            block_table=table, sparse_block_size=1,
            layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2,
            return_softmax_lse=False,
        )
        pieces.append(output)
    return torch.cat(pieces, dim=0) if pieces else q.new_empty(q.shape)


def training_sparse_attention(q, kv, indices, scaling, packed, cp_group, index_width, *, native):
    if (
        q.device.type != "npu" or q.dtype != torch.bfloat16
        or (cp_group is not None and cp_group.size() != 1)
    ):
        return native(q, kv, indices, scaling)
    cpu_bounds = getattr(packed, "cu_seqlens_q_cpu", None)
    boundaries = _boundaries(
        cpu_bounds if cpu_bounds is not None else packed.cu_seqlens_q, q.shape[0],
    )
    aligned = serving_sparse_forward(q, kv, indices, scaling, boundaries, index_width)
    if torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad):
        original, auxiliary = native(q, kv, indices, scaling)
        return ServingValue.apply(original, aligned), auxiliary
    return aligned, None
