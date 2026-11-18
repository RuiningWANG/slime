"""GLM5-next training forward matching the Ascend pooled indexer."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def serving_indexer(layer, x, q_latent, boundaries):
    import torch_npu
    from sglang.srt.hardware_backend.npu.glm53_parity import projection_shard
    from sglang.srt.layers.attention.dsa.kpool_index_npu import kpool_assemble_softmax_write_cache_npu

    heads, dim, pool_size, topk = layer.heads, layer.dim, layer.pool_size, layer.topk
    query = projection_shard(q_latent, layer.wq_b.weight).view(-1, heads, dim)
    key = projection_shard(x, layer.wk.weight)
    key = F.layer_norm(
        key.float(), (dim,), layer.k_norm.weight.float(), layer.k_norm.bias.float(), layer.k_norm.eps,
    ).to(x.dtype)
    gate = projection_shard(x, layer.index_kpool_compress_gate)
    weights = projection_shard(x.float(), layer.weights_proj.weight.float())
    weights = weights * (heads ** -0.5 * dim ** -0.5)
    width = ((topk + pool_size - 1 + 63) // 64) * 64
    output = torch.full((x.shape[0], 1, width), -1, device=x.device, dtype=torch.int32)
    offsets = torch.arange(pool_size, device=x.device, dtype=torch.int32)
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        length = end - begin
        if not length:
            continue
        positions = torch.arange(1, length + 1, device=x.device, dtype=torch.int32)
        pool_lengths = positions // pool_size
        pools = length // pool_size
        if pools:
            page_size = 64
            pages = (pools + page_size - 1) // page_size
            cache = x.new_zeros((pages, page_size, 1, dim))
            ids = torch.arange(pools, device=x.device, dtype=torch.int64)
            zeros = torch.zeros_like(ids)
            tail_key = x.new_zeros((1, pool_size, dim))
            tail_gate = torch.zeros_like(tail_key)
            kpool_assemble_softmax_write_cache_npu(
                cache, key[begin:end].contiguous(), gate[begin:end].contiguous(),
                tail_key, tail_gate, zeros, zeros, ids * pool_size, zeros,
                layer.index_kpool_compress_ape.float().contiguous(), ids,
            )
            table = torch.arange(pages, device=x.device, dtype=torch.int32).expand(length, -1).contiguous()
            indices = torch_npu.npu_lightning_indexer(
                query=query[begin:end].contiguous(), key=cache,
                weights=weights[begin:end].to(x.dtype).contiguous(),
                actual_seq_lengths_query=positions,
                actual_seq_lengths_key=pool_lengths.contiguous(), block_table=table,
                layout_query="TND", layout_key="PA_BSND",
                sparse_count=topk // pool_size, sparse_mode=3,
            )[0].squeeze(1)[:length]
            expanded = indices.unsqueeze(-1) * pool_size + offsets
            valid = (indices >= 0) & (indices < pool_lengths[:, None])
            expanded.masked_fill_(~valid.unsqueeze(-1), -1)
            output[begin:end, 0, :topk] = torch.where(
                expanded.flatten(1) >= 0, expanded.flatten(1) + begin, -1,
            ).int()
        tail = pool_lengths[:, None] * pool_size + offsets[:-1]
        output[begin:end, 0, topk:topk + pool_size - 1] = torch.where(
            tail < positions[:, None], tail + begin, -1,
        )
        # Ascend A5 serving uses all causal tokens in their natural order
        # before the configured sparse budget is reached.
        natural = torch.arange(width, device=x.device, dtype=torch.int32).expand(length, -1)
        natural = torch.where(natural < positions[:, None], natural + begin, -1)
        output[begin:end, 0] = torch.where(
            (positions < topk)[:, None], natural, output[begin:end, 0],
        )
    return output
