"""Pre-plan GLM mHC pipeline shapes without synchronizing each P2P exchange."""

import os
import weakref
from collections import deque
from datetime import timedelta

import torch
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    _build_default_pg_collection,
    get_schedule_table,
)
from megatron.core.utils import get_model_config

from .data import DataIterator

_metadata_groups = weakref.WeakKeyDictionary()


def _metadata_group(pp_group):
    """Keep planning off the accelerator and out of the HCCL P2P sequence."""
    if pp_group not in _metadata_groups:
        # All training ranks create groups in the same order. Local group naming
        # hashes the number of member groups, which can differ at PP endpoints
        # because only those ranks belong to embedding process groups.
        dist = torch.distributed
        bootstrap = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
        ranks = dist.get_process_group_ranks(pp_group)
        memberships = [None] * dist.get_world_size()
        dist.all_gather_object(memberships, ranks, group=bootstrap)
        for members in sorted({tuple(group) for group in memberships}):
            group = dist.new_group(ranks=list(members), backend="gloo", timeout=timedelta(seconds=120))
            if dist.get_rank() in members:
                _metadata_groups[pp_group] = group
        dist.destroy_process_group(bootstrap)
    return _metadata_groups[pp_group]


class PlannedP2PCommunicator(P2PCommunicator):
    """Use independent FIFO plans for the four interleaved P2P directions."""

    def __init__(self, pp_group, config, lengths, num_chunks):
        super().__init__(pp_group, config)
        width = config.glm53_pipeline_hidden_size
        rank, size = pp_group.rank(), pp_group.size()
        table = get_schedule_table(len(lengths), num_chunks, config.microbatch_group_size_per_vp_stage)
        forward = [(micro, chunk) for micro, chunk in table]
        backward = [(micro, num_chunks - 1 - chunk) for micro, chunk in table]

        def shapes(order, exclude):
            return deque((lengths[micro], 1, width) for micro, chunk in order if not exclude(chunk))

        first = lambda chunk: rank == 0 and chunk == 0
        last = lambda chunk: rank == size - 1 and chunk == num_chunks - 1
        self.plans = {
            "send_next": shapes(forward, last),
            "recv_prev": shapes(forward, first),
            "send_prev": shapes(backward, first),
            "recv_next": shapes(backward, last),
        }

    def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        for direction, tensor in (
            ("send_next", tensor_send_next),
            ("send_prev", tensor_send_prev),
        ):
            if tensor is not None:
                expected = self.plans[direction].popleft()
                if tuple(tensor.shape) != expected or tensor.dtype != self.config.pipeline_dtype:
                    raise RuntimeError(
                        f"PP {direction}: expected {expected}/{self.config.pipeline_dtype}, "
                        f"got {tuple(tensor.shape)}/{tensor.dtype}"
                    )
        prev = self.plans["recv_prev"].popleft() if recv_prev else (0, 0, 0)
        nxt = self.plans["recv_next"].popleft() if recv_next else (0, 0, 0)
        return prev, nxt

    def assert_complete(self):
        remaining = {name: len(queue) for name, queue in self.plans.items() if queue}
        if remaining:
            raise RuntimeError(f"Unconsumed PP shape plan: {remaining}")


def _planned_lengths(data_iterator, num_microbatches, pad_multiplier):
    """Mirror CP=TP=1 get_batch padding, reading only CPU tensor metadata."""
    if not data_iterator or not all(isinstance(it, DataIterator) for it in data_iterator):
        return None
    plans = []
    for it in data_iterator:
        indices = it.micro_batch_indices[it.offset : it.offset + num_microbatches]
        tokens = it.rollout_data.get("tokens")
        if len(indices) != num_microbatches or tokens is None:
            return None
        lengths = []
        for batch in indices:
            if not batch or any(tokens[i].ndim != 1 for i in batch):
                return None
            length = sum(tokens[i].numel() for i in batch)
            if length <= 0:
                return None
            lengths.append((length + pad_multiplier - 1) // pad_multiplier * pad_multiplier)
        plans.append(lengths)
    return plans[0] if all(plan == plans[0] for plan in plans) else None


def prepare_pipeline_shapes(args, model, data_iterator, num_microbatches):
    """Return schedule kwargs, or collectively fall back to dynamic exchange.

    Restricted to GLM mHC interleaved training with TP=CP=1. Other models,
    schedules and partitioning schemes retain the existing communicator.
    """
    config = get_model_config(model[0])
    if (
        not getattr(config, "glm53_pipeline_hidden_size", None)
        or (config.virtual_pipeline_model_parallel_size or 1) < 2
        or config.pipeline_model_parallel_size < 2
        or config.tensor_model_parallel_size != 1
        or config.context_parallel_size != 1
    ):
        return {}
    pg = _build_default_pg_collection()
    lengths = None
    if os.environ.get("GLM53_PP_PRECOMPUTE_SHAPES", "1") == "1":
        lengths = _planned_lengths(data_iterator, num_microbatches, args.data_pad_size_multiplier)
    # One small collective per step: every PP rank must choose the same path.
    local = torch.tensor(
        lengths or [-1] * num_microbatches,
        dtype=torch.int64,
        device="cpu",
    )
    gathered = [torch.empty_like(local) for _ in range(pg.pp.size())]
    torch.distributed.all_gather(gathered, local, group=_metadata_group(pg.pp))
    plans = [plan.tolist() for plan in gathered]
    if any(n <= 0 for plan in plans for n in plan) or any(plan != plans[0] for plan in plans):
        return {}
    communicator = PlannedP2PCommunicator(pg.pp, config, plans[0], len(model))
    return {"p2p_communicator": communicator, "pg_collection": pg}
