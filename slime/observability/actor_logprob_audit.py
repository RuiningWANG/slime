"""Per-rollout FP32 bit comparison at the trainable policy loss, for CP1/TP1 runs."""

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
import torch.distributed as dist


_active = ContextVar("actor_logprob_audit", default=None)


@contextmanager
def audit_rollout(rollout_id):
    rank = dist.get_rank()
    stats = dict(rollout=int(rollout_id), rank=rank, microbatches=0, samples=0, valid_tokens=0, mismatch_tokens=0, absolute_sum=0.0, max_abs_diff=0.0)
    handle = _active.set(stats)
    try:
        yield
        if stats["microbatches"]:
            path = Path(os.environ["GLM53_PHASE_DIR"]) / f"strict-actor-audit-rank-{rank}.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                print(json.dumps(stats, allow_nan=False), file=stream)
    finally:
        _active.reset(handle)


def compare_policy_scores(batch, scores):
    stats = _active.get()
    if stats is None or not torch.is_grad_enabled() or not any(score.requires_grad for score in scores):
        return
    for policy, rollout, mask in zip(scores, batch["rollout_log_probs"], batch["loss_masks"], strict=True):
        if policy.dtype != torch.float32 or rollout.dtype != torch.float32:
            raise TypeError(f"Expected FP32 logprobs, got {policy.dtype} and {rollout.dtype}")
        mask = mask.detach().cpu().bool()
        policy = policy.detach().cpu()[mask].contiguous()
        rollout = rollout.detach().cpu()[mask].contiguous()
        stats["samples"] += 1
        stats["valid_tokens"] += policy.numel()
        stats["mismatch_tokens"] += int((policy.view(torch.int32) != rollout.view(torch.int32)).sum())
        if policy.numel():
            delta = (policy.double() - rollout.double()).abs()
            stats["absolute_sum"] += float(delta.sum())
            stats["max_abs_diff"] = max(stats["max_abs_diff"], float(delta.max()))
    stats["microbatches"] += 1
