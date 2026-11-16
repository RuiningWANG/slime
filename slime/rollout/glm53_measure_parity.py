import json
import os
from pathlib import Path

import torch


def compare(args, rollout_id, data):
    compare_logps(rollout_id, data, "log_probs", "native")
    if data.get("sglang_norm_log_probs"):
        compare_logps(rollout_id, data, "sglang_norm_log_probs", "sglang-norm")


def compare_logps(rollout_id, data, key, label):
    training = data.get(key)
    rollout = data.get("rollout_log_probs")
    if not training or not rollout:
        return
    if len(training) != len(rollout) or len(training) != len(data["loss_masks"]):
        raise ValueError("Train/rollout logp sample counts differ")
    rank = torch.distributed.get_rank()
    root = Path(os.environ["GLM53_PARITY_REPORT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    if label == "native":
        snapshot = {
            "partition": data["partition"],
            "tokens": [torch.as_tensor(t).detach().cpu() for t in data["tokens"]],
            "response_lengths": data["response_lengths"],
            "loss_masks": [torch.as_tensor(t).detach().cpu() for t in data["loss_masks"]],
            "native_logp": [torch.as_tensor(t).detach().cpu() for t in training],
            "rollout_logp": [torch.as_tensor(t).detach().cpu() for t in rollout],
        }
        torch.save(snapshot, root / f"frozen-token-logp-rollout-{rollout_id}-rank-{rank}.pt")
    report = []
    for index, (left, right, mask) in enumerate(zip(training, rollout, data["loss_masks"], strict=True)):
        left = torch.as_tensor(left).detach().float().cpu().reshape(-1)
        right = torch.as_tensor(right).detach().float().cpu().reshape(-1)
        mask = torch.as_tensor(mask).detach().cpu().bool().reshape(-1)
        if left.shape != right.shape or left.shape != mask.shape:
            raise ValueError(f"Sample {index} logp shapes differ: {left.shape}, {right.shape}, {mask.shape}")
        left, right = left[mask], right[mask]
        finite = torch.isfinite(left) & torch.isfinite(right)
        diff = (left[finite] - right[finite]).abs()
        different = (left.view(torch.int32) != right.view(torch.int32)) | ~finite
        report.append(
            {
                "sample": int(data["partition"][index]),
                "tokens": left.numel(),
                "mismatches": int(different.sum()),
                "sum_abs_diff": float(diff.double().sum()),
                "max_abs_diff": float(diff.max()) if diff.numel() else None,
            }
        )
    path = root / f"parity-{label}-rollout-{rollout_id}-rank-{rank}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"GLM53_PARITY variant={label} rank={rank} tokens={sum(x['tokens'] for x in report)} mismatches={sum(x['mismatches'] for x in report)} report={path}", flush=True)
