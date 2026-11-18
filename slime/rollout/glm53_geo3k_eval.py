"""Matched Geo3K quality rounds using the framework's original evaluator."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import urllib.request


def generate_three_rounds(args, rollout_id, data_source, evaluation=False):
    if not evaluation:
        raise ValueError("This entry point is for evaluation-only runs")
    from slime.rollout.sglang_rollout import generate_rollout
    from slime.rollout.base_types import RolloutFnEvalOutput

    rounds = int(os.getenv("GLM53_EVAL_ROUNDS", "3"))
    if rounds < 1:
        raise ValueError("GLM53_EVAL_ROUNDS must be at least one")
    phase = Path(os.environ["GLM53_PHASE_DIR"])
    phase.mkdir(parents=True, exist_ok=True)
    source = Path(args.prompt_data)
    fingerprint = Path(os.environ["GLM53_CHECKPOINT_FINGERPRINT"])
    protocol = {
        "rounds": rounds, "seed": args.seed, "checkpoint": args.hf_checkpoint,
        "checkpoint_digest": json.loads(fingerprint.read_text())["content_digest"],
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": sum(bool(line.strip()) for line in source.read_text().splitlines()),
        "samples_per_prompt": args.n_samples_per_eval_prompt,
        "temperature": args.eval_temperature, "top_p": args.eval_top_p,
        "max_prompt_len": args.eval_max_prompt_len, "max_response_len": args.eval_max_response_len,
        "max_context_len": args.eval_max_context_len,
        "request_sampling_seed_override": None,
        "prefix_cache_round_start": "flush, original cache remains enabled",
        "datasets": [
            {key: getattr(dataset, key, None)
             for key in ("name", "path", "input_key", "label_key", "temperature",
                         "top_p", "top_k", "max_response_len", "n_samples_per_eval_prompt")}
            for dataset in args.eval_datasets
        ],
    }
    (phase / "evaluation-protocol.json").write_text(json.dumps(protocol, indent=2))
    pooled = {}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cache_endpoint = os.environ["GLM53_EVAL_CACHE_ENDPOINT"]
    for round_index in range(rounds):
        request = urllib.request.Request(f"{cache_endpoint}/flush_cache?timeout=30", data=b"")
        with opener.open(request, timeout=45) as response:
            response.read()
        output = generate_rollout(args, rollout_id, data_source, evaluation=True)
        counts = {}
        for name, block in output.data.items():
            entry = pooled.setdefault(name, {"rewards": [], "truncated": [], "samples": []})
            for sample in block["samples"]:
                sample.metadata = dict(sample.metadata or {}) | {"strict_eval_round": round_index}
            for key in ("rewards", "truncated", "samples"):
                entry[key].extend(block[key])
            counts[name] = len(block["rewards"])
        print("GLM53_QUALITY_ROUND " + json.dumps({"round": round_index, "requests": counts}), flush=True)
    return RolloutFnEvalOutput(data=pooled)


def save(rollout_id, args, data, extra_metrics=None):
    phase = Path(os.environ["GLM53_PHASE_DIR"])
    rows = []
    summaries = {}
    for name, block in data.items():
        rewards, samples = block["rewards"], block["samples"]
        per_round = defaultdict(list)
        for reward, sample in zip(rewards, samples, strict=True):
            round_index = int(sample.metadata["strict_eval_round"])
            index = int(sample.index)
            tokens = list(sample.tokens)
            value = float(reward)
            per_round[round_index].append(value)
            rows.append({
                "dataset": name, "round": round_index, "sample": index,
                "question": index // args.n_samples_per_eval_prompt,
                "replicate": index % args.n_samples_per_eval_prompt,
                "prompt_tokens": len(tokens) - sample.response_length,
                "response_tokens": sample.response_length,
                "reward": value, "label": sample.label, "response": sample.response,
                "prompt_text": str(sample.prompt),
                "tokens": tokens, "status": str(sample.status),
            })
        summaries[name] = {
            "responses": len(rewards), "reward_sum": sum(float(v) for v in rewards),
            "mean_reward": sum(float(v) for v in rewards) / len(rewards),
            "rounds": {str(i): {"responses": len(values), "mean_reward": sum(values)/len(values)}
                       for i, values in sorted(per_round.items())},
        }
    path = phase / "quality-receipts.jsonl"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    temporary.replace(path)
    (phase / "quality-summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    print("GLM53_QUALITY_RECEIPTS " + json.dumps(summaries), flush=True)
    return False
