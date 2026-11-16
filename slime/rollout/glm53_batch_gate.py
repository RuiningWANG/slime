import json
import math
import os
import struct
import urllib.request
from pathlib import Path


def generate(url, prompts):
    request = urllib.request.Request(
        url + "/generate",
        data=json.dumps(
            {
                "input_ids": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
                "return_logprob": True,
                "logprob_start_len": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=1200) as response:
        return json.load(response)


def compare(reference, candidate):
    result = {}
    for field in ("input_token_logprobs", "output_token_logprobs"):
        left = reference["meta_info"][field]
        right = candidate["meta_info"][field]
        values = [(a[0], b[0]) for a, b in zip(left, right) if a[0] is not None and b[0] is not None]
        differences = [abs(a - b) for a, b in values if math.isfinite(a) and math.isfinite(b)]
        mismatch = sum(not (math.isfinite(a) and math.isfinite(b)) or struct.pack("!d", a) != struct.pack("!d", b) for a, b in values)
        result[field] = {
            "same_token_ids": [x[1] for x in left] == [x[1] for x in right],
            "same_logp_positions": [x[0] is None for x in left] == [x[0] is None for x in right],
            "compared": len(values),
            "bit_mismatches": mismatch,
            "max_abs_diff": max(differences, default=None),
        }
    return result


def check(url):
    report = {}
    for length in (32, 97):
        target = [100 + i % 53 for i in range(length)]
        short = [122 + i % 37 for i in range(15)]
        long = [211 + i % 53 for i in range(128)]
        medium = [301 + i % 29 for i in range(63)]
        reference = generate(url, target)
        comparisons = {
            "repeat": generate(url, target),
            "batch2_position0": generate(url, [target, long])[0],
            "batch2_position1": generate(url, [short, target])[1],
            "batch4_position2": generate(url, [long, medium, target, short])[2],
        }
        report[str(length)] = {name: compare(reference, candidate) for name, candidate in comparisons.items()}
    passed = all(field["same_token_ids"] and field["same_logp_positions"] and field["compared"] > 0 and field["bit_mismatches"] == 0 for cases in report.values() for fields in cases.values() for field in fields.values())
    result_path = Path(os.environ["GLM53_BATCH_REPORT"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"passed": passed, "comparisons": report}, indent=2) + "\n")
    print(f"BATCH_INVARIANCE passed={passed} report={result_path}", flush=True)
    if not passed:
        raise RuntimeError("Batch-invariance gate failed. See the per-token comparison report.")
