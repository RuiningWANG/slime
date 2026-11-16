import json
import math
import os
import struct
import urllib.request
from pathlib import Path
from itertools import groupby

from slime.rollout.sglang_rollout import generate_rollout as native_generate
from slime.rollout.sglang_rollout import get_model_url
from slime.utils.processing_utils import encode_image_for_rollout_engine


def generate(args, rollout_id, data_source, evaluation=False):
    result = native_generate(args, rollout_id, data_source, evaluation)
    if evaluation:
        return result
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = get_model_url(args, "regular")
    report = []
    for sample in (sample for group in result.samples for sample in group):
        n = sample.response_length
        if not n:
            continue
        images = (sample.multimodal_inputs or {}).get("images") or []
        body = {
            "input_ids": [token for key, run in groupby(sample.tokens) for token in ([key] if key == 154854 else run)],
            "image_data": [encode_image_for_rollout_engine(image) for image in images],
            "sampling_params": {"temperature": 0, "max_new_tokens": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        row = {"sample": sample.index, "images": len(images)}
        try:
            with opener.open(request, timeout=180) as response:
                values = json.load(response)["meta_info"]["input_token_logprobs"][-n:]
            if [x[1] for x in values] != sample.tokens[-n:]:
                raise ValueError("prefill response token IDs differ")
            mask = sample.loss_mask if sample.loss_mask is not None else [1] * n
            pairs = [(a[0], b) for a, b, valid in zip(values, sample.rollout_log_probs, mask, strict=True) if valid]
            if not all(math.isfinite(a) and math.isfinite(b) for a, b in pairs):
                raise ValueError("nonfinite logp")
            diffs = [abs(a - b) for a, b in pairs]
            row.update(tokens=len(pairs), mismatches=sum(struct.pack("!f", a) != struct.pack("!f", b) for a, b in pairs), sum_abs_diff=sum(diffs), max_abs_diff=max(diffs, default=0.0))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["error"] = str(error)
        report.append(row)
    report_directory = Path(os.environ["GLM53_PARITY_REPORT_DIR"])
    report_directory.mkdir(parents=True, exist_ok=True)
    output = report_directory / f"multimodal-prefill-rollout-{rollout_id}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")

    trace_dir = os.getenv("GLM53_TRACE_DIR")
    comparable = [row for row in report if row.get("tokens")]
    if trace_dir and comparable:
        worst = max(comparable, key=lambda row: row["max_abs_diff"])
        target = next(sample for group in result.samples for sample in group if sample.index == worst["sample"])
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_trace(opener, url, target, root)
    print(f"GLM53_MM_REPLAY tested={sum('tokens' in row for row in report)} errors={sum('error' in row for row in report)} report={output}", flush=True)
    return result


def run_trace(opener, url, sample, root):
    images = (sample.multimodal_inputs or {}).get("images") or []
    n = sample.response_length
    prompt = sample.tokens[:-n]
    flag = root / "trace-phase"

    def request(ids, max_new_tokens, phase):
        compressed = [token for key, run in groupby(ids) for token in ([key] if key == 154854 else run)]
        payload = {
            "input_ids": compressed,
            "image_data": [encode_image_for_rollout_engine(image) for image in images],
            "sampling_params": {"temperature": 0, "ignore_eos": True, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "logprob_start_len": 0,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        flag.write_text(phase)
        try:
            with opener.open(req, timeout=300) as response:
                return json.load(response)
        finally:
            flag.unlink(missing_ok=True)

    decode = request(prompt, 8, "decode")
    generated = [item[1] for item in decode["meta_info"]["output_token_logprobs"]]
    request(prompt + generated, 0, "prefill")
    (root / "trace-sequence.json").write_text(
        json.dumps(
            {
                "sample": sample.index,
                "prompt": prompt,
                "generated": generated,
            }
        )
    )
    print(f"GLM53_LAYER_TRACE sample={sample.index} generated={len(generated)}", flush=True)
