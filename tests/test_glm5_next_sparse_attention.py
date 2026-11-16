"""Ascend950 sparse-attention forward/backward parity; NPU tests are opt-in."""

import importlib.util
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sparse_attention():
    if os.environ.get("GLM53_TEST_NPU") != "1":
        pytest.skip("Set GLM53_TEST_NPU=1 on an idle Ascend950 NPU")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an available NPU")
    if not torch.npu.get_device_name().startswith("Ascend950"):
        pytest.skip("requires Ascend950")
    path = Path(__file__).resolve().parents[1] / "slime_plugins/models/glm5_next/ascendc_sparse_attention.py"
    spec = importlib.util.spec_from_file_location("_glm53_sparse_attention_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sparse_attention


@pytest.mark.integration
@pytest.mark.parametrize("width,kv_tokens,query_tokens", [(17, 73, 5), (2048, 4096, 13), (2112, 4096, 65), (2112, 73, 5), (3072, 4096, 13), (8192, 2050, 5)])
def test_ascend950_sparse_attention_forward_backward(sparse_attention, width, kv_tokens, query_tokens):
    import torch

    generator = torch.Generator().manual_seed(5302112)
    heads, channels = 64, 512
    scale = 256**-0.5
    q_cpu = torch.randn(query_tokens, heads, channels, generator=generator).to(torch.bfloat16)
    kv_cpu = torch.randn(kv_tokens, 1, channels, generator=generator).to(torch.bfloat16)
    dout_cpu = (torch.randn(query_tokens, heads, channels, generator=generator) * 0.01).to(torch.bfloat16)
    indices = torch.full((query_tokens, 1, width), -1, dtype=torch.int32)
    for row in range(1, query_tokens):
        count = min(width, 2112, kv_tokens + 7) - row % 7
        # High, nonmonotonic indices and repeated slots exercise sparse semantics.
        selected = torch.randint(kv_tokens, (count,), generator=generator, dtype=torch.int32)
        selected[:3] = kv_tokens - 1
        indices[row, 0, :count] = selected
    reference_kv = kv_cpu.float().requires_grad_()
    expected_out = torch.zeros_like(q_cpu, dtype=torch.float32)
    expected_dq = torch.zeros_like(expected_out)
    expected_dkv = torch.zeros_like(reference_kv)
    for row in range(1, query_tokens):
        query = q_cpu[row].float().requires_grad_()
        valid = indices[row, 0]
        shared = reference_kv[valid[valid >= 0].long(), 0]
        output = ((query @ shared.T) * scale).softmax(-1) @ shared
        dq, dkv = torch.autograd.grad(output, (query, reference_kv), dout_cpu[row].float())
        expected_out[row] = output.detach()
        expected_dq[row] = dq
        expected_dkv.add_(dkv)

    q = q_cpu.to("npu").requires_grad_()
    kv = kv_cpu.to("npu").requires_grad_()
    output = sparse_attention(q, kv, indices.to("npu"), scale)
    dq, dkv = torch.autograd.grad(output, (q, kv), dout_cpu.to("npu"))
    actual = tuple(t.detach().float().cpu() for t in (output, dq, dkv))
    expected = (expected_out, expected_dq, expected_dkv)
    errors = {}
    for name, result, reference in zip(("out", "dq", "dkv"), actual, expected, strict=True):
        assert result.shape == reference.shape, name
        assert torch.isfinite(result).all(), name
        error = (result - reference).double()
        relative_rms = error.norm() / reference.double().norm().clamp_min(1e-12)
        errors[name] = relative_rms.item()
        assert relative_rms < 0.01, (name, relative_rms.item())
        if name != "dkv":
            assert torch.count_nonzero(result[0]) == 0, name
            row_error = error[1:].flatten(1).norm(dim=1)
            row_norm = reference[1:].double().flatten(1).norm(dim=1).clamp_min(1e-12)
            assert (row_error / row_norm).max() < 0.01, name
    print(f"SFA_PARITY width={width} kv_tokens={kv_tokens} query_tokens={query_tokens} errors={errors}")


@pytest.mark.integration
def test_ascend950_rejects_selection_width_above_limit(sparse_attention):
    import torch

    q = torch.zeros(1, 64, 512, device="npu", dtype=torch.bfloat16)
    kv = torch.zeros(1, 1, 512, device="npu", dtype=torch.bfloat16)
    indices = torch.zeros(1, 1, 8193, device="npu", dtype=torch.int32)
    with pytest.raises(ValueError, match="at most 8192 selections"):
        sparse_attention(q, kv, indices, 256**-0.5)
