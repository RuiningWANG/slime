"""NPU numerical regression tests for the GLM5-next fused causal convolution."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
from slime_plugins.models.glm5_next import ascendc_conv1d as _module


def reference(x, weight, activation, boundaries):
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        if start == end:
            continue
        y = F.conv1d(
            F.pad(x[:, start:end].transpose(1, 2), (weight.shape[-1] - 1, 0)), weight[:, None], groups=weight.shape[0]
        ).transpose(1, 2)
        outputs.append(F.silu(y) if activation else y)
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", ["dense", "packed", "short", "empty_segment", "full_width"])
def test_forward_backward(dtype, case):
    torch.manual_seed(53)
    batch, length, channels = (2, 129, 128) if case == "dense" else (1, 257, 128)
    boundaries = [0, length]
    if case == "packed":
        boundaries = [0, 1, 17, 128, length]
    if case == "short":
        length, boundaries = 2, [0, 1, 2]
    if case == "empty_segment":
        boundaries = [0, 0, 37, 37, length]
    if case == "full_width":
        length, channels, boundaries = 1024, 8192, [0, 333, 1024]
    x = (torch.randn(batch, length, channels, device="npu", dtype=dtype) * 0.2).requires_grad_()
    w = (torch.randn(channels, 4, device="npu", dtype=dtype) * 0.2).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    dy = torch.randn_like(x) * 0.2
    actual = _module.causal_conv1d_ascendc(
        x, w, "silu", None if case == "dense" else torch.tensor(boundaries, dtype=torch.int32)
    )
    expect = reference(xr, wr, "silu", boundaries)
    actual.backward(dy)
    expect.backward(dy)
    torch.npu.synchronize()
    for name, a, b in [("output", actual, expect), ("dx", x.grad, xr.grad), ("dw", w.grad, wr.grad)]:
        af, bf = a.float(), b.float()
        relative_rms = ((af - bf).square().mean().sqrt() / bf.square().mean().sqrt().clamp_min(1e-8)).item()
        assert torch.isfinite(a).all()
        assert relative_rms < (0.025 if dtype == torch.bfloat16 else 0.005)
