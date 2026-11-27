"""Forward/backward and clamp-boundary checks for the shared torch SwiGLU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
from slime_plugins.models.glm5_next import ops


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows,noncontiguous", [(0, False), (17, False), (512, False), (17, True)])
def test_swiglu_forward_backward(dtype, rows, noncontiguous):
    torch.manual_seed(53)
    width, limit = 2048, 10.0
    data = (torch.randn(rows, 2 * width, device="npu") * 12).to(dtype)
    if rows:
        edges = torch.tensor([-11.0, -10.0, -9.0, 0.0, 9.0, 10.0, 11.0], device="npu", dtype=dtype)
        data[0, :7] = edges
        data[0, width : width + 7] = edges
    if noncontiguous:
        data = data.t().contiguous().t()
    x = data.detach().requires_grad_()
    baseline = data.detach().clone().requires_grad_()
    actual = ops.ClampedSwiGLU(SimpleNamespace(activation_func_clamp_value=limit))(x)
    gate, up = baseline.float().chunk(2, dim=-1)
    expected = (F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)).to(dtype)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.npu.synchronize()
    for actual, reference in ((actual, expected), (x.grad, baseline.grad)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if rows:
        assert x.grad[0, 6].item() == 0  # gate above upper clamp
        assert x.grad[0, width].item() == 0  # up below lower clamp
        assert x.grad[0, width + 6].item() == 0  # up above upper clamp
