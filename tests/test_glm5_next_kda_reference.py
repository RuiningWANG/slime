"""Compare packed KDA and per-document KDA with an independent FP32 recurrence."""

import os

import pytest
import torch
import torch.nn.functional as F


def recurrent_kda(q, k, v, gate, beta, offsets):
    result = []
    for first, last in zip(offsets, offsets[1:]):
        state = q.new_zeros(q.size(2), k.size(-1), v.size(-1))
        for t in range(first, last):
            key = k[0, t]
            state = state * gate[0, t].exp().unsqueeze(-1)
            residual = v[0, t] - torch.einsum("hk,hkv->hv", key, state)
            state = state + key.unsqueeze(-1) * (beta[0, t, :, None] * residual).unsqueeze(-2)
            result.append(torch.einsum("hk,hkv->hv", q[0, t], state) * q.size(-1) ** -0.5)
    return torch.stack(result).unsqueeze(0)


@pytest.mark.skipif(os.getenv("GLM53_TEST_NPU") != "1", reason="requires an idle Ascend NPU")
@pytest.mark.parametrize("mode", ["packed", "loop"])
def test_kda_output_and_gradients(monkeypatch, mode):
    pytest.importorskip("torch_npu")
    from slime_plugins.models.glm5_next.npu_backend import chunk_kda

    monkeypatch.setenv("GLM53_KDA_SEQUENCE_MODE", mode)
    rng = torch.Generator().manual_seed(142)
    shape = (1, 102, 8, 128)
    q, k = [F.normalize(torch.randn(shape, generator=rng), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(shape, generator=rng).bfloat16()
    gate = -torch.rand(shape, generator=rng)
    beta = torch.rand(shape[:-1], generator=rng)
    offsets = [0, 65, 102]
    inputs = [q, k, v, gate, beta]
    ref = [x.float().requires_grad_() for x in inputs]
    native = [x.npu().requires_grad_() for x in inputs]
    expected = recurrent_kda(*ref, offsets)
    actual, _ = chunk_kda(*native[:3], g=native[3], beta=native[4], cu_seqlens=offsets)
    grad = torch.randn(shape, generator=rng).bfloat16()
    ref_grad = torch.autograd.grad(expected, ref, grad.float())
    actual_grad = torch.autograd.grad(actual, native, grad.npu())
    for name, calculated, reference in zip(
        ("out", "dq", "dk", "dv", "dg", "dbeta"), (actual, *actual_grad), (expected, *ref_grad), strict=True
    ):
        value = calculated.detach().float().cpu()
        relative_error = (value - reference.detach()).norm() / reference.detach().norm().clamp_min(1e-8)
        assert relative_error < 0.04, (name, relative_error.item())


def test_reference_resets_state_between_documents():
    q = torch.ones(1, 3, 1, 1)
    v = torch.tensor([2.0, 4.0, 8.0], requires_grad=True)
    y = recurrent_kda(q, q, v.view_as(q), torch.zeros_like(q), q[..., 0] * 0.5, [0, 2, 3])
    torch.testing.assert_close(y.flatten(), torch.tensor([1.0, 2.5, 4.0]))
    torch.testing.assert_close(torch.autograd.grad(y.sum(), v)[0], torch.tensor([0.75, 0.5, 0.5]))
