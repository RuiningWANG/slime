"""Regression coverage for SparseFlashAttention index compaction; NPU tests are opt-in."""

import os
import pytest
import torch

from slime_plugins.models.glm5_next import ascendc_sparse_attention as module


@pytest.mark.parametrize("rows,width", [(0, 2112), (4, 0), (8, 17), (128, 2112), (1024, 2112)])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_compaction(rows, width, device):
    if device == "npu":
        if os.environ.get("GLM53_TEST_NPU") != "1":
            pytest.skip("Set GLM53_TEST_NPU=1 on an idle NPU")
        pytest.importorskip("torch_npu")
    torch.manual_seed(53)
    x = torch.randint(-2, 40, (rows, 1, width), dtype=torch.int32)
    if rows and width:
        x[0] = -1
        if rows > 1:
            x[1] = 7  # duplicates must remain, in original order
    # Non-contiguous inputs exercise the public API.
    x = x.transpose(0, 2).contiguous().transpose(0, 2)
    actual = module.compact_sparse_indices(x.to(device), 32).cpu()
    expect = torch.full_like(x, -1)
    for row in range(rows):
        valid = x[row, 0][(x[row, 0] >= 0) & (x[row, 0] < 32)]
        expect[row, 0, : valid.numel()] = valid
    assert actual.dtype == torch.int32
    assert actual.is_contiguous()
    assert torch.equal(actual, expect)


def test_float_position_keys_preserve_order():
    # Check all possible keys for the optimized width, not just random values.
    width = 2112
    keys = torch.randperm(2 * width, generator=torch.Generator().manual_seed(53))
    assert torch.equal(keys.argsort(), keys.float().argsort())


@pytest.mark.parametrize("device,explicit_kv", [("Ascend910_9382", False), ("Ascend950", True)])
def test_sfa_kv_layout(device, explicit_kv):
    assert module.uses_explicit_kv(device) is explicit_kv


def test_reject_unsupported_soc():
    with pytest.raises(RuntimeError, match="Ascend910B"):
        module.uses_explicit_kv("Ascend910B")
