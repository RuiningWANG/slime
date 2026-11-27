"""Merger storage, checkpoint mappings, and optimizer-update regression tests.

Extract the relevant definitions to avoid initializing Megatron process groups.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.glm_ocr.modeling_glm_ocr import GlmOcrVisionPatchMerger

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    nodes = [
        n
        for n in ast.parse((ROOT / path).read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


ns = dict(torch=torch, nn=nn, F=F, GlmOcrVisionPatchMerger=GlmOcrVisionPatchMerger)
definitions("slime_plugins/models/glm5_next/ops.py", {"swiglu_with_limit"}, ns)
definitions(
    "slime_plugins/models/glm5_next/vision.py",
    {"vision_swiglu_with_limit", "Glm5NextVisionLayerNorm", "Glm5NextVisionMerger"},
    ns,
)
definitions("slime/backends/megatron_utils/hf_to_megatron/common.py", {"strip_mcore_wrappers"}, ns)
definitions("slime/backends/megatron_utils/hf_to_megatron/glm5_next.py", {"glm5_next_hf_tensor"}, ns)
definitions("slime/backends/megatron_utils/megatron_to_hf/glm5_next.py", {"convert_glm5_next_to_hf"}, ns)
definitions("slime/backends/megatron_utils/megatron_to_hf/__init__.py", {"convert_to_hf"}, ns)
CONFIG = SimpleNamespace(out_hidden_size=16, projection_intermediate_size=24, hidden_act="gelu", swiglu_limit=10.0)


class Reader(dict):
    def get_tensor(self, name):
        return self[name]


def model():
    return ns["Glm5NextVisionMerger"](CONFIG)


def test_legacy_and_packed_state_dict_roundtrip():
    old = GlmOcrVisionPatchMerger(16, 24, "gelu")
    state = old.state_dict()
    m = model()
    m.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        m.gate_up_proj.weight, torch.cat((old.gate_proj.weight, old.up_proj.weight)), rtol=0, atol=0
    )
    assert not hasattr(m, "gate_proj") and not hasattr(m, "up_proj")
    assert "gate_up_proj.weight" in dict(m.named_parameters())
    assert len(list(m.parameters())) == len(list(old.parameters())) - 1
    restored = model()
    restored.load_state_dict(m.state_dict(), strict=True)
    x = torch.randn(9, 16)
    torch.testing.assert_close(m(x), restored(x), rtol=0, atol=0)
    assert "gate_proj.weight" in state  # loader must not mutate caller's dict


def test_incomplete_legacy_state_dict_fails():
    state = GlmOcrVisionPatchMerger(16, 24, "gelu").state_dict()
    del state["up_proj.weight"]
    with pytest.raises(RuntimeError):
        model().load_state_dict(state, strict=True)


@pytest.mark.parametrize("wrapper", ["", "module.", "module.module."])
def test_streaming_hf_load_and_rollout_export(wrapper):
    prefix = "model.visual.merger."
    reader = Reader({prefix + "gate_proj.weight": torch.randn(24, 16), prefix + "up_proj.weight": torch.randn(24, 16)})
    name = prefix + "gate_up_proj.weight"
    packed = ns["glm5_next_hf_tensor"](wrapper + name, reader, None)
    exported = dict(ns["convert_to_hf"](None, "glm5_next", wrapper + name, packed))
    assert exported.keys() == reader.keys()
    for key in reader:
        torch.testing.assert_close(exported[key], reader[key], rtol=0, atol=0)
    torch.testing.assert_close(ns["glm5_next_hf_tensor"](name, Reader({name: packed}), None), packed, rtol=0, atol=0)
    # Other model families retain their visual parameter mapping.
    assert ns["convert_to_hf"](None, "qwen3_vl", wrapper + name, packed)[0][0] == name


def test_forward_backward_optimizer_and_no_weight_concat():
    m = model()
    x = torch.randn(9, 16, requires_grad=True)
    z = m.act1(m.post_projection_norm(m.proj(x)))
    reference = m.down_proj(ns["vision_swiglu_with_limit"](F.linear(z, m.gate_up_proj.weight), 10.0))
    with patch.object(torch, "cat", side_effect=AssertionError("forward must not concatenate weights")):
        actual = m(x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    params = [x] + list(m.parameters())
    dy = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(reference, params, dy)
    actual.backward(dy)
    for p, g in zip(params, expected_grads):
        torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
        assert torch.isfinite(p.grad).all()
    before = m.gate_up_proj.weight.detach().clone()
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    assert not torch.equal(before, m.gate_up_proj.weight)
    gate, up = m.gate_up_proj.weight.chunk(2, dim=0)
    assert not torch.equal(before[:24], gate) and not torch.equal(before[24:], up)
    exported = dict(
        ns["convert_to_hf"](None, "glm53", "model.visual.merger.gate_up_proj.weight", m.gate_up_proj.weight)
    )
    torch.testing.assert_close(exported["model.visual.merger.gate_proj.weight"], gate, rtol=0, atol=0)
    assert not torch.equal(actual.detach(), m(x).detach())
