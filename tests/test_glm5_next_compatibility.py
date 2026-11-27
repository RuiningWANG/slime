"""Tests for model backend selection and request validation."""

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from slime_plugins.models.glm5_next.packed_sequence import packed_boundaries

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative_path, names, namespace):
    tree = ast.parse((ROOT / relative_path).read_text())
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(tree, relative_path, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("backend", ["npu", "cuda", "cpu"])
@pytest.mark.parametrize("setting", [None, "auto", "npu", "cuda"])
def test_attention_uses_framework_backend(monkeypatch, backend, setting):
    if setting is None:
        monkeypatch.delenv("SLIME_ACCELERATOR", raising=False)
    else:
        monkeypatch.setenv("SLIME_ACCELERATOR", setting)
    path = ROOT / "slime_plugins/models/glm5_next/attention.py"
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.If))
    imported = []

    def import_backend(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        return SimpleNamespace(**{name: object() for name in fromlist})

    ns = dict(
        accelerator=SimpleNamespace(device_type=lambda: backend),
        __builtins__={**vars(builtins), "__import__": import_backend},
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    assert ("npu_backend" in imported) == (backend == "npu")
    assert ("fla.modules" in imported) == (backend != "npu")


@pytest.mark.parametrize("backend,expected", [("npu", 1e-5), ("cuda", 1e-6), ("cpu", 1e-6)])
def test_vision_normalization_uses_framework_backend(backend, expected):
    path = ROOT / "slime_plugins/models/glm5_next/vision.py"
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "qk_norm_eps" for t in node.targets)
    )
    ns = dict(accelerator=SimpleNamespace(device_type=lambda: backend), config=SimpleNamespace(rms_norm_eps=1e-5))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    assert ns["qk_norm_eps"] == expected


@pytest.mark.parametrize("boundaries", [[], [0], [0, 8, 4, 12], [1, 12], [0, 13], [0, 1.5, 12], [False, 12], [[0, 12]], 12])
def test_invalid_packed_boundaries(boundaries):
    with pytest.raises(ValueError):
        packed_boundaries(boundaries, 12)


def test_empty_packed_sequences_preserve_token_offsets():
    assert packed_boundaries([0, 0, 4, 4, 12], 12) == [0, 0, 4, 4, 12]
    ns = load_definitions(
        "slime_plugins/models/glm5_next/npu_backend.py", {"_boundaries"}, {"packed_boundaries": packed_boundaries}
    )
    assert ns["_boundaries"](None, 12) == [0, 12]


def test_kpool_rejects_fractional_offsets_before_kernel():
    import torch
    from slime_plugins.models.glm5_next.sglang_kpool import pooled_indexer_topk

    q = torch.empty(3, 1, 128, dtype=torch.bfloat16)
    k = torch.empty(3, 128, dtype=q.dtype)
    offsets = torch.tensor([0.0, 1.5, 3.0])
    with pytest.raises(ValueError, match="integer offsets"):
        pooled_indexer_topk(q, k, None, None, None, offsets, 4, 2)


def test_conv1d_missing_fla_bindings_reports_install_hint(monkeypatch):
    import sys
    from slime_plugins.models.glm5_next.ascendc_conv1d import _load_ops

    monkeypatch.setitem(sys.modules, "fla_npu.ops.ascendc", None)
    _load_ops.cache_clear()
    with pytest.raises(ImportError, match="bde95747") as error:
        _load_ops()
    assert isinstance(error.value.__cause__, ImportError)


def test_convolution_without_cp_context():
    calls = []
    ns = load_definitions(
        "slime_plugins/models/glm5_next/npu_backend.py",
        {"causal_conv1d"},
        dict(
            _can_shard_heads=lambda *a: False,
            _gather=lambda x, c: x,
            _local=lambda x, c: x,
            _convolve=lambda *args: calls.append(args) or "output",
        ),
    )
    assert ns["causal_conv1d"]("input", "weight", "silu", None) == ("output", None)
    assert calls == [("input", "weight", "silu", None)]


@pytest.mark.parametrize("device,dtype,expected", [("cpu", "bfloat16", False), ("npu", "float32", False), ("npu", "bfloat16", True)])
def test_fused_mhc_requires_npu_bfloat16(device, dtype, expected):
    import torch

    ns = load_definitions("slime_plugins/models/glm5_next/ops.py", {"_use_fused_mhc"}, {"torch": torch})
    tensor = SimpleNamespace(device=SimpleNamespace(type=device), dtype=getattr(torch, dtype))
    assert ns["_use_fused_mhc"](tensor) is expected


@pytest.mark.parametrize(
    "setting,selected,direct",
    [("npu", "npu", True), ("auto", "npu", True), (None, "npu", True), ("cuda", "cuda", False)],
)
def test_r3_uses_selected_accelerator(monkeypatch, setting, selected, direct):
    if setting is None:
        monkeypatch.delenv("SLIME_ACCELERATOR", raising=False)
    else:
        monkeypatch.setenv("SLIME_ACCELERATOR", setting)
    tree = ast.parse((ROOT / "slime/rollout/sglang_rollout.py").read_text())
    generate = next(node for node in tree.body if getattr(node, "name", None) == "generate")
    branch = next(node for node in generate.body if isinstance(node, ast.If) and "/workers" in ast.unparse(node))
    scope = dict(
        accelerator=SimpleNamespace(device_type=lambda: selected), args=SimpleNamespace(use_rollout_routing_replay=True)
    )
    assert eval(compile(ast.Expression(branch.test), "<r3>", "eval"), scope) is direct
    scope["args"].use_rollout_routing_replay = False
    assert not eval(compile(ast.Expression(branch.test), "<r3>", "eval"), scope)
