"""Allocator and port settings passed to SGLang workers."""

import pytest

from slime.backends.sglang_utils.runtime import prepare_server_environment


@pytest.mark.parametrize(
    "allocator,expected",
    [
        ("expandable_segments:True", None),
        (
            "pinned_use_cuda_host_register:True,expandable_segments:False,max_split_size_mb:128",
            "pinned_use_cuda_host_register:True,max_split_size_mb:128",
        ),
        ("max_split_size_mb:128, expandable_segments:1", "max_split_size_mb:128"),
        ("pinned_use_cuda_host_register:True", "pinned_use_cuda_host_register:True"),
    ],
)
def test_server_environment_preserves_supported_allocator_options(allocator, expected):
    environ = {"PYTORCH_ALLOC_CONF": allocator, "PYTORCH_CUDA_ALLOC_CONF": allocator}
    prepare_server_environment(environ)
    assert environ.get("PYTORCH_ALLOC_CONF") == expected
    assert environ.get("PYTORCH_CUDA_ALLOC_CONF") == expected
    assert environ["MC_MIN_PRC_PORT"] == "27000"
    assert environ["MC_MAX_PRC_PORT"] == "28000"


@pytest.mark.parametrize("name", ["MC_MIN_RPC_PORT", "MC_MAX_RPC_PORT", "MC_MIN_PRC_PORT", "MC_MAX_PRC_PORT"])
def test_server_environment_preserves_explicit_mooncake_range(name):
    environ = {name: "31000"}
    prepare_server_environment(environ)
    assert environ == {name: "31000"}
