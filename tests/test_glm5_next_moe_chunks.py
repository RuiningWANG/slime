"""Check chunked dispatch, replay and recomputation against a two-expert CPU model."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from slime_plugins.models.glm5_next.moe_chunking import _expert_forward, _FixedRoutes, _moe_forward


@pytest.mark.parametrize("recompute", [False, True])
def test_dispatch_with_uneven_routes(monkeypatch, recompute):
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    monkeypatch.setenv("ROUTING_REPLAY_STAGE", "replay_forward")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    routes = torch.tensor([[1], [0], [0], [0], [1], [0], [0]])
    router = SimpleNamespace(routing_replay=_FixedRoutes(routes))
    source = torch.arange(21, dtype=torch.float32).reshape(7, 1, 3) / 10

    def evaluate(chunk):
        x = source.clone().requires_grad_()
        weights = torch.tensor([0.75, 1.25], requires_grad=True)

        def forward(values, intermediate_tensors=None, padding_mask=None):
            indexes = router.routing_replay.pop_forward()[:, 0]
            return (values * weights[indexes, None, None]).sin(), None

        layer = SimpleNamespace(
            _glm53_dispatch_chunk=chunk,
            _glm53_original_forward=forward,
            router=router,
            config=SimpleNamespace(moe_router_topk=1),
            ep_group=None,
        )

        def run(values):
            return _moe_forward(layer, values)[0]

        y = checkpoint(run, x, use_reentrant=True) if recompute else run(x)
        y.square().sum().backward()
        return y.detach(), x.grad, weights.grad

    for actual, expected in zip(evaluate(3), evaluate(0), strict=True):
        torch.testing.assert_close(actual, expected)
    assert torch.equal(router.routing_replay.pop_backward(), routes)


def test_expert_token_chunks_include_empty_expert():
    counts = torch.tensor([0, 5, 2])

    def evaluate(size):
        x = torch.arange(21, dtype=torch.float32).reshape(7, 3).requires_grad_()
        probs = torch.linspace(0.3, 0.9, 7).requires_grad_()

        def forward(values, sizes, routing_probs, *buffers):
            gain = torch.repeat_interleave(torch.tensor([0.5, 1.0, 1.5]), torch.as_tensor(sizes))
            return values * (gain * routing_probs.view(-1))[:, None], None

        experts = SimpleNamespace(_glm53_expert_chunk=size, _glm53_original_forward=forward)
        y, _ = _expert_forward(experts, x, counts, probs)
        return y, *torch.autograd.grad(y.sum(), (x, probs))

    for result, expected in zip(evaluate(2), evaluate(0), strict=True):
        torch.testing.assert_close(result, expected)
