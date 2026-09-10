"""Adaptive-compute K policies in select_experts (adaptive_routing package). GPU only.

Run: ADAPTIVE_ROUTING_POLICY=reference ADAPTIVE_ROUTING_K=4 python -m pytest test/manual/test_adaptive_routing.py
The policy is read once per process, so each configuration is one pytest invocation.
"""

import os

import pytest
import torch

adaptive_routing = pytest.importorskip("adaptive_routing")

from sglang.srt.layers.moe import adaptive_routing as adapter  # noqa: E402
from sglang.srt.layers.moe.topk import TopKConfig, select_experts  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused top-k kernels need a GPU"
)

E, K0, M = 256, 8, 64


def _policy():
    policy = adapter.requested_policy()
    if policy is None:
        pytest.skip("set ADAPTIVE_ROUTING_POLICY / ADAPTIVE_ROUTING_K to run")
    return policy


def _inputs():
    torch.manual_seed(0)
    return (
        torch.randn(M, 16, device="cuda", dtype=torch.bfloat16),
        torch.randn(M, E, device="cuda") * 3,
    )


def test_select_experts_narrows_to_policy_k():
    policy = _policy()
    hidden, logits = _inputs()
    config = TopKConfig(top_k=K0, renormalize=True)
    out = select_experts(hidden, logits, config, layer_id=0)
    spec = adapter._specs[id(config)]
    k = policy.resolve_k(spec)
    expected_width = max(k, K0) if policy.masked else k
    assert out.topk_ids.shape == (M, expected_width)
    assert out.topk_weights.shape == (M, expected_width)

    reference = adaptive_routing.route(logits, spec, policy)
    assert torch.equal(out.topk_ids.long(), reference.ids.long())
    torch.testing.assert_close(
        out.topk_weights.float(), reference.weights, rtol=1e-4, atol=1e-5
    )
    if policy.kind == "normalized" and not policy.masked:
        torch.testing.assert_close(
            out.topk_weights.float().sum(-1),
            torch.full((M,), spec.scale, device="cuda"),
            rtol=1e-4,
            atol=1e-5,
        )


def test_survivors_match_native_selection():
    policy = _policy()
    hidden, logits = _inputs()
    config = TopKConfig(top_k=K0, renormalize=True)
    out = select_experts(hidden, logits, config, layer_id=1)
    spec = adapter._specs[id(config)]
    k = policy.resolve_k(spec)
    # native selection at width max(k, K0), untouched by the policy
    os.environ["ADAPTIVE_ROUTING_POLICY"] = ""
    adapter._policy_loaded = False
    try:
        native = select_experts(
            hidden, logits, TopKConfig(top_k=max(k, K0), renormalize=True), layer_id=2
        )
    finally:
        os.environ["ADAPTIVE_ROUTING_POLICY"] = policy.kind
        adapter._policy_loaded = False
    width = min(k, K0)
    assert torch.equal(
        out.topk_ids[:, :width].long(), native.topk_ids[:, :width].long()
    )
    if policy.kind == "reference" and k < K0:
        # reference keeps the native weights of the survivors
        torch.testing.assert_close(
            out.topk_weights[:, :width].float(),
            native.topk_weights[:, :width].float(),
            rtol=1e-4,
            atol=1e-5,
        )


def test_fail_closed_on_custom_routing_and_fused_shared_experts():
    _policy()
    hidden, logits = _inputs()
    with pytest.raises(RuntimeError, match="custom routing"):
        select_experts(
            hidden,
            logits,
            TopKConfig(top_k=K0, custom_routing_function=lambda **kw: None),
            layer_id=3,
        )
    with pytest.raises(RuntimeError, match="fused shared experts"):
        select_experts(
            hidden, logits, TopKConfig(top_k=K0, num_fused_shared_experts=1), layer_id=4
        )
