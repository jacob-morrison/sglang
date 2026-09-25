"""Adaptive-compute K policies with real reduced dispatch (adaptive_routing package).

Opt-in through the environment: ``ADAPTIVE_ROUTING_POLICY`` (``normalized`` or
``reference``), ``ADAPTIVE_ROUTING_K``, and the optional ``ADAPTIVE_ROUTING_SPEC``,
``ADAPTIVE_ROUTING_NATIVE_K``, ``ADAPTIVE_ROUTING_NUM_EXPERTS``, ``ADAPTIVE_ROUTING_MASKED``
and ``ADAPTIVE_ROUTING_FROM_LOGITS`` switches. When active, ``select_experts`` runs the
engine's own selection at width ``max(K, native_k)`` and then narrows the rows to ``[M, K]``
under the policy, before capture, EPLB post-processing and the MoE kernels. The Triton
runner reads K from ``topk_ids.shape[1]``, so fewer experts are dispatched.

Fails closed on non-standard top-k output formats (Triton-kernels routing, bypassed
in-kernel routing), custom routing functions, fused shared experts, ``sqrtsoftplus``
scoring, and expert parallelism (all-to-all dispatchers size buffers from the config K)
unless ``ADAPTIVE_ROUTING_ALLOW_EP=1``.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKConfig, TopKOutputFormat

logger = logging.getLogger(__name__)

try:
    import adaptive_routing as _ar
except ImportError:  # optional dependency; only needed when a policy is set
    _ar = None

POLICY_ENV = "ADAPTIVE_ROUTING_POLICY"
_policy: Any = None
_policy_loaded = False
_specs: dict[int, Any] = {}
_state: dict[int, dict[str, Any]] = {}
_tracker: Any = None


def requested_policy() -> Any:
    """The ``ADAPTIVE_ROUTING_*`` policy (cached per process), or None when unset."""
    global _policy, _policy_loaded
    if _policy_loaded:
        return _policy
    if os.environ.get(POLICY_ENV, "").strip():
        if _ar is None:
            raise ImportError(
                f"{POLICY_ENV} is set but the adaptive_routing package is not installed"
            )
        policy = _ar.policy_from_env()
        _policy = None if policy.kind == "native" else policy
    _policy_loaded = True
    return _policy


def _is_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _get_tracker() -> Any:
    global _tracker
    if _tracker is None:
        _tracker = _ar.RealizedKTracker(max_k=64)
        atexit.register(_log_summary)
    return _tracker


def _log_summary() -> None:
    if _tracker is None or not _tracker.tokens:
        return
    summary = _tracker.summary()
    try:
        logger.info("%s", summary)
    except Exception:  # log streams may already be closed at interpreter exit
        pass
    try:
        if sys.__stderr__ is not None and not sys.__stderr__.closed:
            sys.__stderr__.write(summary + "\n")
            sys.__stderr__.flush()
    except Exception:
        pass


def check_output_format(output_format: TopKOutputFormat) -> None:
    """Only the standard (weights, ids) format can carry a narrowed K."""
    if requested_policy() is None:
        return
    from sglang.srt.layers.moe.topk import TopKOutputFormat

    if output_format != TopKOutputFormat.STANDARD:
        raise RuntimeError(
            f"adaptive routing: top-k output format {output_format.name} routes inside "
            "the MoE kernel (Triton-kernels or bypassed backend); pick a backend with "
            f"standard top-k output or unset {POLICY_ENV}"
        )


def _check_topk_config(topk_config: TopKConfig) -> None:
    if topk_config.custom_routing_function is not None:
        raise RuntimeError(
            "adaptive routing: this model uses a custom routing function; no spec applies"
        )
    if topk_config.num_fused_shared_experts > 0:
        raise RuntimeError(
            "adaptive routing: fused shared experts occupy top-k columns; unsupported"
        )
    if topk_config.scoring_func not in ("softmax", "sigmoid"):
        raise RuntimeError(
            f"adaptive routing: scoring function {topk_config.scoring_func!r} has no spec"
        )
    if os.environ.get("ADAPTIVE_ROUTING_ALLOW_EP", "0") != "1":
        try:
            from sglang.srt.distributed import get_moe_expert_parallel_world_size

            ep_size = get_moe_expert_parallel_world_size()
        except Exception:  # not initialised (unit tests)
            ep_size = 1
        if ep_size > 1:
            raise RuntimeError(
                "adaptive routing: expert parallelism sizes all-to-all buffers from the "
                "config K; run TP/DP, or set ADAPTIVE_ROUTING_ALLOW_EP=1 after checking "
                "the dispatcher"
            )


def _build_spec(topk_config: TopKConfig, num_experts: int) -> Any:
    """Spec from the TopK config, cross-checked against the environment.

    SGLang applies ``routed_scaling_factor`` to the weights only when
    ``apply_routed_scaling_factor_on_output`` is set (otherwise the MoE layer scales its
    output), and never on the plain fused top-k path.
    """
    overrides = _ar.spec_overrides_from_env()
    top_k = topk_config.top_k
    native_k = int(overrides.get("native_k", top_k))
    if native_k > top_k:
        raise RuntimeError(
            f"adaptive routing: ADAPTIVE_ROUTING_NATIVE_K={native_k} exceeds top_k={top_k}"
        )
    if int(overrides.get("num_experts", num_experts)) != num_experts:
        raise RuntimeError(
            f"adaptive routing: router has {num_experts} experts but "
            f"ADAPTIVE_ROUTING_NUM_EXPERTS={overrides['num_experts']}"
        )
    has_bias = topk_config.correction_bias is not None
    grouped = bool(topk_config.use_grouped_topk)
    scale = 1.0
    if (
        (grouped or has_bias)
        and topk_config.renormalize
        and topk_config.routed_scaling_factor
        and topk_config.apply_routed_scaling_factor_on_output
    ):
        scale = float(topk_config.routed_scaling_factor)
    values = dict(
        score_fn=topk_config.scoring_func,
        native_k=native_k,
        num_experts=num_experts,
        renormalize=bool(topk_config.renormalize),
        scale=scale,
        selection_bias=has_bias,
        num_expert_group=int(topk_config.num_expert_group or 1) if grouped else 1,
        topk_group=int(topk_config.topk_group or 1) if grouped else 1,
        source="sglang-topk-config",
    )
    family = overrides.get("family")
    if family:
        base = _ar.template(family)
        mismatches = {
            name: (values[name], getattr(base, name))
            for name in ("score_fn", "renormalize", "selection_bias")
            if values[name] != getattr(base, name)
        }
        if mismatches:
            raise RuntimeError(
                f"adaptive routing: TopK config disagrees with the {base.family} template "
                f"on {mismatches}"
            )
        return replace(base, verified=True, **values)
    return _ar.RoutingSpec(family="runtime", **values)


def selection_width(topk_config: TopKConfig) -> int:
    """Width the engine's selection must run at: ``max(K, native_k, top_k)``.

    Runs first in ``select_experts``, so unsupported configurations fail closed
    before the engine's own routing (custom functions, fused shared experts) runs.
    """
    policy = requested_policy()
    if policy is None:
        return topk_config.top_k
    if id(topk_config) not in _specs:
        _check_topk_config(topk_config)
    native_k = int(_ar.spec_overrides_from_env().get("native_k", topk_config.top_k))
    return max(policy.k or native_k, native_k, topk_config.top_k)


# Per-row K (drafter-gated verification): a device int32 vector with one K per token row of the
# current forward, set by the speculative worker right before the target's verify forward and
# cleared right after, so prefill / decode outside that window run the plain policy. Requires a
# masked policy at native width (the row mask zeroes the weights past K_row; expert ids keep their
# native width, so this is a fidelity mechanism, not reduced dispatch - see ROW_K_SKIP below).
_row_k: Optional[torch.Tensor] = None
_row_k_stats = {"rows": 0, "sum_k": 0.0, "calls": 0}
ROW_K_ENV = "ADAPTIVE_ROUTING_ROW_K"          # "1": allocate the per-row K buffer on first use (before any graph capture)
ROW_K_ROWS_ENV = "ADAPTIVE_ROUTING_ROW_K_ROWS"  # buffer rows (>= tokens of the largest forward); default 131072


def set_row_k(row_k: Optional[torch.Tensor]) -> None:
    """Install (or clear, with None) the per-row K vector for the forward that follows."""
    global _row_k
    _row_k = row_k


def ensure_row_k(device, native_k: int) -> torch.Tensor:
    """Allocate the static per-row K buffer (filled with the native K) if it does not exist yet, and
    install it. Called from apply_policy on the first routed batch when ROW_K_ENV=1, so the buffer
    exists before any CUDA graph is captured and every captured graph reads it; the speculative
    worker fetches the same buffer and writes its per-row Ks into it before each verify forward."""
    global _row_k
    if _row_k is None:
        if _is_capturing():
            raise RuntimeError("adaptive routing: per-row K buffer requested during CUDA graph capture; it must be allocated by an eager (warmup) forward first")
        rows = int(os.environ.get(ROW_K_ROWS_ENV, str(1 << 17)))
        _row_k = torch.full((rows,), int(native_k), dtype=torch.int32, device=device)
        logger.info("adaptive routing: per-row K buffer allocated, %d rows at native K=%d", rows, native_k)
    return _row_k


def row_k_stats() -> dict:
    return dict(_row_k_stats)


def _apply_row_k(weights: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the weights of the slots at or past each row's K. With ROW_K_SKIP=1 also set those
    slots' expert ids to -1, which the Triton runner's expert-parallel filter skips (real reduced
    dispatch; validated separately)."""
    row_k = _row_k
    if row_k is None:
        return weights, ids
    m = weights.shape[0]
    if row_k.shape[0] < m:
        raise RuntimeError(f"adaptive routing: row K vector has {row_k.shape[0]} rows, forward has {m}")
    keep = torch.arange(weights.shape[1], device=weights.device)[None, :] < row_k[:m, None]
    weights = weights * keep.to(weights.dtype)
    if os.environ.get("ADAPTIVE_ROUTING_ROW_K_SKIP", "0") == "1":
        ids = torch.where(keep, ids, torch.full_like(ids, -1))
    if not _is_capturing():
        _row_k_stats["rows"] += m
        _row_k_stats["sum_k"] += float(row_k[:m].sum())
        _row_k_stats["calls"] += 1
    return weights, ids


def apply_policy(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    router_logits: torch.Tensor,
    topk_config: TopKConfig,
    layer_id: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Narrow the engine's selection to the policy's K. No-op when no policy is set."""
    policy = requested_policy()
    if policy is None:
        if _row_k is not None:
            raise RuntimeError("adaptive routing: per-row K needs ADAPTIVE_ROUTING_POLICY=reference with MASKED=1")
        return topk_weights, topk_ids
    key = id(topk_config)
    spec = _specs.get(key)
    if spec is None:
        _check_topk_config(topk_config)
        spec = _build_spec(topk_config, router_logits.shape[-1])
        _specs[key] = spec
        _state[key] = {"calls": 0, "index": len(_state)}
        from_logits = (
            spec.selection_bias
            or spec.grouped
            or not spec.renormalize
            or os.environ.get("ADAPTIVE_ROUTING_FROM_LOGITS", "0") == "1"
        )
        _state[key]["from_logits"] = from_logits
        if _state[key]["index"] == 0:
            logger.info(
                "adaptive routing: %s with %s; %s path",
                policy.describe(spec),
                spec.describe(),
                "logits" if from_logits else "post-hoc",
            )
    state = _state[key]
    first = state["calls"] == 0 and not _is_capturing()
    k = policy.resolve_k(spec)
    if _row_k is None and os.environ.get(ROW_K_ENV, "0") == "1":
        ensure_row_k(topk_weights.device, spec.native_k)
    if state["from_logits"]:
        bias = topk_config.correction_bias
        out = _ar.route(router_logits, spec, policy, selection_bias=bias)
        new_weights, new_ids = out.weights, out.ids.to(topk_ids.dtype)
    else:
        if first:
            _ar.assert_sorted_descending(topk_weights, where=f"layer {layer_id}")
        new_weights, new_ids = _ar.reweight_selected(
            topk_weights, topk_ids, spec, policy
        )
    if first:
        expected = max(k, spec.native_k) if policy.masked else k
        _ar.assert_width(new_ids, expected, where=f"layer {layer_id}")
        if state["index"] == 0:
            logger.info(
                "adaptive routing: first batch ok, %d tokens, width %d",
                new_ids.shape[0],
                new_ids.shape[-1],
            )
    if _row_k is not None:
        if not policy.masked:
            raise RuntimeError("adaptive routing: per-row K needs a masked (native-width) policy")
        new_weights, new_ids = _apply_row_k(new_weights, new_ids)
    if os.environ.get("ADAPTIVE_ROUTING_TELEMETRY", "1") != "0":
        _get_tracker().record_fixed(state["index"], k, new_ids.shape[0])
    state["calls"] += 1
    return new_weights, new_ids
