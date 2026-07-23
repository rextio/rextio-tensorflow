"""Fail-closed claims for the rank-2 TensorFlow classification head."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ARGMAX,
    DIAGNOSTIC_SOFTMAX,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
    is_tensor_type,
    reject,
)

SOFTMAX_RULE = "rextio-tensorflow/softmax-axis1-f32-cpu-2d"
ARGMAX_RULE = "rextio-tensorflow/argmax-axis1-i64-cpu-2d"
SOFTMAX_TARGETS = frozenset({"tensorflow.nn.softmax", "tf.nn.softmax"})
ARGMAX_TARGETS = frozenset({"tensorflow.argmax", "tf.argmax"})


def _literal_keywords(site: ClaimSite) -> dict[str, object] | None:
    values: dict[str, object] = {}
    for keyword in site.keywords:
        if not keyword.literal.is_literal:
            return None
        values[keyword.name] = keyword.literal.value
    return values


def _axis_one(
    site: ClaimSite, *, diagnostic: str, operation: str
) -> dict[str, object] | ClaimResult:
    values = _literal_keywords(site)
    if values is None:
        return reject(
            site,
            diagnostic,
            f"only static literal keywords are supported for {operation}",
            f"Write tf.{operation}(x, axis=1) with a literal axis.",
        )
    if len(site.operand_types) != 1:
        return reject(
            site,
            diagnostic,
            f"{operation} requires exactly one tensor operand",
            f"Write tf.{operation}(x, axis=1).",
        )
    if set(values) != {"axis"}:
        return reject(
            site,
            diagnostic,
            f"{operation} accepts only the literal axis=1 keyword on this Alpha surface",
            "Do not pass dynamic axes or additional keyword arguments.",
        )
    axis = values["axis"]
    if not isinstance(axis, int) or isinstance(axis, bool) or axis != 1:
        return reject(
            site,
            diagnostic,
            f"Alpha {operation} requires axis=1 literal; got axis={axis!r}",
            "Use the literal keyword axis=1.",
        )
    return values


def _rank2_input(
    site: ClaimSite, *, diagnostic: str, operation: str
) -> ClaimResult | None:
    (operand,) = site.operand_types
    if operand is None:
        return NotCovered()
    if not is_tensor_type(operand):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand type is outside the float32 CPU tensor surface",
            "Annotate the operand as rextio_tensorflow.types.TensorF32Cpu2D.",
        )
    if operand != TENSOR_F32_CPU_2D:
        return reject(
            site,
            diagnostic,
            f"Alpha {operation} requires a float32 CPU rank-2 operand; got {operand!r}",
            "Use TensorF32Cpu2D for the classification-head operand.",
        )
    return None


def _try_softmax(site: ClaimSite) -> ClaimResult:
    checked = _axis_one(site, diagnostic=DIAGNOSTIC_SOFTMAX, operation="nn.softmax")
    if not isinstance(checked, dict):
        return checked
    rejected = _rank2_input(site, diagnostic=DIAGNOSTIC_SOFTMAX, operation="softmax")
    if rejected is not None:
        return rejected
    return Claimed(rule_id=SOFTMAX_RULE, result_type=TENSOR_F32_CPU_2D)


def _try_argmax(site: ClaimSite) -> ClaimResult:
    checked = _axis_one(site, diagnostic=DIAGNOSTIC_ARGMAX, operation="argmax")
    if not isinstance(checked, dict):
        return checked
    rejected = _rank2_input(site, diagnostic=DIAGNOSTIC_ARGMAX, operation="argmax")
    if rejected is not None:
        return rejected
    return Claimed(rule_id=ARGMAX_RULE, result_type=TENSOR_I64_CPU_1D)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim only literal-axis rank-2 softmax/argmax classification steps."""
    if site.kind != "call":
        return None
    if site.receiver is not None:
        if site.target in SOFTMAX_TARGETS | ARGMAX_TARGETS:
            return NotCovered()
        return None
    if site.target in SOFTMAX_TARGETS:
        return _try_softmax(site)
    if site.target in ARGMAX_TARGETS:
        return _try_argmax(site)
    return None


__all__ = [
    "ARGMAX_RULE",
    "ARGMAX_TARGETS",
    "SOFTMAX_RULE",
    "SOFTMAX_TARGETS",
    "try_claim",
]
