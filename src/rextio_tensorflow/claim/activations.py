"""Fail-closed claims for ``tf.nn.relu`` and ``tf.nn.sigmoid``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_RELU,
    DIAGNOSTIC_SIGMOID,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

RELU_RULE = "rextio-tensorflow/relu-f32-cpu-2d"
SIGMOID_RULE = "rextio-tensorflow/sigmoid-f32-cpu-2d"

RELU_TARGETS = frozenset({"tensorflow.nn.relu", "tf.nn.relu"})
SIGMOID_TARGETS = frozenset({"tensorflow.nn.sigmoid", "tf.nn.sigmoid"})


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim relu/sigmoid on float32 CPU rank-2 tensors, else None."""
    if site.kind != "call":
        return None
    if site.target in RELU_TARGETS:
        return _unary_claim(
            site,
            rule_id=RELU_RULE,
            diagnostic=DIAGNOSTIC_RELU,
            name="relu",
        )
    if site.target in SIGMOID_TARGETS:
        return _unary_claim(
            site,
            rule_id=SIGMOID_RULE,
            diagnostic=DIAGNOSTIC_SIGMOID,
            name="sigmoid",
        )
    return None


def _unary_claim(
    site: ClaimSite,
    *,
    rule_id: str,
    diagnostic: str,
    name: str,
) -> ClaimResult:
    if site.receiver is not None:
        return NotCovered()
    operands = tuple(site.operand_types)
    if site.keywords or len(operands) != 1:
        return reject(
            site,
            diagnostic,
            f"only tf.nn.{name}(x) with one positional tensor is supported",
            f"Call tf.nn.{name}(x) with a single TensorF32Cpu2D argument.",
        )
    (operand,) = operands
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
            DIAGNOSTIC_UNSUPPORTED,
            f"Alpha {name} requires float32 CPU rank-2; got {operand!r}",
            "Use TensorF32Cpu2D for the activation operand.",
        )
    return Claimed(rule_id=rule_id, result_type=TENSOR_F32_CPU_2D)


__all__ = [
    "RELU_RULE",
    "RELU_TARGETS",
    "SIGMOID_RULE",
    "SIGMOID_TARGETS",
    "try_claim",
]
