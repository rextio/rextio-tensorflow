"""Fail-closed claims for rank-1/rank-2 TensorFlow unary activations."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_RELU,
    DIAGNOSTIC_RELU_1D,
    DIAGNOSTIC_SIGMOID,
    DIAGNOSTIC_SIGMOID_1D,
    DIAGNOSTIC_TANH,
    DIAGNOSTIC_TANH_1D,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

RELU_RULE = "rextio-tensorflow/relu-f32-cpu-2d"
SIGMOID_RULE = "rextio-tensorflow/sigmoid-f32-cpu-2d"
TANH_RULE = "rextio-tensorflow/tanh-f32-cpu-2d"
RELU_1D_RULE = "rextio-tensorflow/relu-f32-cpu-1d"
SIGMOID_1D_RULE = "rextio-tensorflow/sigmoid-f32-cpu-1d"
TANH_1D_RULE = "rextio-tensorflow/tanh-f32-cpu-1d"

RELU_TARGETS = frozenset({"tensorflow.nn.relu", "tf.nn.relu"})
SIGMOID_TARGETS = frozenset({"tensorflow.nn.sigmoid", "tf.nn.sigmoid"})
TANH_TARGETS = frozenset({"tensorflow.nn.tanh", "tf.nn.tanh"})


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim supported unary activations on float32 CPU rank-1/rank-2 tensors."""
    if site.kind != "call":
        return None
    if site.target in RELU_TARGETS:
        return _unary_claim(
            site,
            rank1_rule_id=RELU_1D_RULE,
            rank2_rule_id=RELU_RULE,
            diagnostic=DIAGNOSTIC_RELU,
            rank1_diagnostic=DIAGNOSTIC_RELU_1D,
            name="relu",
        )
    if site.target in SIGMOID_TARGETS:
        return _unary_claim(
            site,
            rank1_rule_id=SIGMOID_1D_RULE,
            rank2_rule_id=SIGMOID_RULE,
            diagnostic=DIAGNOSTIC_SIGMOID,
            rank1_diagnostic=DIAGNOSTIC_SIGMOID_1D,
            name="sigmoid",
        )
    if site.target in TANH_TARGETS:
        return _unary_claim(
            site,
            rank1_rule_id=TANH_1D_RULE,
            rank2_rule_id=TANH_RULE,
            diagnostic=DIAGNOSTIC_TANH,
            rank1_diagnostic=DIAGNOSTIC_TANH_1D,
            name="tanh",
        )
    return None


def _unary_claim(
    site: ClaimSite,
    *,
    rank1_rule_id: str,
    rank2_rule_id: str,
    diagnostic: str,
    rank1_diagnostic: str,
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
            (
                f"Call tf.nn.{name}(x) with a single TensorF32Cpu1D or "
                "TensorF32Cpu2D argument."
            ),
        )
    (operand,) = operands
    if operand is None:
        return NotCovered()
    if not is_tensor_type(operand):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand type is outside the float32 CPU tensor surface",
            (
                "Annotate the operand as rextio_tensorflow.types.TensorF32Cpu1D "
                "or TensorF32Cpu2D."
            ),
        )
    if operand == TENSOR_F32_CPU_1D:
        return Claimed(rule_id=rank1_rule_id, result_type=TENSOR_F32_CPU_1D)
    if operand == TENSOR_F32_CPU_2D:
        return Claimed(rule_id=rank2_rule_id, result_type=TENSOR_F32_CPU_2D)
    return reject(
        site,
        rank1_diagnostic,
        f"bounded {name} requires float32 CPU rank-1 or rank-2; got {operand!r}",
        "Use TensorF32Cpu1D or TensorF32Cpu2D for the activation operand.",
    )


__all__ = [
    "RELU_1D_RULE",
    "RELU_RULE",
    "RELU_TARGETS",
    "SIGMOID_1D_RULE",
    "SIGMOID_RULE",
    "SIGMOID_TARGETS",
    "TANH_1D_RULE",
    "TANH_RULE",
    "TANH_TARGETS",
    "try_claim",
]
