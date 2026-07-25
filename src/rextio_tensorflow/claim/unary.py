"""Fail-closed claims for bounded TensorFlow math unary operations."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ABS,
    DIAGNOSTIC_EXP,
    DIAGNOSTIC_LOG,
    DIAGNOSTIC_NEGATIVE,
    DIAGNOSTIC_SQRT,
    DIAGNOSTIC_SQUARE,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

ABS_RULE = "rextio-tensorflow/abs-f32-cpu"
NEGATIVE_RULE = "rextio-tensorflow/negative-f32-cpu"
SQUARE_RULE = "rextio-tensorflow/square-f32-cpu"
EXP_RULE = "rextio-tensorflow/exp-f32-cpu"
LOG_RULE = "rextio-tensorflow/log-f32-cpu"
SQRT_RULE = "rextio-tensorflow/sqrt-f32-cpu"

UNARY_RULES = {
    "abs": ABS_RULE,
    "negative": NEGATIVE_RULE,
    "square": SQUARE_RULE,
    "exp": EXP_RULE,
    "log": LOG_RULE,
    "sqrt": SQRT_RULE,
}

UNARY_DIAGNOSTICS = {
    "abs": DIAGNOSTIC_ABS,
    "negative": DIAGNOSTIC_NEGATIVE,
    "square": DIAGNOSTIC_SQUARE,
    "exp": DIAGNOSTIC_EXP,
    "log": DIAGNOSTIC_LOG,
    "sqrt": DIAGNOSTIC_SQRT,
}

TARGET_TO_UNARY = {
    "tensorflow.abs": "abs",
    "tf.abs": "abs",
    "tensorflow.negative": "negative",
    "tf.negative": "negative",
    "tensorflow.square": "square",
    "tf.square": "square",
    "tensorflow.exp": "exp",
    "tf.exp": "exp",
    "tensorflow.math.log": "log",
    "tf.math.log": "log",
    "tensorflow.math.sqrt": "sqrt",
    "tf.math.sqrt": "sqrt",
}


def _literal_metadata_is_aligned(site: ClaimSite) -> bool:
    return not site.operand_literals or (
        len(site.operand_literals) == 1
        and not site.operand_literals[0].is_literal
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact unary aliases on float32 CPU rank-1/rank-2 tensors."""
    if site.kind != "call":
        return None
    operation = TARGET_TO_UNARY.get(site.target)
    if operation is None:
        return None
    if site.receiver is not None:
        return NotCovered()
    diagnostic = UNARY_DIAGNOSTICS[operation]
    if (
        len(site.operand_types) != 1
        or site.keywords
        or not _literal_metadata_is_aligned(site)
    ):
        return reject(
            site,
            diagnostic,
            f"bounded {operation} requires exactly one positional tensor",
            f"Call the supported {operation} spelling with one annotated tensor.",
        )
    operand = site.operand_types[0]
    if operand is None:
        return NotCovered()
    if not is_tensor_type(operand):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand type is outside the float32 CPU tensor surface",
            "Annotate the operand as TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    if operand not in {TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D}:
        return reject(
            site,
            diagnostic,
            f"bounded {operation} requires float32 CPU rank-1 or rank-2; got {operand!r}",
            "Use TensorF32Cpu1D or TensorF32Cpu2D for the unary operand.",
        )
    return Claimed(rule_id=UNARY_RULES[operation], result_type=operand)


__all__ = [
    "ABS_RULE",
    "EXP_RULE",
    "LOG_RULE",
    "NEGATIVE_RULE",
    "SQRT_RULE",
    "SQUARE_RULE",
    "TARGET_TO_UNARY",
    "UNARY_RULES",
    "try_claim",
]
