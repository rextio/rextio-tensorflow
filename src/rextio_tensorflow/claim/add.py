"""Fail-closed claims for the bounded elementwise binary surface."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ADD,
    DIAGNOSTIC_ADD_BINOP,
    DIAGNOSTIC_BIAS_ADD,
    DIAGNOSTIC_DIV_BINOP,
    DIAGNOSTIC_DIV_CALL,
    DIAGNOSTIC_MUL_BINOP,
    DIAGNOSTIC_MUL_CALL,
    DIAGNOSTIC_SUB_BINOP,
    DIAGNOSTIC_SUB_CALL,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

ADD_CALL_RULE = "rextio-tensorflow/add-call-f32-cpu"
ADD_BINOP_RULE = "rextio-tensorflow/add-binop-f32-cpu"
MUL_BINOP_RULE = "rextio-tensorflow/mul-binop-f32-cpu"
MUL_CALL_RULE = "rextio-tensorflow/mul-call-f32-cpu"
SUB_CALL_RULE = "rextio-tensorflow/sub-call-f32-cpu"
SUB_BINOP_RULE = "rextio-tensorflow/sub-binop-f32-cpu"
DIV_CALL_RULE = "rextio-tensorflow/div-call-f32-cpu"
DIV_BINOP_RULE = "rextio-tensorflow/div-binop-f32-cpu"
# Back-compat alias used by older tests / docs references.
ADD_RULE = ADD_CALL_RULE
ADD_TARGETS = frozenset(
    {
        "tensorflow.add",
        "tensorflow.math.add",
        "tf.add",
        "tf.math.add",
    }
)
MUL_TARGETS = frozenset(
    {
        "tensorflow.multiply",
        "tensorflow.math.multiply",
        "tf.multiply",
        "tf.math.multiply",
    }
)
SUB_TARGETS = frozenset(
    {
        "tensorflow.subtract",
        "tensorflow.math.subtract",
        "tf.subtract",
        "tf.math.subtract",
    }
)
DIV_TARGETS = frozenset(
    {
        "tensorflow.divide",
        "tensorflow.math.divide",
        "tf.divide",
        "tf.math.divide",
    }
)
BIAS_ADD_TARGETS = frozenset({"tensorflow.nn.bias_add", "tf.nn.bias_add"})

_SUPPORTED_PAIRS: dict[tuple[str, str], str] = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact bounded binary spellings on the float32 CPU surface."""
    binops = {
        "+": (ADD_BINOP_RULE, DIAGNOSTIC_ADD_BINOP, "add", "x + y"),
        "*": (MUL_BINOP_RULE, DIAGNOSTIC_MUL_BINOP, "multiply", "x * y"),
        "-": (SUB_BINOP_RULE, DIAGNOSTIC_SUB_BINOP, "subtract", "x - y"),
        "/": (DIV_BINOP_RULE, DIAGNOSTIC_DIV_BINOP, "divide", "x / y"),
    }
    if site.kind == "binop" and site.target in binops:
        rule_id, diagnostic, operation, syntax = binops[site.target]
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=rule_id,
            diagnostic=diagnostic,
            operation=operation,
            syntax=syntax,
        )
    if site.kind != "call":
        return None
    if site.target in BIAS_ADD_TARGETS:
        if site.receiver is not None:
            return NotCovered()
        return reject(
            site,
            DIAGNOSTIC_BIAS_ADD,
            (
                "tf.nn.bias_add remains fallback until its exact TFE data_format "
                "attribute, symbol provenance, and error semantics are certified"
            ),
            "Use an already-supported explicit add spelling or keep tf.nn.bias_add on Python.",
        )
    calls = {
        **{
            target: (
                ADD_CALL_RULE,
                DIAGNOSTIC_ADD,
                "add",
                "tf.add(x, y)",
            )
            for target in ADD_TARGETS
        },
        **{
            target: (
                MUL_CALL_RULE,
                DIAGNOSTIC_MUL_CALL,
                "multiply",
                "tf.multiply(x, y)",
            )
            for target in MUL_TARGETS
        },
        **{
            target: (
                SUB_CALL_RULE,
                DIAGNOSTIC_SUB_CALL,
                "subtract",
                "tf.subtract(x, y)",
            )
            for target in SUB_TARGETS
        },
        **{
            target: (
                DIV_CALL_RULE,
                DIAGNOSTIC_DIV_CALL,
                "divide",
                "tf.divide(x, y)",
            )
            for target in DIV_TARGETS
        },
    }
    if site.target in calls:
        if site.receiver is not None:
            return NotCovered()
        rule_id, diagnostic, operation, syntax = calls[site.target]
        if site.keywords:
            return reject(
                site,
                diagnostic,
                f"keywords on {operation} are not supported",
                f"Call {syntax} with two positional tensors only.",
            )
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=rule_id,
            diagnostic=diagnostic,
            operation=operation,
            syntax=syntax,
        )
    return None


def _claim_operands(
    site: ClaimSite,
    operands: tuple[str | None, ...],
    *,
    rule_id: str,
    diagnostic: str,
    operation: str,
    syntax: str,
) -> ClaimResult:
    if len(operands) != 2:
        return reject(
            site,
            diagnostic,
            f"{operation} requires exactly two operands",
            f"Write {syntax} with two tensors.",
        )
    left, right = operands
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU tensor surface",
            "Annotate operands with rextio_tensorflow.types.TensorF32Cpu2D / TensorF32Cpu1D.",
        )
    result = _SUPPORTED_PAIRS.get((left, right))
    if result is None:
        return reject(
            site,
            diagnostic,
            f"unsupported {operation} operand pair {operands!r}",
            "Use same-rank tensors or rank-2/rank-1 trailing broadcast.",
        )
    return Claimed(rule_id=rule_id, result_type=result)


__all__ = [
    "ADD_BINOP_RULE",
    "ADD_CALL_RULE",
    "ADD_RULE",
    "ADD_TARGETS",
    "BIAS_ADD_TARGETS",
    "DIV_BINOP_RULE",
    "DIV_CALL_RULE",
    "DIV_TARGETS",
    "MUL_BINOP_RULE",
    "MUL_CALL_RULE",
    "MUL_TARGETS",
    "SUB_BINOP_RULE",
    "SUB_CALL_RULE",
    "SUB_TARGETS",
    "try_claim",
]
