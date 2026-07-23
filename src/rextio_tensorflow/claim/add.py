"""Fail-closed claims for elementwise add and multiplication."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ADD,
    DIAGNOSTIC_ADD_BINOP,
    DIAGNOSTIC_MUL_BINOP,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

ADD_CALL_RULE = "rextio-tensorflow/add-call-f32-cpu"
ADD_BINOP_RULE = "rextio-tensorflow/add-binop-f32-cpu"
MUL_BINOP_RULE = "rextio-tensorflow/mul-binop-f32-cpu"
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

_SUPPORTED_PAIRS: dict[tuple[str, str], str] = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim add / ``+`` on the Alpha float32 CPU surface, else None."""
    if site.kind == "binop" and site.target == "+":
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=ADD_BINOP_RULE,
            diagnostic=DIAGNOSTIC_ADD_BINOP,
            operation="add",
            syntax="tf.add(x, y) or x + y",
        )
    if site.kind == "binop" and site.target == "*":
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=MUL_BINOP_RULE,
            diagnostic=DIAGNOSTIC_MUL_BINOP,
            operation="multiply",
            syntax="x * y",
        )
    if site.kind == "call" and site.target in ADD_TARGETS:
        if site.receiver is not None:
            return NotCovered()
        if site.keywords:
            return reject(
                site,
                DIAGNOSTIC_ADD,
                "keywords on tf.add are not supported",
                "Call tf.add(x, y) with two positional tensors only.",
            )
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=ADD_CALL_RULE,
            diagnostic=DIAGNOSTIC_ADD,
            operation="add",
            syntax="tf.add(x, y) or x + y",
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
    "MUL_BINOP_RULE",
    "try_claim",
]
