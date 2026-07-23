"""Lower bounded elementwise binary claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.add import (
    ADD_BINOP_RULE,
    ADD_CALL_RULE,
    ADD_TARGETS,
    MUL_BINOP_RULE,
)
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

_SUPPORTED = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}
_ADD_RULES = frozenset({ADD_CALL_RULE, ADD_BINOP_RULE})


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed bounded binary site, or return None."""
    is_add_binop = claimed.kind == "binop" and claimed.target == "+"
    is_mul_binop = claimed.kind == "binop" and claimed.target == "*"
    is_call = claimed.kind == "call" and claimed.target in ADD_TARGETS
    if not is_add_binop and not is_mul_binop and not is_call:
        return None
    if is_mul_binop:
        expected_rule = MUL_BINOP_RULE
        operation = "multiply"
        helper = "mul"
    elif is_add_binop:
        expected_rule = ADD_BINOP_RULE
        operation = "add"
        helper = "add"
    else:
        expected_rule = ADD_CALL_RULE
        operation = "add"
        helper = "add"
    if claimed.rule_id != expected_rule:
        raise ValueError(
            f"rextio-tensorflow {operation} lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(f"rextio-tensorflow functional {operation} lower forbids receivers")
    if claimed.keywords or len(claimed.operand_types) != 2:
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires two positional operands and no keywords"
        )
    left, right = claimed.operand_types[0], claimed.operand_types[1]
    if left is None or right is None:
        raise ValueError("rextio-tensorflow add lower requires resolved operand types")
    pair = (left, right)
    expected = _SUPPORTED.get(pair)
    if expected is None or claimed.result_type != expected:
        raise ValueError(
            f"rextio-tensorflow {operation} lower operand/result types changed between claim and lower: "
            f"operands={claimed.operand_types!r} result={claimed.result_type!r}"
        )
    if len(ctx.operands) != 2:
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires two ctx.operands entries; "
            f"got {len(ctx.operands)}"
        )
    a, b = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{helper}(&{a}, &{b})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
