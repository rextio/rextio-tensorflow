"""Lower add / ``+`` claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.add import (
    ADD_BINOP_RULE,
    ADD_CALL_RULE,
    ADD_TARGETS,
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
    """Lower a previously claimed add site, or return None."""
    is_binop = claimed.kind == "binop" and claimed.target == "+"
    is_call = claimed.kind == "call" and claimed.target in ADD_TARGETS
    if not is_binop and not is_call:
        return None
    if claimed.rule_id not in _ADD_RULES:
        raise ValueError(
            "rextio-tensorflow add lower received mismatched rule_id: "
            f"{claimed.rule_id!r}"
        )
    if is_binop and claimed.rule_id != ADD_BINOP_RULE:
        raise ValueError(
            "rextio-tensorflow add binop lower requires ADD_BINOP_RULE"
        )
    if is_call and claimed.rule_id != ADD_CALL_RULE:
        raise ValueError(
            "rextio-tensorflow add call lower requires ADD_CALL_RULE"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional add lower forbids receivers")
    if claimed.keywords or len(claimed.operand_types) != 2:
        raise ValueError(
            "rextio-tensorflow add lower requires two positional operands and no keywords"
        )
    left, right = claimed.operand_types[0], claimed.operand_types[1]
    if left is None or right is None:
        raise ValueError("rextio-tensorflow add lower requires resolved operand types")
    pair = (left, right)
    expected = _SUPPORTED.get(pair)
    if expected is None or claimed.result_type != expected:
        raise ValueError(
            "rextio-tensorflow add lower operand/result types changed between claim and lower: "
            f"operands={claimed.operand_types!r} result={claimed.result_type!r}"
        )
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-tensorflow add lower requires two ctx.operands entries; "
            f"got {len(ctx.operands)}"
        )
    a, b = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::add(&{a}, &{b})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
