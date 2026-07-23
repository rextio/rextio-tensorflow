"""Lower bounded elementwise binary claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.add import (
    ADD_BINOP_RULE,
    ADD_CALL_RULE,
    ADD_TARGETS,
    BIAS_ADD_RULE,
    BIAS_ADD_TARGETS,
    DIV_BINOP_RULE,
    DIV_CALL_RULE,
    DIV_TARGETS,
    MUL_BINOP_RULE,
    MUL_CALL_RULE,
    MUL_TARGETS,
    MAXIMUM_CALL_RULE,
    MAXIMUM_TARGETS,
    MINIMUM_CALL_RULE,
    MINIMUM_TARGETS,
    SUB_BINOP_RULE,
    SUB_CALL_RULE,
    SUB_TARGETS,
)
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

_SUPPORTED = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}
_SAME_RANK_SUPPORTED = {
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}


def _lower_bias_add(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(
            "rextio-tensorflow functional bias_add lower forbids receivers"
        )
    if (
        claimed.rule_id != BIAS_ADD_RULE
        or claimed.result_type != TENSOR_F32_CPU_2D
        or claimed.operand_types
        != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)
    ):
        raise ValueError(
            "rextio-tensorflow received malformed bias_add lower metadata"
        )
    if claimed.operand_literals and (
        len(claimed.operand_literals) != 2
        or any(literal.is_literal for literal in claimed.operand_literals)
    ):
        raise ValueError(
            "rextio-tensorflow bias_add lower rejects forged positional literal metadata"
        )
    keywords = {keyword.name: keyword for keyword in claimed.keywords}
    if len(keywords) != len(claimed.keywords):
        raise ValueError(
            "rextio-tensorflow bias_add lower rejects duplicate keyword metadata"
        )
    if keywords:
        if set(keywords) != {"data_format"}:
            raise ValueError(
                "rextio-tensorflow bias_add lower accepts only data_format='NHWC'"
            )
        data_format = keywords["data_format"]
        if (
            data_format.arg_type != "str"
            or not data_format.literal.is_literal
            or data_format.literal.value != "NHWC"
        ):
            raise ValueError(
                "rextio-tensorflow bias_add lower requires literal data_format='NHWC'"
            )
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-tensorflow bias_add lower requires two ctx.operands entries"
        )
    value, bias = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::bias_add(&{value}, &{bias})?",
        helpers=(runtime_module_helpers(),),
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed bounded binary site, or return None."""
    if claimed.kind == "call" and claimed.target in BIAS_ADD_TARGETS:
        return _lower_bias_add(claimed, ctx)
    binops = {
        "+": (ADD_BINOP_RULE, "add", "add"),
        "*": (MUL_BINOP_RULE, "multiply", "mul"),
        "-": (SUB_BINOP_RULE, "subtract", "sub"),
        "/": (DIV_BINOP_RULE, "divide", "div"),
    }
    calls = {
        **{target: (ADD_CALL_RULE, "add", "add") for target in ADD_TARGETS},
        **{target: (MUL_CALL_RULE, "multiply", "mul") for target in MUL_TARGETS},
        **{target: (SUB_CALL_RULE, "subtract", "sub") for target in SUB_TARGETS},
        **{target: (DIV_CALL_RULE, "divide", "div") for target in DIV_TARGETS},
        **{
            target: (MAXIMUM_CALL_RULE, "maximum", "maximum")
            for target in MAXIMUM_TARGETS
        },
        **{
            target: (MINIMUM_CALL_RULE, "minimum", "minimum")
            for target in MINIMUM_TARGETS
        },
    }
    if claimed.kind == "binop" and claimed.target in binops:
        expected_rule, operation, helper = binops[claimed.target]
    elif claimed.kind == "call" and claimed.target in calls:
        expected_rule, operation, helper = calls[claimed.target]
    else:
        return None
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
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires resolved operand types"
        )
    pair = (left, right)
    same_rank_operation = expected_rule in {
        MAXIMUM_CALL_RULE,
        MINIMUM_CALL_RULE,
    }
    expected = (
        _SAME_RANK_SUPPORTED.get(pair)
        if same_rank_operation
        else _SUPPORTED.get(pair)
    )
    if expected is None or claimed.result_type != expected:
        raise ValueError(
            f"rextio-tensorflow {operation} lower operand/result types changed between claim and lower: "
            f"operands={claimed.operand_types!r} result={claimed.result_type!r}"
        )
    if same_rank_operation and claimed.operand_literals and (
        len(claimed.operand_literals) != 2
        or any(literal.is_literal for literal in claimed.operand_literals)
    ):
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires non-literal tensor operands"
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
