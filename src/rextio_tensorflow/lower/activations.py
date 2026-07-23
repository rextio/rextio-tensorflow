"""Lower bounded unary activation claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.activations import (
    RELU_1D_RULE,
    RELU_RULE,
    RELU_TARGETS,
    SIGMOID_1D_RULE,
    SIGMOID_RULE,
    SIGMOID_TARGETS,
    TANH_1D_RULE,
    TANH_RULE,
    TANH_TARGETS,
)
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed unary activation site, or return None."""
    if claimed.kind != "call":
        return None
    if claimed.target in RELU_TARGETS:
        return _unary_lower(
            claimed,
            ctx,
            rank1_rule_id=RELU_1D_RULE,
            rank2_rule_id=RELU_RULE,
            helper="relu",
        )
    if claimed.target in SIGMOID_TARGETS:
        return _unary_lower(
            claimed,
            ctx,
            rank1_rule_id=SIGMOID_1D_RULE,
            rank2_rule_id=SIGMOID_RULE,
            helper="sigmoid",
        )
    if claimed.target in TANH_TARGETS:
        return _unary_lower(
            claimed,
            ctx,
            rank1_rule_id=TANH_1D_RULE,
            rank2_rule_id=TANH_RULE,
            helper="tanh",
        )
    return None


def _unary_lower(
    claimed: ClaimSite,
    ctx: LoweringContext,
    *,
    rank1_rule_id: str,
    rank2_rule_id: str,
    helper: str,
) -> LoweredExpr:
    rules = {
        TENSOR_F32_CPU_1D: rank1_rule_id,
        TENSOR_F32_CPU_2D: rank2_rule_id,
    }
    if len(claimed.operand_types) != 1:
        raise ValueError(f"rextio-tensorflow received malformed {helper} lower metadata")
    operand_type = claimed.operand_types[0]
    expected_rule = rules.get(operand_type)
    if expected_rule is None or claimed.rule_id != expected_rule:
        raise ValueError(
            f"rextio-tensorflow {helper} lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {expected_rule!r}"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(
            f"rextio-tensorflow functional {helper} lower forbids receivers"
        )
    if (
        claimed.keywords
        or claimed.result_type != operand_type
    ):
        raise ValueError(f"rextio-tensorflow received malformed {helper} lower metadata")
    if len(ctx.operands) != 1:
        raise ValueError(
            f"rextio-tensorflow {helper} lower requires one ctx.operands entry"
        )
    (x,) = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{helper}(&{x})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
