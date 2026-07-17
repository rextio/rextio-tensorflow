"""Lower relu/sigmoid claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.activations import (
    RELU_RULE,
    RELU_TARGETS,
    SIGMOID_RULE,
    SIGMOID_TARGETS,
)
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed relu/sigmoid site, or return None."""
    if claimed.kind != "call":
        return None
    if claimed.target in RELU_TARGETS:
        return _unary_lower(claimed, ctx, rule_id=RELU_RULE, helper="relu")
    if claimed.target in SIGMOID_TARGETS:
        return _unary_lower(claimed, ctx, rule_id=SIGMOID_RULE, helper="sigmoid")
    return None


def _unary_lower(
    claimed: ClaimSite,
    ctx: LoweringContext,
    *,
    rule_id: str,
    helper: str,
) -> LoweredExpr:
    if claimed.rule_id != rule_id:
        raise ValueError(
            f"rextio-tensorflow {helper} lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {rule_id!r}"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(
            f"rextio-tensorflow functional {helper} lower forbids receivers"
        )
    if (
        claimed.keywords
        or len(claimed.operand_types) != 1
        or claimed.operand_types[0] != TENSOR_F32_CPU_2D
        or claimed.result_type != TENSOR_F32_CPU_2D
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
