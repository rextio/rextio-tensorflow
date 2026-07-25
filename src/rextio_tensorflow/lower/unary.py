"""Lower bounded TensorFlow math unary claims after revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.unary import TARGET_TO_UNARY, UNARY_RULES
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def _literal_metadata_is_aligned(claimed: ClaimSite) -> bool:
    return not claimed.operand_literals or (
        len(claimed.operand_literals) == 1
        and not claimed.operand_literals[0].is_literal
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed exact unary call, or return None."""
    if claimed.kind != "call":
        return None
    operation = TARGET_TO_UNARY.get(claimed.target)
    if operation is None:
        return None
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(
            f"rextio-tensorflow functional {operation} lower forbids receivers"
        )
    if len(claimed.operand_types) != 1:
        raise ValueError(
            f"rextio-tensorflow received malformed {operation} lower metadata"
        )
    operand_type = claimed.operand_types[0]
    if (
        operand_type not in {TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D}
        or claimed.rule_id != UNARY_RULES[operation]
        or claimed.result_type != operand_type
        or claimed.keywords
        or not _literal_metadata_is_aligned(claimed)
    ):
        raise ValueError(
            f"rextio-tensorflow received malformed {operation} lower metadata"
        )
    if len(ctx.operands) != 1:
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires one ctx.operands entry"
        )
    (operand,) = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{operation}(&{operand})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
