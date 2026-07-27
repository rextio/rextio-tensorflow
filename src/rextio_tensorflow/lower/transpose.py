"""Lower default rank-2 float32 CPU ``tf.transpose`` after revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.transpose import TRANSPOSE_RULE, TRANSPOSE_TARGETS
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def _literal_metadata_is_aligned(claimed: ClaimSite) -> bool:
    return not claimed.operand_literals or (
        len(claimed.operand_literals) == 1
        and not claimed.operand_literals[0].is_literal
    )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed default transpose site, or return None."""
    if claimed.kind != "call":
        return None
    if claimed.target not in TRANSPOSE_TARGETS:
        return None
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional transpose lower forbids receivers")
    if (
        len(claimed.operand_types) != 1
        or claimed.operand_types[0] != TENSOR_F32_CPU_2D
        or claimed.rule_id != TRANSPOSE_RULE
        or claimed.result_type != TENSOR_F32_CPU_2D
        or claimed.keywords
        or not _literal_metadata_is_aligned(claimed)
    ):
        raise ValueError("rextio-tensorflow received malformed transpose lower metadata")
    if len(ctx.operands) != 1:
        raise ValueError(
            "rextio-tensorflow transpose lower requires one ctx.operands entry"
        )
    (operand,) = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::transpose(&{operand})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
