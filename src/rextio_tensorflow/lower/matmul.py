"""Lower matmul claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.matmul import MATMUL_RULE, MATMUL_TARGETS
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed matmul site, or return None."""
    if claimed.kind != "call" or claimed.target not in MATMUL_TARGETS:
        return None
    if claimed.rule_id != MATMUL_RULE:
        raise ValueError(
            "rextio-tensorflow matmul lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {MATMUL_RULE!r}"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional matmul lower forbids receivers")
    if claimed.keywords or len(claimed.operand_types) != 2:
        raise ValueError(
            "rextio-tensorflow matmul lower requires two positional operands and no keywords"
        )
    if tuple(claimed.operand_types) != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D):
        raise ValueError(
            "rextio-tensorflow matmul lower operand types changed between claim and lower: "
            f"{claimed.operand_types!r}"
        )
    if claimed.result_type != TENSOR_F32_CPU_2D:
        raise ValueError(
            "rextio-tensorflow matmul lower result type changed between claim and lower: "
            f"{claimed.result_type!r}"
        )
    if len(ctx.operands) != 2:
        raise ValueError(
            "rextio-tensorflow matmul lower requires two ctx.operands entries; "
            f"got {len(ctx.operands)}"
        )
    a, b = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::matmul(&{a}, &{b})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
