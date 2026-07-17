"""Lower reduce_mean claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.reductions import MEAN_RULE, MEAN_TARGETS
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed reduce_mean site, or return None."""
    if claimed.kind != "call" or claimed.target not in MEAN_TARGETS:
        return None
    if claimed.rule_id != MEAN_RULE:
        raise ValueError(
            "rextio-tensorflow mean lower received mismatched rule_id: "
            f"{claimed.rule_id!r} != {MEAN_RULE!r}"
        )
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional mean lower forbids receivers")
    if (
        len(claimed.operand_types) != 1
        or claimed.operand_types[0] != TENSOR_F32_CPU_2D
        or claimed.result_type != TENSOR_F32_CPU_1D
    ):
        raise ValueError("rextio-tensorflow received malformed mean lower metadata")
    values = {kw.name: kw.literal for kw in claimed.keywords}
    if "axis" not in values:
        raise ValueError("rextio-tensorflow mean lower requires axis keyword")
    axis_lit = values["axis"]
    if (
        not axis_lit.is_literal
        or not isinstance(axis_lit.value, int)
        or isinstance(axis_lit.value, bool)
        or axis_lit.value != 1
    ):
        raise ValueError(
            f"rextio-tensorflow mean lower requires axis=1 literal; got {axis_lit.value!r}"
        )
    if "keepdims" in values:
        keep = values["keepdims"]
        if not keep.is_literal or keep.value is not False:
            raise ValueError(
                "rextio-tensorflow mean lower requires keepdims=False when present"
            )
    extra = set(values) - {"axis", "keepdims"}
    if extra:
        raise ValueError(f"rextio-tensorflow mean lower unexpected keywords {sorted(extra)!r}")
    if len(ctx.operands) != 1:
        raise ValueError("rextio-tensorflow mean lower requires one ctx.operands entry")
    (x,) = ctx.operands
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::reduce_mean_axis1(&{x})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
