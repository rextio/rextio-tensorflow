"""Lower literal-axis reduction claims after defensive revalidation."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.reductions import (
    MEAN_GENERAL_RULE,
    MEAN_RULE,
    MEAN_TARGETS,
    SUM_GENERAL_RULE,
    SUM_RULE,
    SUM_TARGETS,
)
from rextio_tensorflow.diagnostics import TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a previously claimed supported reduction site, or return None."""
    if claimed.kind != "call":
        return None
    if claimed.target in MEAN_TARGETS:
        legacy_rule = MEAN_RULE
        general_rule = MEAN_GENERAL_RULE
        operation = "mean"
        helper_prefix = "reduce_mean"
    elif claimed.target in SUM_TARGETS:
        legacy_rule = SUM_RULE
        general_rule = SUM_GENERAL_RULE
        operation = "sum"
        helper_prefix = "reduce_sum"
    else:
        return None
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError(f"rextio-tensorflow functional {operation} lower forbids receivers")
    if not claimed.operand_types or claimed.operand_types[0] != TENSOR_F32_CPU_2D:
        raise ValueError(f"rextio-tensorflow received malformed {operation} lower metadata")
    values = {kw.name: kw.literal for kw in claimed.keywords}
    if len(values) != len(claimed.keywords):
        raise ValueError(
            f"rextio-tensorflow {operation} lower rejects duplicate axis/keyword metadata"
        )
    extra = set(values) - {"axis", "keepdims"}
    if extra:
        raise ValueError(
            f"rextio-tensorflow {operation} lower unexpected keywords {sorted(extra)!r}"
        )
    if len(claimed.operand_types) == 1:
        if "axis" not in values:
            raise ValueError(f"rextio-tensorflow {operation} lower requires axis metadata")
        axis_lit = values["axis"]
    elif len(claimed.operand_types) == 2:
        if "axis" in values:
            raise ValueError(
                f"rextio-tensorflow {operation} lower rejects positional/keyword duplicate axis"
            )
        if (
            claimed.operand_types[1] != "int"
            or len(claimed.operand_literals) != 2
        ):
            raise ValueError(
                f"rextio-tensorflow {operation} lower positional axis metadata is not aligned"
            )
        axis_lit = claimed.operand_literals[1]
    else:
        raise ValueError(
            f"rextio-tensorflow {operation} lower positional axis arity is invalid"
        )
    if (
        not axis_lit.is_literal
        or not isinstance(axis_lit.value, int)
        or isinstance(axis_lit.value, bool)
        or axis_lit.value not in {0, 1}
    ):
        axis_source = (
            "positional axis"
            if len(claimed.operand_types) == 2
            else "axis keyword"
        )
        raise ValueError(
            f"rextio-tensorflow {operation} lower {axis_source} requires "
            f"axis=0 or axis=1 literal; got {axis_lit.value!r}"
        )
    axis = axis_lit.value
    keepdims = False
    if "keepdims" in values:
        keep = values["keepdims"]
        if not keep.is_literal or not isinstance(keep.value, bool):
            raise ValueError(
                f"rextio-tensorflow {operation} lower requires literal bool keepdims"
            )
        keepdims = keep.value
    expected_result = TENSOR_F32_CPU_2D if keepdims else TENSOR_F32_CPU_1D
    expected_rule = legacy_rule if axis == 1 and not keepdims else general_rule
    if claimed.rule_id != expected_rule or claimed.result_type != expected_result:
        raise ValueError(
            f"rextio-tensorflow {operation} lower rule/result metadata changed after claim"
        )
    if len(ctx.operands) not in {1, len(claimed.operand_types)}:
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires the tensor operand; "
            "a positional literal axis is metadata-only"
        )
    x = ctx.operands[0]
    helper = f"{helper_prefix}_axis{axis}"
    if keepdims:
        helper += "_keepdims"
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{helper}(&{x})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
