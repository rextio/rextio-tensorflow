"""Lower the statically-proven TensorFlow classification-head steps."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.classification import (
    ARGMAX_RULE,
    ARGMAX_TARGETS,
    SOFTMAX_RULE,
    SOFTMAX_TARGETS,
)
from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers


def _axis_one(claimed: ClaimSite, operation: str) -> None:
    values = {keyword.name: keyword.literal for keyword in claimed.keywords}
    if len(values) != len(claimed.keywords) or set(values) != {"axis"}:
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires only axis=1 literal"
        )
    axis = values["axis"]
    if (
        not axis.is_literal
        or not isinstance(axis.value, int)
        or isinstance(axis.value, bool)
        or axis.value != 1
    ):
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires axis=1 literal; got {axis.value!r}"
        )


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a classification-head claim after repeating all static guards."""
    if claimed.kind != "call" or claimed.target not in SOFTMAX_TARGETS | ARGMAX_TARGETS:
        return None
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional classification lower forbids receivers")
    if len(claimed.operand_types) != 1 or claimed.operand_types[0] != TENSOR_F32_CPU_2D:
        raise ValueError("rextio-tensorflow classification lower requires rank-2 float32 input")
    if len(ctx.operands) != 1:
        raise ValueError("rextio-tensorflow classification lower requires one ctx.operands entry")
    (x,) = ctx.operands
    if claimed.target in SOFTMAX_TARGETS:
        if claimed.rule_id != SOFTMAX_RULE or claimed.result_type != TENSOR_F32_CPU_2D:
            raise ValueError("rextio-tensorflow received malformed softmax lower metadata")
        _axis_one(claimed, "softmax")
        helper = "softmax_axis1"
    else:
        if claimed.rule_id != ARGMAX_RULE or claimed.result_type != TENSOR_I64_CPU_1D:
            raise ValueError("rextio-tensorflow received malformed argmax lower metadata")
        _axis_one(claimed, "argmax")
        helper = "argmax_axis1"
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{helper}(&{x})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
