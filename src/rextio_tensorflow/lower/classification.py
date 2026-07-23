"""Lower the statically-proven TensorFlow classification-head steps."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.classification import (
    ARGMAX_AXIS0_RULE,
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


def _literal_axis(claimed: ClaimSite, operation: str) -> int:
    values = {keyword.name: keyword for keyword in claimed.keywords}
    if len(values) != len(claimed.keywords):
        raise ValueError(
            f"rextio-tensorflow {operation} lower rejects duplicate axis/keyword metadata"
        )
    if len(claimed.operand_types) == 1:
        if set(values) != {"axis"}:
            raise ValueError(
                f"rextio-tensorflow {operation} lower requires only one literal axis keyword"
            )
        axis_keyword = values["axis"]
        if axis_keyword.arg_type != "int":
            raise ValueError(
                f"rextio-tensorflow {operation} lower axis keyword requires arg_type='int'"
            )
        axis = axis_keyword.literal
    elif len(claimed.operand_types) == 2:
        if values:
            raise ValueError(
                f"rextio-tensorflow {operation} lower positional axis accepts no keywords"
            )
        if (
            claimed.operand_types[1] != "int"
            or len(claimed.operand_literals) != 2
        ):
            raise ValueError(
                f"rextio-tensorflow {operation} lower positional axis metadata is not aligned"
            )
        axis = claimed.operand_literals[1]
    else:
        raise ValueError(
            f"rextio-tensorflow {operation} lower positional axis arity is invalid"
        )
    if (
        not axis.is_literal
        or not isinstance(axis.value, int)
        or isinstance(axis.value, bool)
        or axis.value not in {0, 1}
    ):
        raise ValueError(
            f"rextio-tensorflow {operation} lower requires axis=0 or axis=1 literal; "
            f"got {axis.value!r}"
        )
    return axis.value


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower a classification-head claim after repeating all static guards."""
    if claimed.kind != "call" or claimed.target not in SOFTMAX_TARGETS | ARGMAX_TARGETS:
        return None
    if claimed.receiver is not None or ctx.receiver is not None:
        raise ValueError("rextio-tensorflow functional classification lower forbids receivers")
    if not claimed.operand_types or claimed.operand_types[0] != TENSOR_F32_CPU_2D:
        raise ValueError("rextio-tensorflow classification lower requires rank-2 float32 input")
    if len(ctx.operands) != len(claimed.operand_types):
        raise ValueError(
            "rextio-tensorflow classification lower requires one rendered operand "
            "per claimed positional argument; the literal axis slot is validated "
            "but not emitted as a TFE input"
        )
    x = ctx.operands[0]
    axis = _literal_axis(claimed, "classification")
    if claimed.target in SOFTMAX_TARGETS:
        if claimed.rule_id != SOFTMAX_RULE or claimed.result_type != TENSOR_F32_CPU_2D:
            raise ValueError("rextio-tensorflow received malformed softmax lower metadata")
        if axis != 1:
            raise ValueError(
                "rextio-tensorflow softmax lower supports only final rank-2 axis=1"
            )
        helper = "softmax_axis1"
    else:
        expected_rule = ARGMAX_AXIS0_RULE if axis == 0 else ARGMAX_RULE
        if claimed.rule_id != expected_rule or claimed.result_type != TENSOR_I64_CPU_1D:
            raise ValueError("rextio-tensorflow received malformed argmax lower metadata")
        helper = f"argmax_axis{axis}"
    return LoweredExpr(
        rust=f"rextio_tensorflow_runtime::{helper}(&{x})?",
        helpers=(runtime_module_helpers(),),
    )


__all__ = ["try_lower"]
