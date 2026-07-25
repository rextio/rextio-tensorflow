"""Fail-closed claims for the bounded elementwise binary surface."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ADD,
    DIAGNOSTIC_ADD_BINOP,
    DIAGNOSTIC_BIAS_ADD,
    DIAGNOSTIC_DIV_BINOP,
    DIAGNOSTIC_DIV_CALL,
    DIAGNOSTIC_MAXIMUM,
    DIAGNOSTIC_MINIMUM,
    DIAGNOSTIC_MUL_BINOP,
    DIAGNOSTIC_MUL_CALL,
    DIAGNOSTIC_SUB_BINOP,
    DIAGNOSTIC_SUB_CALL,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

ADD_CALL_RULE = "rextio-tensorflow/add-call-f32-cpu"
ADD_BINOP_RULE = "rextio-tensorflow/add-binop-f32-cpu"
MUL_BINOP_RULE = "rextio-tensorflow/mul-binop-f32-cpu"
MUL_CALL_RULE = "rextio-tensorflow/mul-call-f32-cpu"
SUB_CALL_RULE = "rextio-tensorflow/sub-call-f32-cpu"
SUB_BINOP_RULE = "rextio-tensorflow/sub-binop-f32-cpu"
DIV_CALL_RULE = "rextio-tensorflow/div-call-f32-cpu"
DIV_BINOP_RULE = "rextio-tensorflow/div-binop-f32-cpu"
BIAS_ADD_RULE = "rextio-tensorflow/bias-add-nhwc-f32-cpu-2d"
MAXIMUM_CALL_RULE = "rextio-tensorflow/maximum-call-f32-cpu"
MINIMUM_CALL_RULE = "rextio-tensorflow/minimum-call-f32-cpu"
# Back-compat alias used by older tests / docs references.
ADD_RULE = ADD_CALL_RULE
ADD_TARGETS = frozenset(
    {
        "tensorflow.add",
        "tensorflow.math.add",
        "tf.add",
        "tf.math.add",
    }
)
MUL_TARGETS = frozenset(
    {
        "tensorflow.multiply",
        "tensorflow.math.multiply",
        "tf.multiply",
        "tf.math.multiply",
    }
)
SUB_TARGETS = frozenset(
    {
        "tensorflow.subtract",
        "tensorflow.math.subtract",
        "tf.subtract",
        "tf.math.subtract",
    }
)
DIV_TARGETS = frozenset(
    {
        "tensorflow.divide",
        "tensorflow.math.divide",
        "tf.divide",
        "tf.math.divide",
    }
)
BIAS_ADD_TARGETS = frozenset({"tensorflow.nn.bias_add", "tf.nn.bias_add"})
MAXIMUM_TARGETS = frozenset({"tensorflow.maximum", "tf.maximum"})
MINIMUM_TARGETS = frozenset({"tensorflow.minimum", "tf.minimum"})

_SUPPORTED_PAIRS: dict[tuple[str, str], str] = {
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_2D,
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}
_SAME_RANK_PAIRS: dict[tuple[str, str], str] = {
    (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D): TENSOR_F32_CPU_1D,
    (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D): TENSOR_F32_CPU_2D,
}


def _claim_bias_add(site: ClaimSite) -> ClaimResult:
    if site.receiver is not None:
        return NotCovered()
    if len(site.operand_types) != 2:
        return reject(
            site,
            DIAGNOSTIC_BIAS_ADD,
            "bounded bias_add requires value and bias as two positional tensors",
            "Call tf.nn.bias_add(value, bias) with rank-2 value and rank-1 bias.",
        )
    if site.operand_literals and (
        len(site.operand_literals) != 2
        or any(literal.is_literal for literal in site.operand_literals)
    ):
        return reject(
            site,
            DIAGNOSTIC_BIAS_ADD,
            "bias_add received forged positional literal metadata",
            "Pass value and bias as positional tensors.",
        )
    keywords = {keyword.name: keyword for keyword in site.keywords}
    if len(keywords) != len(site.keywords):
        return reject(
            site,
            DIAGNOSTIC_BIAS_ADD,
            "bias_add received duplicate keyword metadata",
            "Omit data_format or pass data_format='NHWC' exactly once.",
        )
    if keywords:
        if set(keywords) != {"data_format"}:
            return reject(
                site,
                DIAGNOSTIC_BIAS_ADD,
                "bounded bias_add accepts only the literal data_format='NHWC' option",
                "Do not pass name or tensor operands by keyword.",
            )
        data_format = keywords["data_format"]
        if (
            data_format.arg_type != "str"
            or not data_format.literal.is_literal
            or data_format.literal.value != "NHWC"
        ):
            return reject(
                site,
                DIAGNOSTIC_BIAS_ADD,
                "bounded bias_add requires static literal data_format='NHWC'",
                "Omit data_format or write data_format='NHWC'.",
            )
    value_type, bias_type = site.operand_types
    if value_type is None or bias_type is None:
        return NotCovered()
    if not is_tensor_type(value_type) or not is_tensor_type(bias_type):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "bias_add operand types are outside the float32 CPU tensor surface",
            "Annotate value as TensorF32Cpu2D and bias as TensorF32Cpu1D.",
        )
    if (value_type, bias_type) != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D):
        return reject(
            site,
            DIAGNOSTIC_BIAS_ADD,
            (
                "bounded NHWC bias_add requires rank-2 float32 CPU value followed "
                f"by rank-1 float32 CPU bias; got {site.operand_types!r}"
            ),
            "Use TensorF32Cpu2D value and TensorF32Cpu1D bias in that order.",
        )
    return Claimed(rule_id=BIAS_ADD_RULE, result_type=TENSOR_F32_CPU_2D)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim exact bounded binary spellings on the float32 CPU surface."""
    binops = {
        "+": (ADD_BINOP_RULE, DIAGNOSTIC_ADD_BINOP, "add", "x + y"),
        "*": (MUL_BINOP_RULE, DIAGNOSTIC_MUL_BINOP, "multiply", "x * y"),
        "-": (SUB_BINOP_RULE, DIAGNOSTIC_SUB_BINOP, "subtract", "x - y"),
        "/": (DIV_BINOP_RULE, DIAGNOSTIC_DIV_BINOP, "divide", "x / y"),
    }
    if site.kind == "binop" and site.target in binops:
        rule_id, diagnostic, operation, syntax = binops[site.target]
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=rule_id,
            diagnostic=diagnostic,
            operation=operation,
            syntax=syntax,
        )
    if site.kind != "call":
        return None
    if site.target in BIAS_ADD_TARGETS:
        return _claim_bias_add(site)
    calls = {
        **{
            target: (
                ADD_CALL_RULE,
                DIAGNOSTIC_ADD,
                "add",
                "tf.add(x, y)",
            )
            for target in ADD_TARGETS
        },
        **{
            target: (
                MUL_CALL_RULE,
                DIAGNOSTIC_MUL_CALL,
                "multiply",
                "tf.multiply(x, y)",
            )
            for target in MUL_TARGETS
        },
        **{
            target: (
                SUB_CALL_RULE,
                DIAGNOSTIC_SUB_CALL,
                "subtract",
                "tf.subtract(x, y)",
            )
            for target in SUB_TARGETS
        },
        **{
            target: (
                DIV_CALL_RULE,
                DIAGNOSTIC_DIV_CALL,
                "divide",
                "tf.divide(x, y)",
            )
            for target in DIV_TARGETS
        },
        **{
            target: (
                MAXIMUM_CALL_RULE,
                DIAGNOSTIC_MAXIMUM,
                "maximum",
                "tf.maximum(x, y)",
            )
            for target in MAXIMUM_TARGETS
        },
        **{
            target: (
                MINIMUM_CALL_RULE,
                DIAGNOSTIC_MINIMUM,
                "minimum",
                "tf.minimum(x, y)",
            )
            for target in MINIMUM_TARGETS
        },
    }
    if site.target in calls:
        if site.receiver is not None:
            return NotCovered()
        rule_id, diagnostic, operation, syntax = calls[site.target]
        if site.keywords:
            return reject(
                site,
                diagnostic,
                f"keywords on {operation} are not supported",
                f"Call {syntax} with two positional tensors only.",
            )
        if site.target in MAXIMUM_TARGETS or site.target in MINIMUM_TARGETS:
            return _claim_same_rank_operands(
                site,
                tuple(site.operand_types),
                rule_id=rule_id,
                diagnostic=diagnostic,
                operation=operation,
                syntax=syntax,
            )
        return _claim_operands(
            site,
            tuple(site.operand_types),
            rule_id=rule_id,
            diagnostic=diagnostic,
            operation=operation,
            syntax=syntax,
        )
    return None


def _claim_same_rank_operands(
    site: ClaimSite,
    operands: tuple[str | None, ...],
    *,
    rule_id: str,
    diagnostic: str,
    operation: str,
    syntax: str,
) -> ClaimResult:
    if len(operands) != 2:
        return reject(
            site,
            diagnostic,
            f"{operation} requires exactly two positional tensor operands",
            f"Write {syntax} with two same-rank tensors.",
        )
    if site.operand_literals and (
        len(site.operand_literals) != 2
        or any(literal.is_literal for literal in site.operand_literals)
    ):
        return reject(
            site,
            diagnostic,
            f"{operation} requires non-literal tensor operands",
            f"Write {syntax} with two annotated tensors.",
        )
    left, right = operands
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU tensor surface",
            "Annotate both operands as TensorF32Cpu1D or TensorF32Cpu2D.",
        )
    result = _SAME_RANK_PAIRS.get((left, right))
    if result is None:
        return reject(
            site,
            diagnostic,
            f"unsupported {operation} operand pair {operands!r}",
            "Use two rank-1 tensors or two rank-2 tensors with compatible shapes.",
        )
    return Claimed(rule_id=rule_id, result_type=result)


def _claim_operands(
    site: ClaimSite,
    operands: tuple[str | None, ...],
    *,
    rule_id: str,
    diagnostic: str,
    operation: str,
    syntax: str,
) -> ClaimResult:
    if len(operands) != 2:
        return reject(
            site,
            diagnostic,
            f"{operation} requires exactly two operands",
            f"Write {syntax} with two tensors.",
        )
    left, right = operands
    if left is None or right is None:
        return NotCovered()
    if not is_tensor_type(left) or not is_tensor_type(right):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU tensor surface",
            "Annotate operands with rextio_tensorflow.types.TensorF32Cpu2D / TensorF32Cpu1D.",
        )
    result = _SUPPORTED_PAIRS.get((left, right))
    if result is None:
        return reject(
            site,
            diagnostic,
            f"unsupported {operation} operand pair {operands!r}",
            "Use same-rank tensors or rank-2/rank-1 trailing broadcast.",
        )
    return Claimed(rule_id=rule_id, result_type=result)


__all__ = [
    "ADD_BINOP_RULE",
    "ADD_CALL_RULE",
    "ADD_RULE",
    "ADD_TARGETS",
    "BIAS_ADD_RULE",
    "BIAS_ADD_TARGETS",
    "DIV_BINOP_RULE",
    "DIV_CALL_RULE",
    "DIV_TARGETS",
    "MUL_BINOP_RULE",
    "MUL_CALL_RULE",
    "MUL_TARGETS",
    "MAXIMUM_CALL_RULE",
    "MAXIMUM_TARGETS",
    "MINIMUM_CALL_RULE",
    "MINIMUM_TARGETS",
    "SUB_BINOP_RULE",
    "SUB_CALL_RULE",
    "SUB_TARGETS",
    "try_claim",
]
