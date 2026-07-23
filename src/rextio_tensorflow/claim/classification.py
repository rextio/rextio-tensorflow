"""Fail-closed claims for the rank-2 TensorFlow classification head."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, KeywordArg, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_ARGMAX,
    DIAGNOSTIC_SOFTMAX,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
    is_tensor_type,
    reject,
)

SOFTMAX_RULE = "rextio-tensorflow/softmax-axis1-f32-cpu-2d"
ARGMAX_RULE = "rextio-tensorflow/argmax-axis1-i64-cpu-2d"
ARGMAX_AXIS0_RULE = "rextio-tensorflow/argmax-axis0-i64-cpu-2d"
SOFTMAX_TARGETS = frozenset({"tensorflow.nn.softmax", "tf.nn.softmax"})
ARGMAX_TARGETS = frozenset({"tensorflow.argmax", "tf.argmax"})


def _literal_keywords(site: ClaimSite) -> dict[str, KeywordArg] | None:
    values: dict[str, KeywordArg] = {}
    for keyword in site.keywords:
        if not keyword.literal.is_literal:
            return None
        if keyword.name in values:
            return None
        values[keyword.name] = keyword
    return values


def _literal_axis(
    site: ClaimSite, *, diagnostic: str, operation: str
) -> int | ClaimResult:
    values = _literal_keywords(site)
    if values is None:
        return reject(
            site,
            diagnostic,
            f"only static literal keywords are supported for {operation}",
            f"Write tf.{operation}(x, axis=0|1) with a literal axis.",
        )
    if len(site.operand_types) == 1:
        if set(values) != {"axis"}:
            return reject(
                site,
                diagnostic,
                f"{operation} accepts only one literal axis keyword",
                "Do not pass keepdims, output_type, name, or other keywords.",
            )
        axis_keyword = values["axis"]
        if axis_keyword.arg_type != "int":
            return reject(
                site,
                diagnostic,
                f"{operation} axis metadata must have arg_type='int'",
                "Use a literal integer axis 0 or 1.",
            )
        axis_literal = axis_keyword.literal
    elif len(site.operand_types) == 2:
        if values:
            return reject(
                site,
                diagnostic,
                f"{operation} positional axis form accepts no keywords",
                "Use one positional literal axis and omit additional keywords.",
            )
        if (
            site.operand_types[1] != "int"
            or len(site.operand_literals) != 2
        ):
            return reject(
                site,
                diagnostic,
                "positional axis metadata is not exactly aligned with two operands",
                "Use one positional literal integer axis.",
            )
        axis_literal = site.operand_literals[1]
        if not axis_literal.is_literal:
            return reject(
                site,
                diagnostic,
                "positional axis is not statically literal",
                "Use positional literal axis 0 or 1.",
            )
    else:
        return reject(
            site,
            diagnostic,
            f"{operation} requires one tensor and one explicit literal axis",
            f"Write tf.{operation}(x, axis=0|1).",
        )
    axis = axis_literal.value
    if not isinstance(axis, int) or isinstance(axis, bool) or axis not in {0, 1}:
        return reject(
            site,
            diagnostic,
            f"bounded {operation} requires axis=0 or axis=1 literal; got axis={axis!r}",
            "Use literal axis 0 or 1.",
        )
    return axis


def _rank2_input(
    site: ClaimSite, *, diagnostic: str, operation: str
) -> ClaimResult | None:
    operand = site.operand_types[0]
    if operand is None:
        return NotCovered()
    if not is_tensor_type(operand):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand type is outside the float32 CPU tensor surface",
            "Annotate the operand as rextio_tensorflow.types.TensorF32Cpu2D.",
        )
    if operand != TENSOR_F32_CPU_2D:
        return reject(
            site,
            diagnostic,
            f"Alpha {operation} requires a float32 CPU rank-2 operand; got {operand!r}",
            "Use TensorF32Cpu2D for the classification-head operand.",
        )
    return None


def _try_softmax(site: ClaimSite) -> ClaimResult:
    axis = _literal_axis(site, diagnostic=DIAGNOSTIC_SOFTMAX, operation="nn.softmax")
    if not isinstance(axis, int):
        return axis
    if axis == 0:
        return reject(
            site,
            DIAGNOSTIC_SOFTMAX,
            (
                "raw TFE Softmax is last-axis-only; axis=0 would require a "
                "transpose outside this bounded surface"
            ),
            "Use literal axis=1 or keep axis=0 softmax on Python fallback.",
        )
    rejected = _rank2_input(site, diagnostic=DIAGNOSTIC_SOFTMAX, operation="softmax")
    if rejected is not None:
        return rejected
    return Claimed(rule_id=SOFTMAX_RULE, result_type=TENSOR_F32_CPU_2D)


def _try_argmax(site: ClaimSite) -> ClaimResult:
    axis = _literal_axis(site, diagnostic=DIAGNOSTIC_ARGMAX, operation="argmax")
    if not isinstance(axis, int):
        return axis
    rejected = _rank2_input(site, diagnostic=DIAGNOSTIC_ARGMAX, operation="argmax")
    if rejected is not None:
        return rejected
    rule_id = ARGMAX_AXIS0_RULE if axis == 0 else ARGMAX_RULE
    return Claimed(rule_id=rule_id, result_type=TENSOR_I64_CPU_1D)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim only literal-axis rank-2 softmax/argmax classification steps."""
    if site.kind != "call":
        return None
    if site.receiver is not None:
        if site.target in SOFTMAX_TARGETS | ARGMAX_TARGETS:
            return NotCovered()
        return None
    if site.target in SOFTMAX_TARGETS:
        return _try_softmax(site)
    if site.target in ARGMAX_TARGETS:
        return _try_argmax(site)
    return None


__all__ = [
    "ARGMAX_AXIS0_RULE",
    "ARGMAX_RULE",
    "ARGMAX_TARGETS",
    "SOFTMAX_RULE",
    "SOFTMAX_TARGETS",
    "try_claim",
]
