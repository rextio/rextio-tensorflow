"""Fail-closed claims for bounded literal-axis TensorFlow reductions."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_MEAN,
    DIAGNOSTIC_SUM,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

MEAN_RULE = "rextio-tensorflow/reduce-mean-axis1-f32-cpu-2d"
MEAN_GENERAL_RULE = "rextio-tensorflow/reduce-mean-literal-axis-f32-cpu-2d"
MEAN_TARGETS = frozenset(
    {
        "tensorflow.reduce_mean",
        "tensorflow.math.reduce_mean",
        "tf.reduce_mean",
        "tf.math.reduce_mean",
    }
)
SUM_RULE = "rextio-tensorflow/reduce-sum-axis1-f32-cpu-2d"
SUM_GENERAL_RULE = "rextio-tensorflow/reduce-sum-literal-axis-f32-cpu-2d"
SUM_TARGETS = frozenset(
    {
        "tensorflow.reduce_sum",
        "tensorflow.math.reduce_sum",
        "tf.reduce_sum",
        "tf.math.reduce_sum",
    }
)


def _keyword_map(site: ClaimSite) -> dict[str, object] | None:
    values: dict[str, object] = {}
    for keyword in site.keywords:
        if not keyword.literal.is_literal:
            return None
        if keyword.name in values:
            return None
        values[keyword.name] = keyword.literal.value
    return values


def _axis_and_keepdims(
    site: ClaimSite, *, diagnostic: str, operation: str
) -> tuple[int, bool] | ClaimResult:
    operands = tuple(site.operand_types)
    keywords = _keyword_map(site)
    if keywords is None:
        return reject(
            site,
            diagnostic,
            f"only static literal keywords are supported for {operation}",
            "Pass axis=0 or axis=1 and optional keepdims as static literals.",
        )
    extra = set(keywords) - {"axis", "keepdims"}
    if extra:
        return reject(
            site,
            diagnostic,
            f"unsupported {operation} keywords {sorted(extra)!r}",
            "Only axis and optional named keepdims are supported.",
        )
    if len(operands) == 1:
        if "axis" not in keywords:
            return reject(
                site,
                diagnostic,
                f"{operation} requires a literal axis=0 or axis=1",
                f"Write tf.{operation}(x, axis=0) or tf.{operation}(x, axis=1).",
            )
        axis_value = keywords["axis"]
    elif len(operands) == 2:
        if "axis" in keywords:
            return reject(
                site,
                diagnostic,
                "axis must not be both positional and keyword",
                "Use either a literal axis keyword or one literal positional axis.",
            )
        if len(site.operand_literals) != 2 or operands[1] != "int":
            return reject(
                site,
                diagnostic,
                "positional axis metadata is not exactly aligned with two operands",
                "Use one statically literal integer positional axis.",
            )
        axis_literal = site.operand_literals[1]
        if not axis_literal.is_literal:
            return reject(
                site,
                diagnostic,
                "positional axis is not statically literal",
                "Use positional literal axis 0 or 1.",
            )
        axis_value = axis_literal.value
    else:
        return reject(
            site,
            diagnostic,
            f"unexpected {operation} arity",
            (
                f"Write tf.{operation}(x, axis=0|1); positional keepdims is "
                "outside the bounded surface."
            ),
        )
    if (
        not isinstance(axis_value, int)
        or isinstance(axis_value, bool)
        or axis_value not in {0, 1}
    ):
        return reject(
            site,
            diagnostic,
            f"bounded {operation} requires axis=0 or axis=1 literal; got axis={axis_value!r}",
            "Use literal axis 0 or 1.",
        )
    keepdims = keywords.get("keepdims", False)
    if not isinstance(keepdims, bool):
        return reject(
            site,
            diagnostic,
            f"bounded {operation} requires literal bool keepdims; got {keepdims!r}",
            "Omit keepdims or pass named keepdims=True/False.",
        )
    return axis_value, keepdims


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim rank-2 reductions with literal axis 0/1 and named keepdims."""
    if site.kind != "call":
        return None
    if site.receiver is not None:
        return NotCovered()
    if site.target in MEAN_TARGETS:
        legacy_rule = MEAN_RULE
        general_rule = MEAN_GENERAL_RULE
        diagnostic = DIAGNOSTIC_MEAN
        operation = "reduce_mean"
    elif site.target in SUM_TARGETS:
        legacy_rule = SUM_RULE
        general_rule = SUM_GENERAL_RULE
        diagnostic = DIAGNOSTIC_SUM
        operation = "reduce_sum"
    else:
        return None
    parsed = _axis_and_keepdims(site, diagnostic=diagnostic, operation=operation)
    if not isinstance(parsed, tuple):
        return parsed
    axis, keepdims = parsed

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
            DIAGNOSTIC_UNSUPPORTED,
            f"Alpha {operation} requires float32 CPU rank-2; got {operand!r}",
            f"Use TensorF32Cpu2D for the {operation} operand.",
        )
    result_type = TENSOR_F32_CPU_2D if keepdims else TENSOR_F32_CPU_1D
    rule_id = legacy_rule if axis == 1 and not keepdims else general_rule
    return Claimed(rule_id=rule_id, result_type=result_type)


__all__ = [
    "MEAN_GENERAL_RULE",
    "MEAN_RULE",
    "MEAN_TARGETS",
    "SUM_GENERAL_RULE",
    "SUM_RULE",
    "SUM_TARGETS",
    "try_claim",
]
