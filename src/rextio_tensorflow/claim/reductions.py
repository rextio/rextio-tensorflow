"""Fail-closed claims for literal-axis TensorFlow rank-2 reductions."""

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
MEAN_TARGETS = frozenset(
    {
        "tensorflow.reduce_mean",
        "tensorflow.math.reduce_mean",
        "tf.reduce_mean",
        "tf.math.reduce_mean",
    }
)
SUM_RULE = "rextio-tensorflow/reduce-sum-axis1-f32-cpu-2d"
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


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim supported reductions with static axis=1 on rank-2 float32 CPU."""
    if site.kind != "call":
        return None
    if site.receiver is not None:
        return NotCovered()
    if site.target in MEAN_TARGETS:
        rule_id = MEAN_RULE
        diagnostic = DIAGNOSTIC_MEAN
        operation = "reduce_mean"
    elif site.target in SUM_TARGETS:
        rule_id = SUM_RULE
        diagnostic = DIAGNOSTIC_SUM
        operation = "reduce_sum"
    else:
        return None

    operands = tuple(site.operand_types)
    keywords = _keyword_map(site)
    if keywords is None:
        return reject(
            site,
            diagnostic,
            f"only static literal keywords are supported for {operation}",
            "Pass axis=1 as a static literal keyword.",
        )

    # Static proof is limited to ``operation(x, axis=1[, keepdims=False])``.
    axis_value: object | None = None
    if len(operands) == 1:
        if "axis" not in keywords:
            return reject(
                site,
                diagnostic,
                f"{operation} requires axis=1",
                f"Write tf.{operation}(x, axis=1).",
            )
        axis_value = keywords["axis"]
        extra = set(keywords) - {"axis", "keepdims"}
        if extra:
            return reject(
                site,
                diagnostic,
                f"unsupported {operation} keywords {sorted(extra)!r}",
                "Only axis and optional keepdims=False are supported.",
            )
    elif len(operands) == 2:
        # Second positional is axis; only accept when axis keyword absent.
        if "axis" in keywords:
            return reject(
                site,
                diagnostic,
                "axis must not be both positional and keyword",
                "Use either axis=1 keyword or a single positional axis.",
            )
        # Static literal axis is not in operand_types (types only). Without
        # callable schema, require keyword form for static proof.
        return reject(
            site,
            diagnostic,
            "positional axis is not statically proven on the Alpha surface",
            f"Write tf.{operation}(x, axis=1) with a literal keyword.",
        )
    else:
        return reject(
            site,
            diagnostic,
            f"unexpected {operation} arity",
            f"Write tf.{operation}(x, axis=1).",
        )

    if not isinstance(axis_value, int) or isinstance(axis_value, bool) or axis_value != 1:
        return reject(
            site,
            diagnostic,
            f"Alpha {operation} requires axis=1 literal; got axis={axis_value!r}",
            "Use the literal keyword axis=1.",
        )
    if "keepdims" in keywords and keywords["keepdims"] is not False:
        return reject(
            site,
            diagnostic,
            f"Alpha {operation} requires keepdims=False or omitted; got {keywords['keepdims']!r}",
            "Omit keepdims or pass keepdims=False.",
        )

    (operand,) = operands
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
    return Claimed(rule_id=rule_id, result_type=TENSOR_F32_CPU_1D)


__all__ = ["MEAN_RULE", "MEAN_TARGETS", "SUM_RULE", "SUM_TARGETS", "try_claim"]
