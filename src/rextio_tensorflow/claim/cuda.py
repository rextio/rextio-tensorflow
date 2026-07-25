"""Fail-closed claims for the bounded TensorFlow CUDA E3 build candidate."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_CUDA_BIAS_ADD,
    DIAGNOSTIC_CUDA_E3,
    DIAGNOSTIC_CUDA_MATMUL,
    DIAGNOSTIC_CUDA_MEAN,
    DIAGNOSTIC_CUDA_RELU,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
    reject,
)

CUDA_MATMUL_RULE = "rextio-tensorflow/cuda0-matmul-f32-2d"
CUDA_BIAS_ADD_RULE = "rextio-tensorflow/cuda0-bias-add-nhwc-f32-2d-1d"
CUDA_RELU_RULE = "rextio-tensorflow/cuda0-relu-f32-2d"
CUDA_MEAN_AXIS1_RULE = "rextio-tensorflow/cuda0-reduce-mean-axis1-f32-2d"

CUDA_RULES = frozenset(
    {CUDA_MATMUL_RULE, CUDA_BIAS_ADD_RULE, CUDA_RELU_RULE, CUDA_MEAN_AXIS1_RULE}
)
CUDA_TYPES = frozenset({TENSOR_F32_CUDA0_1D, TENSOR_F32_CUDA0_2D})

MATMUL_TARGETS = frozenset({"tensorflow.matmul", "tf.matmul"})
BIAS_ADD_TARGETS = frozenset({"tensorflow.nn.bias_add", "tf.nn.bias_add"})
RELU_TARGETS = frozenset({"tensorflow.nn.relu", "tf.nn.relu"})
MEAN_TARGETS = frozenset({"tensorflow.reduce_mean", "tf.reduce_mean"})


def _has_cuda_type(site: ClaimSite) -> bool:
    return any(item in CUDA_TYPES for item in site.operand_types) or (
        site.receiver is not None and site.receiver.arg_type in CUDA_TYPES
    )


def _tensor_operands_are_nonliteral(site: ClaimSite, count: int) -> bool:
    return not site.operand_literals or (
        len(site.operand_literals) == count
        and not any(item.is_literal for item in site.operand_literals)
    )


def _literal_keywords(site: ClaimSite) -> dict[str, object] | None:
    values: dict[str, object] = {}
    expected_types = {"data_format": "str", "axis": "int", "keepdims": "bool"}
    for keyword in site.keywords:
        if (
            keyword.name in values
            or keyword.name not in expected_types
            or keyword.arg_type != expected_types[keyword.name]
            or not keyword.literal.is_literal
        ):
            return None
        values[keyword.name] = keyword.literal.value
    return values


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim only the exact four-op CUDA E3 functional slice."""
    if not _has_cuda_type(site):
        return None

    if site.kind == "call" and site.target in MATMUL_TARGETS:
        if (
            site.receiver is None
            and not site.keywords
            and site.operand_types
            == (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)
            and _tensor_operands_are_nonliteral(site, 2)
        ):
            return Claimed(CUDA_MATMUL_RULE, TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_MATMUL,
            "CUDA matmul requires two positional cuda:0 float32 rank-2 tensors",
            "Use tf.matmul(x, w) with TensorF32Cuda0_2D operands and no options.",
        )

    if site.kind == "call" and site.target in BIAS_ADD_TARGETS:
        keywords = _literal_keywords(site)
        if (
            site.receiver is None
            and site.operand_types
            == (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D)
            and _tensor_operands_are_nonliteral(site, 2)
            and keywords is not None
            and set(keywords) <= {"data_format"}
            and keywords.get("data_format", "NHWC") == "NHWC"
        ):
            return Claimed(CUDA_BIAS_ADD_RULE, TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_BIAS_ADD,
            "CUDA bias_add requires rank-2 value, rank-1 bias, and default/NHWC format",
            "Use tf.nn.bias_add(value, bias) or literal data_format='NHWC'.",
        )

    if site.kind == "call" and site.target in RELU_TARGETS:
        if (
            site.receiver is None
            and not site.keywords
            and site.operand_types == (TENSOR_F32_CUDA0_2D,)
            and _tensor_operands_are_nonliteral(site, 1)
        ):
            return Claimed(CUDA_RELU_RULE, TENSOR_F32_CUDA0_2D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_RELU,
            "CUDA ReLU requires one positional cuda:0 float32 rank-2 tensor",
            "Use tf.nn.relu(x) with TensorF32Cuda0_2D.",
        )

    if site.kind == "call" and site.target in MEAN_TARGETS:
        keywords = _literal_keywords(site)
        if (
            site.receiver is None
            and site.operand_types == (TENSOR_F32_CUDA0_2D,)
            and _tensor_operands_are_nonliteral(site, 1)
            and keywords is not None
            and set(keywords) in ({"axis"}, {"axis", "keepdims"})
            and type(keywords.get("axis")) is int
            and keywords.get("axis") == 1
            and type(keywords.get("keepdims", False)) is bool
            and keywords.get("keepdims", False) is False
        ):
            return Claimed(CUDA_MEAN_AXIS1_RULE, TENSOR_F32_CUDA0_1D)
        return reject(
            site,
            DIAGNOSTIC_CUDA_MEAN,
            "CUDA reduce_mean requires literal axis=1 and omitted/False keepdims",
            "Use tf.reduce_mean(x, axis=1) on TensorF32Cuda0_2D.",
        )

    if site.kind in {"call", "binop"}:
        return reject(
            site,
            DIAGNOSTIC_CUDA_E3,
            "operation is outside the build-only CUDA E3 vertical slice",
            "Use only tf.matmul, tf.nn.bias_add, tf.nn.relu, and tf.reduce_mean(axis=1).",
        )
    return NotCovered()


__all__ = [
    "CUDA_BIAS_ADD_RULE",
    "CUDA_MATMUL_RULE",
    "CUDA_MEAN_AXIS1_RULE",
    "CUDA_RELU_RULE",
    "CUDA_RULES",
    "CUDA_TYPES",
    "try_claim",
]
