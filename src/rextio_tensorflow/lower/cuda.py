"""Authorization-bound lowering for the TensorFlow CUDA E3 build candidate."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.claim.cuda import (
    BIAS_ADD_TARGETS,
    CUDA_BIAS_ADD_RULE,
    CUDA_MATMUL_RULE,
    CUDA_MEAN_AXIS1_RULE,
    CUDA_RELU_RULE,
    CUDA_RULES,
    MATMUL_TARGETS,
    MEAN_TARGETS,
    RELU_TARGETS,
)
from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
)
from rextio_tensorflow.plugin_types import plugin_type
from rextio_tensorflow.rust_snippets.cuda_runtime import cuda_runtime_module_helpers

CUDA_PROVIDER_ID = "rextio-device-cuda"
CUDA_CAPABILITY_ID = "cuda-tensorflow-tfe-linux-x86_64"


def _require_authorization(ctx: LoweringContext, result_type: str) -> None:
    authorization = ctx.device_authorization
    metadata = plugin_type(result_type).device_value_metadata
    if (
        ctx.backend != "pyo3"
        or authorization is None
        or authorization.provider_id != CUDA_PROVIDER_ID
        or authorization.capability_id != CUDA_CAPABILITY_ID
        or metadata is None
        or not authorization.authorizes(metadata)
    ):
        raise ValueError(
            "rextio-tensorflow CUDA lowering requires exact PyO3 authorization "
            "from rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64"
        )


def _nonliteral_operands(claimed: ClaimSite, count: int) -> bool:
    return not claimed.operand_literals or (
        len(claimed.operand_literals) == count
        and not any(item.is_literal for item in claimed.operand_literals)
    )


def _keywords(claimed: ClaimSite) -> dict[str, object]:
    values: dict[str, object] = {}
    expected_types = {"data_format": "str", "axis": "int", "keepdims": "bool"}
    for keyword in claimed.keywords:
        if (
            keyword.name in values
            or keyword.name not in expected_types
            or keyword.arg_type != expected_types[keyword.name]
            or not keyword.literal.is_literal
        ):
            raise ValueError("CUDA lowering requires unique literal keywords")
        values[keyword.name] = keyword.literal.value
    return values


def try_lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr | None:
    """Lower one exact CUDA rule after independent metadata revalidation."""
    if claimed.rule_id not in CUDA_RULES:
        return None
    _require_authorization(ctx, claimed.result_type or "")
    helper = cuda_runtime_module_helpers()

    if claimed.rule_id == CUDA_MATMUL_RULE:
        if (
            claimed.kind != "call"
            or claimed.target not in MATMUL_TARGETS
            or claimed.receiver is not None
            or ctx.receiver is not None
            or claimed.keywords
            or claimed.operand_types
            != (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)
            or not _nonliteral_operands(claimed, 2)
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or len(ctx.operands) != 2
        ):
            raise ValueError("TensorFlow CUDA matmul metadata changed after claim")
        left, right = ctx.operands
        return LoweredExpr(
            f"rextio_tensorflow_cuda_runtime::matmul(&{left}, &{right})?",
            helpers=(helper,),
        )

    if claimed.rule_id == CUDA_BIAS_ADD_RULE:
        values = _keywords(claimed)
        if (
            claimed.kind != "call"
            or claimed.target not in BIAS_ADD_TARGETS
            or claimed.receiver is not None
            or ctx.receiver is not None
            or claimed.operand_types
            != (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D)
            or not _nonliteral_operands(claimed, 2)
            or set(values) not in (set(), {"data_format"})
            or values.get("data_format", "NHWC") != "NHWC"
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or len(ctx.operands) != 2
        ):
            raise ValueError("TensorFlow CUDA bias_add metadata changed after claim")
        value, bias = ctx.operands
        return LoweredExpr(
            f"rextio_tensorflow_cuda_runtime::bias_add(&{value}, &{bias})?",
            helpers=(helper,),
        )

    if claimed.rule_id == CUDA_RELU_RULE:
        if (
            claimed.kind != "call"
            or claimed.target not in RELU_TARGETS
            or claimed.receiver is not None
            or ctx.receiver is not None
            or claimed.keywords
            or claimed.operand_types != (TENSOR_F32_CUDA0_2D,)
            or not _nonliteral_operands(claimed, 1)
            or claimed.result_type != TENSOR_F32_CUDA0_2D
            or len(ctx.operands) != 1
        ):
            raise ValueError("TensorFlow CUDA ReLU metadata changed after claim")
        return LoweredExpr(
            f"rextio_tensorflow_cuda_runtime::relu(&{ctx.operands[0]})?",
            helpers=(helper,),
        )

    if claimed.rule_id != CUDA_MEAN_AXIS1_RULE:
        raise ValueError(f"unexpected TensorFlow CUDA rule {claimed.rule_id!r}")
    values = _keywords(claimed)
    if (
        claimed.kind != "call"
        or claimed.target not in MEAN_TARGETS
        or claimed.receiver is not None
        or ctx.receiver is not None
        or claimed.operand_types != (TENSOR_F32_CUDA0_2D,)
        or not _nonliteral_operands(claimed, 1)
        or set(values) not in ({"axis"}, {"axis", "keepdims"})
        or type(values.get("axis")) is not int
        or values.get("axis") != 1
        or type(values.get("keepdims", False)) is not bool
        or values.get("keepdims", False) is not False
        or claimed.result_type != TENSOR_F32_CUDA0_1D
        or len(ctx.operands) != 1
    ):
        raise ValueError("TensorFlow CUDA reduce_mean metadata changed after claim")
    return LoweredExpr(
        f"rextio_tensorflow_cuda_runtime::reduce_mean_axis1(&{ctx.operands[0]})?",
        helpers=(helper,),
    )


__all__ = ["CUDA_CAPABILITY_ID", "CUDA_PROVIDER_ID", "try_lower"]
