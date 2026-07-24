"""Plugin type keys, diagnostic codes, and rejection helpers."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

PLUGIN_ID = "rextio-tensorflow"

TENSOR_F32_CPU_1D = "rextio-tensorflow/tensor-f32-cpu-1d"
TENSOR_F32_CPU_2D = "rextio-tensorflow/tensor-f32-cpu-2d"
TENSOR_I64_CPU_1D = "rextio-tensorflow/tensor-i64-cpu-1d"
TENSOR_F32_CUDA0_1D = "rextio-tensorflow/tensor-f32-cuda0-1d"
TENSOR_F32_CUDA0_2D = "rextio-tensorflow/tensor-f32-cuda0-2d"

TENSOR_TYPE_KEYS: frozenset[str] = frozenset(
    {
        TENSOR_F32_CPU_1D,
        TENSOR_F32_CPU_2D,
        TENSOR_F32_CUDA0_1D,
        TENSOR_F32_CUDA0_2D,
        TENSOR_I64_CPU_1D,
    }
)

_TENSOR_META: dict[str, tuple[str, str, int]] = {
    TENSOR_F32_CPU_1D: ("f32", "cpu", 1),
    TENSOR_F32_CPU_2D: ("f32", "cpu", 2),
    TENSOR_I64_CPU_1D: ("i64", "cpu", 1),
    TENSOR_F32_CUDA0_1D: ("f32", "cuda:0", 1),
    TENSOR_F32_CUDA0_2D: ("f32", "cuda:0", 2),
}

DIAGNOSTIC_MATMUL = "RXTP-TENSORFLOW-001"
DIAGNOSTIC_RELU = "RXTP-TENSORFLOW-002"
DIAGNOSTIC_ADD = "RXTP-TENSORFLOW-003"
DIAGNOSTIC_MEAN = "RXTP-TENSORFLOW-004"
DIAGNOSTIC_SIGMOID = "RXTP-TENSORFLOW-005"
DIAGNOSTIC_ADD_BINOP = "RXTP-TENSORFLOW-006"
DIAGNOSTIC_SOFTMAX = "RXTP-TENSORFLOW-007"
DIAGNOSTIC_ARGMAX = "RXTP-TENSORFLOW-008"
DIAGNOSTIC_TANH = "RXTP-TENSORFLOW-009"
DIAGNOSTIC_UNSUPPORTED = "RXTP-TENSORFLOW-010"
DIAGNOSTIC_SUM = "RXTP-TENSORFLOW-011"
DIAGNOSTIC_MUL_BINOP = "RXTP-TENSORFLOW-012"
DIAGNOSTIC_MUL_CALL = "RXTP-TENSORFLOW-013"
DIAGNOSTIC_SUB_CALL = "RXTP-TENSORFLOW-014"
DIAGNOSTIC_SUB_BINOP = "RXTP-TENSORFLOW-015"
DIAGNOSTIC_DIV_CALL = "RXTP-TENSORFLOW-016"
DIAGNOSTIC_DIV_BINOP = "RXTP-TENSORFLOW-017"
DIAGNOSTIC_RELU_1D = "RXTP-TENSORFLOW-018"
DIAGNOSTIC_SIGMOID_1D = "RXTP-TENSORFLOW-019"
DIAGNOSTIC_TANH_1D = "RXTP-TENSORFLOW-020"
DIAGNOSTIC_BIAS_ADD = "RXTP-TENSORFLOW-021"
DIAGNOSTIC_MEAN_GENERAL = "RXTP-TENSORFLOW-022"
DIAGNOSTIC_SUM_GENERAL = "RXTP-TENSORFLOW-023"
DIAGNOSTIC_ARGMAX_AXIS0 = "RXTP-TENSORFLOW-024"
DIAGNOSTIC_SOFTMAX_1D = "RXTP-TENSORFLOW-025"
DIAGNOSTIC_ABS = "RXTP-TENSORFLOW-026"
DIAGNOSTIC_NEGATIVE = "RXTP-TENSORFLOW-027"
DIAGNOSTIC_SQUARE = "RXTP-TENSORFLOW-028"
DIAGNOSTIC_EXP = "RXTP-TENSORFLOW-029"
DIAGNOSTIC_LOG = "RXTP-TENSORFLOW-030"
DIAGNOSTIC_SQRT = "RXTP-TENSORFLOW-031"
DIAGNOSTIC_MAXIMUM = "RXTP-TENSORFLOW-032"
DIAGNOSTIC_MINIMUM = "RXTP-TENSORFLOW-033"
DIAGNOSTIC_CUDA_E3 = "RXTP-TENSORFLOW-034"
DIAGNOSTIC_CUDA_MATMUL = "RXTP-TENSORFLOW-035"
DIAGNOSTIC_CUDA_BIAS_ADD = "RXTP-TENSORFLOW-036"
DIAGNOSTIC_CUDA_RELU = "RXTP-TENSORFLOW-037"
DIAGNOSTIC_CUDA_MEAN = "RXTP-TENSORFLOW-038"

RUNTIME_ERRORS = {
    "not_tensor": "rextio-tensorflow: expected a TensorFlow EagerTensor",
    "device": "rextio-tensorflow: expected a CPU tensor",
    "dtype": "rextio-tensorflow: expected a float32 tensor",
    "rank": "rextio-tensorflow: tensor rank does not match the annotated type",
    "version": "rextio-tensorflow: TensorFlow runtime version mismatch",
    "symbol": "rextio-tensorflow: required TensorFlow C/Eager symbol missing",
}


def tensor_meta(type_key: str) -> tuple[str, str, int] | None:
    """Return ``(dtype, device, rank)`` for a plugin tensor key, else None."""
    return _TENSOR_META.get(type_key)


def is_tensor_type(type_key: str | None) -> bool:
    """Report whether ``type_key`` is one of this plugin's tensor types."""
    return type_key is not None and type_key in TENSOR_TYPE_KEYS


def reject(site: ClaimSite, code: str, message: str, suggestion: str) -> Rejected:
    """Build a location-neutral plugin rejection; core stamps the source site."""
    return Rejected(
        diagnostic=Diagnostic(
            code=code,
            severity="error",
            message=f"rextio-tensorflow cannot lower {site.target!r}: {message}",
            file_path="",
            line=0,
            column=0,
            suggestion=suggestion,
        )
    )


__all__ = [
    "DIAGNOSTIC_ADD",
    "DIAGNOSTIC_ADD_BINOP",
    "DIAGNOSTIC_ABS",
    "DIAGNOSTIC_ARGMAX",
    "DIAGNOSTIC_ARGMAX_AXIS0",
    "DIAGNOSTIC_BIAS_ADD",
    "DIAGNOSTIC_CUDA_BIAS_ADD",
    "DIAGNOSTIC_CUDA_E3",
    "DIAGNOSTIC_CUDA_MATMUL",
    "DIAGNOSTIC_CUDA_MEAN",
    "DIAGNOSTIC_CUDA_RELU",
    "DIAGNOSTIC_DIV_BINOP",
    "DIAGNOSTIC_DIV_CALL",
    "DIAGNOSTIC_EXP",
    "DIAGNOSTIC_LOG",
    "DIAGNOSTIC_MATMUL",
    "DIAGNOSTIC_MAXIMUM",
    "DIAGNOSTIC_MEAN",
    "DIAGNOSTIC_MEAN_GENERAL",
    "DIAGNOSTIC_MINIMUM",
    "DIAGNOSTIC_MUL_BINOP",
    "DIAGNOSTIC_MUL_CALL",
    "DIAGNOSTIC_NEGATIVE",
    "DIAGNOSTIC_RELU",
    "DIAGNOSTIC_RELU_1D",
    "DIAGNOSTIC_SIGMOID",
    "DIAGNOSTIC_SIGMOID_1D",
    "DIAGNOSTIC_SOFTMAX",
    "DIAGNOSTIC_SOFTMAX_1D",
    "DIAGNOSTIC_SQRT",
    "DIAGNOSTIC_SQUARE",
    "DIAGNOSTIC_SUB_BINOP",
    "DIAGNOSTIC_SUB_CALL",
    "DIAGNOSTIC_SUM",
    "DIAGNOSTIC_SUM_GENERAL",
    "DIAGNOSTIC_TANH",
    "DIAGNOSTIC_TANH_1D",
    "DIAGNOSTIC_UNSUPPORTED",
    "PLUGIN_ID",
    "RUNTIME_ERRORS",
    "TENSOR_F32_CPU_1D",
    "TENSOR_F32_CPU_2D",
    "TENSOR_F32_CUDA0_1D",
    "TENSOR_F32_CUDA0_2D",
    "TENSOR_I64_CPU_1D",
    "TENSOR_TYPE_KEYS",
    "is_tensor_type",
    "reject",
    "tensor_meta",
]
