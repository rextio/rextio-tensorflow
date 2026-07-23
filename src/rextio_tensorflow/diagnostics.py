"""Plugin type keys, diagnostic codes, and rejection helpers."""

from __future__ import annotations

from rextio.analyzer.diagnostics import Diagnostic
from rextio.plugins.api import ClaimSite, Rejected

PLUGIN_ID = "rextio-tensorflow"

TENSOR_F32_CPU_1D = "rextio-tensorflow/tensor-f32-cpu-1d"
TENSOR_F32_CPU_2D = "rextio-tensorflow/tensor-f32-cpu-2d"
TENSOR_I64_CPU_1D = "rextio-tensorflow/tensor-i64-cpu-1d"

TENSOR_TYPE_KEYS: frozenset[str] = frozenset(
    {TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_I64_CPU_1D}
)

_TENSOR_META: dict[str, tuple[str, str, int]] = {
    TENSOR_F32_CPU_1D: ("f32", "cpu", 1),
    TENSOR_F32_CPU_2D: ("f32", "cpu", 2),
    TENSOR_I64_CPU_1D: ("i64", "cpu", 1),
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
    "DIAGNOSTIC_ARGMAX",
    "DIAGNOSTIC_MATMUL",
    "DIAGNOSTIC_MEAN",
    "DIAGNOSTIC_MUL_BINOP",
    "DIAGNOSTIC_RELU",
    "DIAGNOSTIC_SIGMOID",
    "DIAGNOSTIC_SOFTMAX",
    "DIAGNOSTIC_SUM",
    "DIAGNOSTIC_TANH",
    "DIAGNOSTIC_UNSUPPORTED",
    "PLUGIN_ID",
    "RUNTIME_ERRORS",
    "TENSOR_F32_CPU_1D",
    "TENSOR_F32_CPU_2D",
    "TENSOR_I64_CPU_1D",
    "TENSOR_TYPE_KEYS",
    "is_tensor_type",
    "reject",
    "tensor_meta",
]
