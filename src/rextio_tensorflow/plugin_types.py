"""Materialized TensorFlow tensor plugin types for Rextio plugin API 1.6."""

from __future__ import annotations

from rextio.artifacts.models import RuntimeRequirement
from rextio.devices import DeviceValueMetadata
from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers
from rextio_tensorflow.rust_snippets.cuda_runtime import cuda_runtime_module_helpers

_BOUNDARY_SUPPORT = runtime_module_helpers()
_CUDA_BOUNDARY_SUPPORT = cuda_runtime_module_helpers()

CUDA_RUNTIME_REQUIREMENTS = (
    RuntimeRequirement(
        "tensorflow",
        "2.21.0",
        ("cuda", "python-wheel", "tfe-c-api"),
    ),
    RuntimeRequirement("cpython", "3.11", ("private-eager-abi",)),
)


def _tensor_type(
    key: str,
    annotation: str,
    extractor: str,
    *,
    rust_type: str = "rextio_tensorflow_runtime::RxtTfTensor",
    extractor_module: str = "rextio_tensorflow_runtime",
    materializer: str = "rextio_tensorflow_runtime::materialize_tensor",
    helpers: tuple[str, ...] = (_BOUNDARY_SUPPORT,),
    device_value_metadata: DeviceValueMetadata | None = None,
) -> PluginType:
    return PluginType(
        key=key,
        annotations=(f"rextio_tensorflow.types.{annotation}",),
        rust_type=rust_type,
        conversion=BoundaryConversion(
            param_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            param_expr=f"{extractor_module}::{extractor}(py, &{{param}})?",
            return_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            return_expr=f"{materializer}(py, {{value}})?",
        ),
        helpers=helpers,
        device_value_metadata=device_value_metadata,
    )


def _cuda_metadata(rank: int) -> DeviceValueMetadata:
    return DeviceValueMetadata(
        logical_device="cuda:0",
        backend="cuda",
        dtype="float32",
        rank=rank,
        layout="dense",
        runtime="tensorflow-tfe",
        runtime_version="2.21.0",
        reuse_domain_runtime=True,
        features=("eager", "inference", "no-grad"),
        memory_spaces=("device",),
        runtime_requirements=CUDA_RUNTIME_REQUIREMENTS,
    )


PLUGIN_TYPES: tuple[PluginType, ...] = (
    _tensor_type(
        TENSOR_F32_CPU_2D,
        "TensorF32Cpu2D",
        "extract_f32_cpu_2d",
    ),
    _tensor_type(
        TENSOR_F32_CPU_1D,
        "TensorF32Cpu1D",
        "extract_f32_cpu_1d",
    ),
    _tensor_type(
        TENSOR_I64_CPU_1D,
        "TensorI64Cpu1D",
        "extract_i64_cpu_1d",
    ),
    _tensor_type(
        TENSOR_F32_CUDA0_2D,
        "TensorF32Cuda0_2D",
        "extract_f32_cuda0_2d",
        rust_type="rextio_tensorflow_cuda_runtime::RxtTfCudaTensor",
        extractor_module="rextio_tensorflow_cuda_runtime",
        materializer="rextio_tensorflow_cuda_runtime::materialize_f32_cuda0_2d",
        helpers=(_CUDA_BOUNDARY_SUPPORT,),
        device_value_metadata=_cuda_metadata(2),
    ),
    _tensor_type(
        TENSOR_F32_CUDA0_1D,
        "TensorF32Cuda0_1D",
        "extract_f32_cuda0_1d",
        rust_type="rextio_tensorflow_cuda_runtime::RxtTfCudaTensor",
        extractor_module="rextio_tensorflow_cuda_runtime",
        materializer="rextio_tensorflow_cuda_runtime::materialize_f32_cuda0_1d",
        helpers=(_CUDA_BOUNDARY_SUPPORT,),
        device_value_metadata=_cuda_metadata(1),
    ),
)

_BY_KEY = {plugin_type.key: plugin_type for plugin_type in PLUGIN_TYPES}


def plugin_types() -> tuple[PluginType, ...]:
    """Return the Alpha materialized tensor vocabulary."""
    return PLUGIN_TYPES


def plugin_type(key: str) -> PluginType:
    """Return one registered type by key."""
    return _BY_KEY[key]


def plugin_type_keys() -> frozenset[str]:
    """Return all owned type keys."""
    return frozenset(_BY_KEY)


__all__ = [
    "CUDA_RUNTIME_REQUIREMENTS",
    "PLUGIN_TYPES",
    "plugin_type",
    "plugin_type_keys",
    "plugin_types",
]
