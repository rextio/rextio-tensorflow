"""Materialized TensorFlow tensor plugin types for Rextio plugin API 1.3."""

from __future__ import annotations

from rextio.plugins.api import BoundaryConversion, PluginType

from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

_BOUNDARY_SUPPORT = runtime_module_helpers()


def _tensor_type(key: str, annotation: str, extractor: str) -> PluginType:
    return PluginType(
        key=key,
        annotations=(f"rextio_tensorflow.types.{annotation}",),
        rust_type="rextio_tensorflow_runtime::RxtTfTensor",
        conversion=BoundaryConversion(
            param_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            param_expr=f"rextio_tensorflow_runtime::{extractor}(py, &{{param}})?",
            return_rust="pyo3::Bound<'py, pyo3::types::PyAny>",
            return_expr="rextio_tensorflow_runtime::materialize_tensor(py, {value})?",
        ),
        helpers=(_BOUNDARY_SUPPORT,),
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


__all__ = ["PLUGIN_TYPES", "plugin_type", "plugin_type_keys", "plugin_types"]
