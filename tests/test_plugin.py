"""Plugin facade, coverage, crate pin, and loader contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.config.schema import PluginConfig, RextioConfig
from rextio.plugins.api import (
    PLUGIN_DIAGNOSTIC_CODE_PATTERN,
    CoverageDecl,
    PluginType,
    RuleRecord,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_tensorflow import __version__
from rextio_tensorflow.plugin import PLUGIN_ID, REQUIRED_PLUGIN_API, RextioTensorflowPlugin, plugin
from rextio_tensorflow.rules import COVERAGE, tensorflow_rule_records

ROOT = Path(__file__).resolve().parents[1]


class FakeEntryPoint:
    name = PLUGIN_ID

    def load(self) -> Any:
        return plugin


def load_registry(enabled: tuple[str, ...] = (PLUGIN_ID,)):
    return load_plugin_registry(
        PluginConfig(enabled=enabled),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def test_entry_point_factory_returns_plugin() -> None:
    obj = plugin()
    assert isinstance(obj, RextioTensorflowPlugin)
    assert obj.plugin_id == PLUGIN_ID
    assert obj.api_version == REQUIRED_PLUGIN_API == "1.3"
    assert __version__ == "0.1.1"


def test_core_loader_accepts_the_plugin() -> None:
    registry = load_registry()
    active = registry.active[0]
    assert active.id == PLUGIN_ID
    assert active.rules_provided is True
    assert active.lowering_provided is True
    assert active.api_version == "1.3"
    assert active.packages == ("tensorflow",)
    assert __version__ in active.name
    assert registry.coverages[0].coverage == COVERAGE
    assert [record.id for record in registry.rule_records] == [
        record.id for record in tensorflow_rule_records()
    ]


def test_core_loader_accepts_api_13_provider_without_artifact_capability() -> None:
    """Core owns API compatibility; this API 1.3 provider remains host-only."""
    registry = load_registry()
    active = registry.active[0]

    assert active.api_version == "1.3"
    assert getattr(active, "artifact_capability_declared", False) is False
    assert not hasattr(plugin(), "artifact_capability")


def test_covers_alpha_surface() -> None:
    coverage = plugin().covers()
    assert isinstance(coverage, CoverageDecl)
    assert coverage.packages == ("tensorflow",)
    assert "tensorflow.nn" in coverage.modules
    assert "tensorflow.matmul" in coverage.symbols
    assert "tensorflow.nn.relu" in coverage.symbols
    assert "tensorflow.reduce_mean" in coverage.symbols


def test_rule_records_are_namespaced_and_well_formed() -> None:
    records = plugin().describe(RextioConfig())
    assert records
    codes: set[str] = set()
    for record in records:
        assert isinstance(record, RuleRecord)
        assert record.id.startswith("rextio-tensorflow/")
        assert record.provider == "rextio-tensorflow"
        if record.diagnostic_code is not None:
            match = PLUGIN_DIAGNOSTIC_CODE_PATTERN.match(record.diagnostic_code)
            assert match is not None
            assert match.group(1) == "TENSORFLOW"
            assert record.diagnostic_code not in codes
            codes.add(record.diagnostic_code)
    assert "RXTP-TENSORFLOW-001" in codes
    assert "RXTP-TENSORFLOW-010" in codes


def test_type_vocabulary_keys_and_boundary() -> None:
    types = plugin().type_vocabulary()
    assert {t.key for t in types} == {
        "rextio-tensorflow/tensor-f32-cpu-2d",
        "rextio-tensorflow/tensor-f32-cpu-1d",
    }
    for plugin_type in types:
        assert isinstance(plugin_type, PluginType)
        assert plugin_type.rust_type == "rextio_tensorflow_runtime::RxtTfTensor"
        assert plugin_type.conversion is not None
        assert plugin_type.conversion.param_rust == "pyo3::Bound<'py, pyo3::types::PyAny>"
        assert "rextio_tensorflow_runtime::materialize_tensor" in (
            plugin_type.conversion.return_expr
        )
        assert plugin_type.helpers
        assert "mod rextio_tensorflow_runtime" in plugin_type.helpers[0]
        assert "EagerTensor_Handle" in plugin_type.helpers[0]
        assert "EagerTensorFromHandle" in plugin_type.helpers[0]
        assert "TFE_Execute" in plugin_type.helpers[0]
    spellings = {a for t in types for a in t.annotations}
    assert spellings == {
        "rextio_tensorflow.types.TensorF32Cpu2D",
        "rextio_tensorflow.types.TensorF32Cpu1D",
    }


def test_crate_dependencies_empty() -> None:
    deps = plugin().crate_dependencies()
    assert deps == ()


def test_no_tensorflow_sys_or_crate_dependency_in_helpers() -> None:
    helper = plugin().type_vocabulary()[0].helpers[0]
    assert "tensorflow-sys" not in helper
    assert "extern crate tensorflow" not in helper
    assert "use tensorflow::" not in helper
