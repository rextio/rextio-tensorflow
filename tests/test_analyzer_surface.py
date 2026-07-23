"""Analyzer integration for exact TensorFlow aliases and literal-axis metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rextio.analyzer.models import FunctionAnalysis, ProjectAnalysis
from rextio.analyzer.project_scanner import analyze_project
from rextio.codegen.rust.generator import generate_rust_module
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.ir.lowering import PluginTypeMaps, lower_project
from rextio.ir.types import RxtPluginType
from rextio.plugins.api import ClaimLiteral
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec

from rextio_tensorflow.claim.add import (
    DIV_CALL_RULE,
    MUL_CALL_RULE,
    SUB_CALL_RULE,
)
from rextio_tensorflow.claim.classification import (
    ARGMAX_AXIS0_RULE,
    SOFTMAX_RULE,
)
from rextio_tensorflow.claim.reductions import (
    MEAN_GENERAL_RULE,
    SUM_RULE,
)
from rextio_tensorflow.plugin import PLUGIN_ID, plugin

SURFACE_SOURCE = """
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import tensorflow as tf
import tensorflow.math as tf_math
from tensorflow import divide as top_divide
from tensorflow import multiply as top_multiply
from tensorflow import subtract as top_subtract
from tensorflow.math import divide as math_divide
from tensorflow.math import multiply as math_multiply
from tensorflow.math import subtract as math_subtract


def multiply_top(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return top_multiply(left, right)


def multiply_math(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return math_multiply(left, right)


def subtract_top(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return top_subtract(left, right)


def subtract_math(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return math_subtract(left, right)


def divide_top(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return top_divide(left, right)


def divide_math(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return math_divide(left, right)


def mean_axis0_keepdims(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.reduce_mean(x, 0, keepdims=True)


def sum_axis1_positional(x: TensorF32Cpu2D) -> TensorF32Cpu1D:
    return tf_math.reduce_sum(x, 1, keepdims=False)


def argmax_axis0_positional(x: TensorF32Cpu2D) -> TensorI64Cpu1D:
    return tf.argmax(x, 0)


def softmax_axis1_positional(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.nn.softmax(x, 1)


def pseudo_truediv(left: TensorF32Cpu2D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.math.truediv(left, right)


def pseudo_raw_sub(left: TensorF32Cpu2D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.raw_ops.Sub(x=left, y=right)


def positional_keepdims_bool_is_not_offered(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.reduce_mean(x, 0, True)


def softmax_axis0_fallback(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.nn.softmax(x, 0)


def bias_add_fallback(
    x: TensorF32Cpu2D, bias: TensorF32Cpu1D
) -> TensorF32Cpu2D:
    return tf.nn.bias_add(x, bias)
"""


class FakeEntryPoint:
    """Entry-point shim for the analyzer's real plugin loader."""

    name = PLUGIN_ID

    def load(self) -> Any:
        """Return the import-minimal plugin factory."""
        return plugin


def _write_project(root: Path) -> Path:
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-tensorflow"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "surface_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(SURFACE_SOURCE, encoding="utf-8")
    return root


def _registry():
    return load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(FakeEntryPoint(),),
        full_config=RextioConfig(),
    )


def _lowering_inputs(registry):
    by_key: dict[str, RxtPluginType] = {}
    by_spelling: dict[str, RxtPluginType] = {}
    for binding in registry.types:
        plugin_type = binding.plugin_type
        conversion = plugin_type.conversion
        assert conversion is not None
        rxt_type = RxtPluginType(
            key=plugin_type.key,
            native_rust=plugin_type.rust_type,
            param_rust=conversion.param_rust,
            param_expr=conversion.param_expr,
            return_rust=conversion.return_rust,
            return_expr=conversion.return_expr,
            uses=plugin_type.uses,
            helpers=plugin_type.helpers,
        )
        by_key[plugin_type.key] = rxt_type
        for spelling in plugin_type.annotations:
            by_spelling[spelling] = rxt_type
    providers = {
        binding.plugin_id: binding.provider for binding in registry.providers
    }
    return PluginTypeMaps(by_key=by_key, by_spelling=by_spelling), providers, by_key


def _function(analysis: ProjectAnalysis, name: str) -> FunctionAnalysis:
    qualname = f"surface_app.kernels.{name}"
    for module in analysis.modules:
        for function in module.functions:
            if function.qualname == qualname:
                return function
    raise AssertionError(f"function {qualname!r} not found")


def test_analyzer_resolves_only_explicit_binary_import_aliases(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    expected = {
        "multiply_top": ("tensorflow.multiply", MUL_CALL_RULE),
        "multiply_math": ("tensorflow.math.multiply", MUL_CALL_RULE),
        "subtract_top": ("tensorflow.subtract", SUB_CALL_RULE),
        "subtract_math": ("tensorflow.math.subtract", SUB_CALL_RULE),
        "divide_top": ("tensorflow.divide", DIV_CALL_RULE),
        "divide_math": ("tensorflow.math.divide", DIV_CALL_RULE),
    }
    for name, (target, rule) in expected.items():
        function = _function(analysis, name)
        assert function.accepted is True
        assert function.route == f"native-plugin:{PLUGIN_ID}"
        assert len(function.plugin_claims) == 1
        claim = function.plugin_claims[0]
        assert claim.target == target
        assert claim.rule_id == rule

    for name in ("pseudo_truediv", "pseudo_raw_sub"):
        function = _function(analysis, name)
        assert not function.plugin_claims


def test_analyzer_preserves_positional_axis_literal_alignment(tmp_path: Path) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    expected = {
        "mean_axis0_keepdims": (MEAN_GENERAL_RULE, 0),
        "sum_axis1_positional": (SUM_RULE, 1),
        "argmax_axis0_positional": (ARGMAX_AXIS0_RULE, 0),
        "softmax_axis1_positional": (SOFTMAX_RULE, 1),
    }
    for name, (rule, axis) in expected.items():
        function = _function(analysis, name)
        assert function.accepted is True
        assert len(function.plugin_claims) == 1
        claim = function.plugin_claims[0]
        assert claim.rule_id == rule
        assert len(claim.operand_types) == 2
        assert claim.operand_types[1] == "int"
        assert claim.operand_literals == (
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=True, value=axis),
        )

    mean = _function(analysis, "mean_axis0_keepdims").plugin_claims[0]
    assert len(mean.keywords) == 1
    assert mean.keywords[0].name == "keepdims"
    assert mean.keywords[0].literal == ClaimLiteral(is_literal=True, value=True)

    for name in (
        "positional_keepdims_bool_is_not_offered",
        "softmax_axis0_fallback",
        "bias_add_fallback",
    ):
        function = _function(analysis, name)
        assert not function.plugin_claims


def test_analyzer_positional_axes_lower_with_core_rendered_full_arity(
    tmp_path: Path,
) -> None:
    registry = _registry()
    analysis = analyze_project(
        _write_project(tmp_path),
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=RextioConfig(),
    )
    type_maps, providers, by_key = _lowering_inputs(registry)
    module_ir = lower_project(analysis, plugin_types=type_maps)
    source = generate_rust_module(
        module_ir,
        plugin_providers=providers,
        plugin_types_by_key=by_key,
    )

    assert "rextio_tensorflow_runtime::reduce_mean_axis0_keepdims(&x)?" in source
    assert "rextio_tensorflow_runtime::reduce_sum_axis1(&x)?" in source
    assert "rextio_tensorflow_runtime::argmax_axis0(&x)?" in source
    assert "rextio_tensorflow_runtime::softmax_axis1(&x)?" in source
    assert "reduce_mean_axis0_keepdims(&x, " not in source
    assert "reduce_sum_axis1(&x, " not in source
    assert "argmax_axis0(&x, " not in source
    assert "softmax_axis1(&x, " not in source
