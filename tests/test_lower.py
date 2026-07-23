"""Focused lower/codegen tests for the Alpha surface (no real Cargo)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
from rextio.plugins.api import (
    ClaimLiteral,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    ReceiverMeta,
)

from rextio_tensorflow.claim.activations import (
    RELU_1D_RULE,
    RELU_RULE,
    SIGMOID_1D_RULE,
    SIGMOID_RULE,
    TANH_1D_RULE,
    TANH_RULE,
)
from rextio_tensorflow.claim.add import (
    ADD_BINOP_RULE,
    DIV_BINOP_RULE,
    DIV_CALL_RULE,
    MUL_BINOP_RULE,
    MUL_CALL_RULE,
    SUB_BINOP_RULE,
    SUB_CALL_RULE,
)
from rextio_tensorflow.claim.classification import (
    ARGMAX_AXIS0_RULE,
    ARGMAX_RULE,
    SOFTMAX_RULE,
)
from rextio_tensorflow.claim.matmul import MATMUL_RULE
from rextio_tensorflow.claim.reductions import (
    MEAN_GENERAL_RULE,
    MEAN_RULE,
    SUM_GENERAL_RULE,
    SUM_RULE,
)
from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_tensorflow.plugin import plugin
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

PLUGIN = plugin()


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_0"


def test_lower_matmul_targets_runtime_module() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = LoweringContext(
        operands=("x", "w"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == "rextio_tensorflow_runtime::matmul(&x, &w)?"
    assert runtime_module_helpers() in lowered.helpers
    assert "mod rextio_tensorflow_runtime" in runtime_module_helpers()
    assert "TFE_Execute" in runtime_module_helpers()
    # Fallible path must not panic; CString construction uses map_err, not unwrap.
    assert ".unwrap()" not in runtime_module_helpers()
    assert "panic!" not in runtime_module_helpers()
    assert "c_string(" in runtime_module_helpers()


def test_lower_rejects_standalone_rust_backend() -> None:
    """The Alpha runtime requires the CPython/PyO3 boundary and active TF wheel."""
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = SimpleNamespace(backend="standalone-rust")

    with pytest.raises(ValueError, match="standalone-rust"):
        PLUGIN.lower(claimed, cast(LoweringContext, ctx))


def test_lower_defaults_missing_legacy_backend_to_pyo3() -> None:
    """API 1.3 LoweringContext instances do not expose the API 1.4 field."""
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    legacy_ctx = SimpleNamespace(operands=("x", "w"), receiver=None)

    lowered = PLUGIN.lower(claimed, cast(LoweringContext, legacy_ctx))

    assert lowered.rust == "rextio_tensorflow_runtime::matmul(&x, &w)?"


def test_runtime_helper_has_same_wheel_and_ownership_hardening() -> None:
    helper = runtime_module_helpers()
    assert "RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD" in helper
    assert "dlopen(c_path.as_ptr(), flags)" in helper
    assert "tf_dlopen_flags" in helper
    assert "PlatformAbiProfile" in helper
    assert "dladdr" in helper
    assert "canonicalize" in helper
    assert 'cc.resolve("TFE_NewOp")' in helper
    assert 'framework.resolve("TF_NewStatus")' in helper
    assert "pywrap.resolve(SYM_EAGER_TENSOR_HANDLE)" in helper
    assert "cc_handle: usize" in helper
    assert "Rc<OwnedTensorHandle>" in helper
    assert "_python_context: Py<PyAny>" in helper
    assert "_python_capsule: Py<PyAny>" in helper
    assert "TFE_TensorHandleBackingDeviceName" in helper
    assert "TFE_OpSetDevice" in helper
    assert "TF_AllocateTensor" in helper
    assert "TF_TensorData" in helper
    assert "pub fn extract_i64_cpu_1d" in helper
    assert "extract_common(py, value, TF_INT64, 1)" in helper
    assert "tensor.validate_i64(expected_rank)?" in helper
    assert "unsafe impl Send" not in helper
    assert "RTLD_DEFAULT" not in helper
    assert "TF_NewTensor" not in helper
    assert "TFE_TensorHandleResolve" not in helper
    assert 'binary(left, right, "Sub", false, expected_rank)' in helper
    assert 'binary(left, right, "RealDiv", false, expected_rank)' in helper
    assert "input.validate_f32(expected_rank)?" in helper
    assert ".unwrap()" not in helper
    assert ".expect(" not in helper
    assert "panic!" not in helper


def test_lower_unary_activations() -> None:
    for target, operand, rule, helper in (
        ("tensorflow.nn.relu", TENSOR_F32_CPU_1D, RELU_1D_RULE, "relu"),
        ("tf.nn.relu", TENSOR_F32_CPU_2D, RELU_RULE, "relu"),
        ("tensorflow.nn.sigmoid", TENSOR_F32_CPU_1D, SIGMOID_1D_RULE, "sigmoid"),
        ("tf.nn.sigmoid", TENSOR_F32_CPU_2D, SIGMOID_RULE, "sigmoid"),
        ("tensorflow.nn.tanh", TENSOR_F32_CPU_1D, TANH_1D_RULE, "tanh"),
        ("tf.nn.tanh", TENSOR_F32_CPU_2D, TANH_RULE, "tanh"),
    ):
        claimed = ClaimSite(
            kind="call",
            target=target,
            operand_types=(operand,),
            file_path="",
            line=0,
            column=0,
            rule_id=rule,
            result_type=operand,
        )
        ctx = LoweringContext(
            operands=("tmp",),
            target_language="rust",
            fresh_name=_fresh_name,
        )
        lowered = PLUGIN.lower(claimed, ctx)
        assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&tmp)?"


def test_lower_add_binop() -> None:
    claimed = ClaimSite(
        kind="binop",
        target="+",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=ADD_BINOP_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = LoweringContext(
        operands=("x", "b"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == "rextio_tensorflow_runtime::add(&x, &b)?"


@pytest.mark.parametrize(
    ("kind", "target", "rule", "helper"),
    (
        ("call", "tensorflow.multiply", MUL_CALL_RULE, "mul"),
        ("call", "tf.math.multiply", MUL_CALL_RULE, "mul"),
        ("binop", "*", MUL_BINOP_RULE, "mul"),
        ("call", "tensorflow.subtract", SUB_CALL_RULE, "sub"),
        ("call", "tf.math.subtract", SUB_CALL_RULE, "sub"),
        ("binop", "-", SUB_BINOP_RULE, "sub"),
        ("call", "tensorflow.divide", DIV_CALL_RULE, "div"),
        ("call", "tf.math.divide", DIV_CALL_RULE, "div"),
        ("binop", "/", DIV_BINOP_RULE, "div"),
    ),
)
def test_lower_binary_surface_revalidates_broadcast_metadata(
    kind: str, target: str, rule: str, helper: str
) -> None:
    claimed = ClaimSite(
        kind=kind,
        target=target,
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=TENSOR_F32_CPU_2D,
    )
    ctx = LoweringContext(
        operands=("x", "scale"), target_language="rust", fresh_name=_fresh_name
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&x, &scale)?"
    malformed = replace(claimed, result_type=TENSOR_F32_CPU_1D)
    with pytest.raises(ValueError, match="operand/result"):
        PLUGIN.lower(malformed, ctx)
    with pytest.raises(ValueError, match="lower requires resolved"):
        PLUGIN.lower(replace(claimed, operand_types=(None, TENSOR_F32_CPU_1D)), ctx)


def test_lower_reduce_mean_axis1() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.reduce_mean",
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=MEAN_RULE,
        result_type=TENSOR_F32_CPU_1D,
        keywords=(
            KeywordArg(
                name="axis",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=1),
            ),
        ),
    )
    ctx = LoweringContext(
        operands=("h",),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == "rextio_tensorflow_runtime::reduce_mean_axis1(&h)?"


def test_lower_reduce_sum_axis1() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.reduce_sum",
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=SUM_RULE,
        result_type=TENSOR_F32_CPU_1D,
        keywords=(
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)
            ),
            KeywordArg(
                name="keepdims", arg_type="bool", literal=ClaimLiteral(is_literal=True, value=False)
            ),
        ),
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
    )
    assert lowered.rust == "rextio_tensorflow_runtime::reduce_sum_axis1(&h)?"


@pytest.mark.parametrize(
    ("target", "rule", "axis", "keepdims", "result_type", "helper"),
    (
        (
            "tensorflow.reduce_mean",
            MEAN_GENERAL_RULE,
            0,
            False,
            TENSOR_F32_CPU_1D,
            "reduce_mean_axis0",
        ),
        (
            "tensorflow.reduce_mean",
            MEAN_GENERAL_RULE,
            1,
            True,
            TENSOR_F32_CPU_2D,
            "reduce_mean_axis1_keepdims",
        ),
        (
            "tensorflow.reduce_sum",
            SUM_GENERAL_RULE,
            0,
            True,
            TENSOR_F32_CPU_2D,
            "reduce_sum_axis0_keepdims",
        ),
        (
            "tensorflow.reduce_sum",
            SUM_RULE,
            1,
            False,
            TENSOR_F32_CPU_1D,
            "reduce_sum_axis1",
        ),
    ),
)
def test_lower_reduction_positional_axis_is_metadata_only(
    target: str,
    rule: str,
    axis: int,
    keepdims: bool,
    result_type: str,
    helper: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(TENSOR_F32_CPU_2D, "int"),
        operand_literals=(
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=True, value=axis),
        ),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=result_type,
        keywords=(
            KeywordArg(
                name="keepdims",
                arg_type="bool",
                literal=ClaimLiteral(is_literal=True, value=keepdims),
            ),
        ),
    )
    one_operand_ctx = LoweringContext(
        operands=("h",), target_language="rust", fresh_name=_fresh_name
    )
    lowered = PLUGIN.lower(claimed, one_operand_ctx)
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"

    core_direct_ctx = replace(
        one_operand_ctx, operands=("h", "axis_must_not_reach_runtime")
    )
    lowered_direct = PLUGIN.lower(claimed, core_direct_ctx)
    assert lowered_direct.rust == lowered.rust
    assert "axis_must_not_reach_runtime" not in lowered_direct.rust


@pytest.mark.parametrize(
    "operand_literals",
    (
        (),
        (ClaimLiteral(is_literal=False),),
        (
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=False),
        ),
        (
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=True, value=True),
        ),
    ),
)
def test_lower_rejects_forged_positional_axis_alignment(
    operand_literals: tuple[ClaimLiteral, ...],
) -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.reduce_sum",
        operand_types=(TENSOR_F32_CPU_2D, "int"),
        operand_literals=operand_literals,
        file_path="",
        line=0,
        column=0,
        rule_id=SUM_GENERAL_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    with pytest.raises(ValueError, match="positional axis"):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("h",), target_language="rust", fresh_name=_fresh_name
            ),
        )


@pytest.mark.parametrize(
    ("target", "rule", "result_type", "helper"),
    (
        ("tensorflow.nn.softmax", SOFTMAX_RULE, TENSOR_F32_CPU_2D, "softmax_axis1"),
        ("tensorflow.argmax", ARGMAX_RULE, TENSOR_I64_CPU_1D, "argmax_axis1"),
    ),
)
def test_lower_classification_head(target: str, rule: str, result_type: str, helper: str) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=result_type,
        keywords=(
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)
            ),
        ),
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
    )
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"
    assert "axis_one_scalar" in runtime_module_helpers()
    assert "TF_INT64" in runtime_module_helpers()


@pytest.mark.parametrize(
    ("target", "axis", "rule", "result_type", "helper"),
    (
        (
            "tensorflow.nn.softmax",
            1,
            SOFTMAX_RULE,
            TENSOR_F32_CPU_2D,
            "softmax_axis1",
        ),
        (
            "tensorflow.argmax",
            0,
            ARGMAX_AXIS0_RULE,
            TENSOR_I64_CPU_1D,
            "argmax_axis0",
        ),
        (
            "tensorflow.argmax",
            1,
            ARGMAX_RULE,
            TENSOR_I64_CPU_1D,
            "argmax_axis1",
        ),
    ),
)
def test_lower_classification_positional_axis_is_metadata_only(
    target: str,
    axis: int,
    rule: str,
    result_type: str,
    helper: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(TENSOR_F32_CPU_2D, "int"),
        operand_literals=(
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=True, value=axis),
        ),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=result_type,
    )
    ctx = LoweringContext(
        operands=("h",), target_language="rust", fresh_name=_fresh_name
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"
    lowered_direct = PLUGIN.lower(
        claimed, replace(ctx, operands=("h", "axis_must_not_reach_runtime"))
    )
    assert lowered_direct.rust == lowered.rust


def test_classification_lower_revalidates_default_int64_contract() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.argmax",
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=ARGMAX_RULE,
        result_type=TENSOR_I64_CPU_1D,
        keywords=(
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=0)
            ),
        ),
    )
    with pytest.raises(ValueError, match="malformed argmax"):
        PLUGIN.lower(
            claimed,
            LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
        )


@pytest.mark.parametrize(
    ("target", "rule", "result_type"),
    (
        ("tensorflow.nn.softmax", SOFTMAX_RULE, TENSOR_F32_CPU_2D),
        ("tensorflow.argmax", ARGMAX_RULE, TENSOR_I64_CPU_1D),
        ("tensorflow.reduce_mean", MEAN_RULE, TENSOR_F32_CPU_1D),
        ("tensorflow.reduce_sum", SUM_RULE, TENSOR_F32_CPU_1D),
    ),
)
def test_lower_rejects_forged_duplicate_literal_axis_metadata(
    target: str, rule: str, result_type: str
) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(TENSOR_F32_CPU_2D,),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=result_type,
        keywords=(
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)
            ),
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)
            ),
        ),
    )
    with pytest.raises(ValueError, match="axis"):
        PLUGIN.lower(
            claimed,
            LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
        )


def test_functional_lower_rejects_claimed_or_rendered_receiver() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.matmul",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        file_path="",
        line=0,
        column=0,
        rule_id=MATMUL_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    context = LoweringContext(
        operands=("x", "w"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    with pytest.raises(ValueError, match="forbids receivers"):
        PLUGIN.lower(
            replace(
                claimed,
                receiver=ReceiverMeta(
                    arg_type=TENSOR_F32_CPU_2D,
                    expr_kind="name",
                    is_safe=True,
                ),
            ),
            context,
        )
    with pytest.raises(ValueError, match="forbids receivers"):
        PLUGIN.lower(claimed, replace(context, receiver="x"))
