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
    ADD_CALL_RULE,
    BIAS_ADD_RULE,
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
    SOFTMAX_1D_RULE,
    SOFTMAX_RULE,
)
from rextio_tensorflow.claim.matmul import MATMUL_RULE
from rextio_tensorflow.claim.reductions import (
    MEAN_GENERAL_RULE,
    MEAN_RULE,
    SUM_GENERAL_RULE,
    SUM_RULE,
)
from rextio_tensorflow.claim.unary import (
    ABS_RULE,
    EXP_RULE,
    LOG_RULE,
    NEGATIVE_RULE,
    SQRT_RULE,
    SQUARE_RULE,
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


@pytest.mark.parametrize(
    ("target", "rule", "helper", "tfe_operation"),
    (
        ("tensorflow.abs", ABS_RULE, "abs", "Abs"),
        ("tensorflow.negative", NEGATIVE_RULE, "negative", "Neg"),
        ("tensorflow.square", SQUARE_RULE, "square", "Square"),
        ("tensorflow.exp", EXP_RULE, "exp", "Exp"),
        ("tensorflow.math.log", LOG_RULE, "log", "Log"),
        ("tensorflow.math.sqrt", SQRT_RULE, "sqrt", "Sqrt"),
    ),
)
@pytest.mark.parametrize("operand_type", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D))
def test_lower_math_unary_surface(
    target: str,
    rule: str,
    helper: str,
    tfe_operation: str,
    operand_type: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(operand_type,),
        operand_literals=(ClaimLiteral(is_literal=False),),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=operand_type,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("value",),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&value)?"
    assert f'unary(input, "{tfe_operation}")' in runtime_module_helpers()


@pytest.mark.parametrize(
    ("mutations", "message"),
    (
        ({"rule_id": NEGATIVE_RULE}, "malformed abs"),
        ({"result_type": TENSOR_F32_CPU_2D}, "malformed abs"),
        ({"operand_types": (TENSOR_I64_CPU_1D,)}, "malformed abs"),
        (
            {
                "keywords": (
                    KeywordArg(
                        name="name",
                        arg_type="str",
                        literal=ClaimLiteral(is_literal=True, value="abs"),
                    ),
                )
            },
            "malformed abs",
        ),
        (
            {"operand_literals": (ClaimLiteral(is_literal=True, value=1),)},
            "malformed abs",
        ),
    ),
)
def test_math_unary_lower_rejects_forged_metadata(
    mutations: dict[str, object],
    message: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.abs",
        operand_types=(TENSOR_F32_CPU_1D,),
        file_path="",
        line=0,
        column=0,
        rule_id=ABS_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    with pytest.raises(ValueError, match=message):
        PLUGIN.lower(
            replace(claimed, **mutations),
            LoweringContext(
                operands=("value",),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


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
    ("target", "rule", "helper"),
    (
        (
            "tensorflow.maximum",
            "rextio-tensorflow/maximum-call-f32-cpu",
            "maximum",
        ),
        (
            "tensorflow.minimum",
            "rextio-tensorflow/minimum-call-f32-cpu",
            "minimum",
        ),
    ),
)
@pytest.mark.parametrize("operand_type", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D))
def test_lower_maximum_minimum_revalidates_same_rank_metadata(
    target: str,
    rule: str,
    helper: str,
    operand_type: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target=target,
        operand_types=(operand_type, operand_type),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=operand_type,
    )
    ctx = LoweringContext(
        operands=("left", "right"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&left, &right)?"

    with pytest.raises(ValueError, match="operand/result"):
        PLUGIN.lower(
            replace(claimed, operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)),
            ctx,
        )
    with pytest.raises(ValueError, match="operand/result"):
        PLUGIN.lower(replace(claimed, result_type=TENSOR_I64_CPU_1D), ctx)


def test_maximum_minimum_runtime_checks_exact_shape_before_building_tfe_op() -> None:
    helper = runtime_module_helpers()
    shape_check = helper.index("ensure_exact_same_shape(left, right)?;")
    maximum_op = helper.index(
        'binary_same_shape(left, right, "Maximum", expected_rank)'
    )
    minimum_op = helper.index(
        'binary_same_shape(left, right, "Minimum", expected_rank)'
    )
    op_construction = helper.index(
        "let op = OwnedOp::new(Rc::clone(&context), op_name, &status)?;",
        shape_check,
    )
    assert shape_check < op_construction
    assert maximum_op > op_construction
    assert minimum_op > op_construction


@pytest.mark.parametrize("explicit_nhwc", (False, True))
def test_lower_bounded_nhwc_bias_add(explicit_nhwc: bool) -> None:
    keywords = (
        (
            KeywordArg(
                name="data_format",
                arg_type="str",
                literal=ClaimLiteral(is_literal=True, value="NHWC"),
            ),
        )
        if explicit_nhwc
        else ()
    )
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.nn.bias_add",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        operand_literals=(
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=False),
        ),
        file_path="",
        line=0,
        column=0,
        rule_id=BIAS_ADD_RULE,
        result_type=TENSOR_F32_CPU_2D,
        keywords=keywords,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=("value", "bias"),
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == "rextio_tensorflow_runtime::bias_add(&value, &bias)?"
    helper = runtime_module_helpers()
    assert 'cc.resolve("TFE_OpSetAttrString")' in helper
    assert 'op.set_string("data_format", "NHWC")?' in helper
    assert "RTLD_DEFAULT" not in helper


@pytest.mark.parametrize(
    ("mutations", "message"),
    (
        ({"rule_id": ADD_CALL_RULE}, "malformed bias_add"),
        ({"result_type": TENSOR_F32_CPU_1D}, "malformed bias_add"),
        (
            {"operand_types": (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D)},
            "malformed bias_add",
        ),
        (
            {
                "keywords": (
                    KeywordArg(
                        name="data_format",
                        arg_type="str",
                        literal=ClaimLiteral(is_literal=True, value="NCHW"),
                    ),
                )
            },
            "literal data_format='NHWC'",
        ),
        (
            {
                "keywords": (
                    KeywordArg(
                        name="name",
                        arg_type="str",
                        literal=ClaimLiteral(is_literal=True, value="bias"),
                    ),
                )
            },
            "accepts only data_format",
        ),
        (
            {
                "operand_literals": (
                    ClaimLiteral(is_literal=False),
                    ClaimLiteral(is_literal=True, value=1),
                )
            },
            "forged positional literal metadata",
        ),
    ),
)
def test_bias_add_lower_rejects_forged_metadata(
    mutations: dict[str, object],
    message: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.nn.bias_add",
        operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
        file_path="",
        line=0,
        column=0,
        rule_id=BIAS_ADD_RULE,
        result_type=TENSOR_F32_CPU_2D,
    )
    with pytest.raises(ValueError, match=message):
        PLUGIN.lower(
            replace(claimed, **mutations),
            LoweringContext(
                operands=("value", "bias"),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


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
    ctx = LoweringContext(operands=("x", "scale"), target_language="rust", fresh_name=_fresh_name)
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
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
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
    core_direct_ctx = LoweringContext(
        operands=("h", "axis_must_not_reach_runtime"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered_direct = PLUGIN.lower(claimed, core_direct_ctx)
    assert lowered_direct.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"
    assert "axis_must_not_reach_runtime" not in lowered_direct.rust
    with pytest.raises(ValueError, match="one rendered operand"):
        PLUGIN.lower(
            claimed,
            replace(core_direct_ctx, operands=("h",)),
        )


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
                operands=("h", "axis"),
                target_language="rust",
                fresh_name=_fresh_name,
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
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        ),
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
    )
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"
    assert "axis_one_scalar" in runtime_module_helpers()
    assert "TF_INT64" in runtime_module_helpers()


@pytest.mark.parametrize("axis_form", ("default", "keyword", "positional"))
def test_lower_rank1_softmax_final_axis_forms(axis_form: str) -> None:
    operand_types: tuple[str | None, ...] = (TENSOR_F32_CPU_1D,)
    operand_literals: tuple[ClaimLiteral, ...] = ()
    keywords: tuple[KeywordArg, ...] = ()
    operands = ("h",)
    if axis_form == "keyword":
        keywords = (
            KeywordArg(
                name="axis",
                arg_type="int",
                literal=ClaimLiteral(is_literal=True, value=0),
            ),
        )
    elif axis_form == "positional":
        operand_types = (TENSOR_F32_CPU_1D, "int")
        operand_literals = (
            ClaimLiteral(is_literal=False),
            ClaimLiteral(is_literal=True, value=0),
        )
        operands = ("h", "axis_must_not_reach_runtime")

    claimed = ClaimSite(
        kind="call",
        target="tensorflow.nn.softmax",
        operand_types=operand_types,
        operand_literals=operand_literals,
        file_path="",
        line=0,
        column=0,
        rule_id=SOFTMAX_1D_RULE,
        result_type=TENSOR_F32_CPU_1D,
        keywords=keywords,
    )
    lowered = PLUGIN.lower(
        claimed,
        LoweringContext(
            operands=operands,
            target_language="rust",
            fresh_name=_fresh_name,
        ),
    )
    assert lowered.rust == "rextio_tensorflow_runtime::softmax_axis0(&h)?"
    assert "axis_must_not_reach_runtime" not in lowered.rust
    assert 'unary(input, "Softmax")' in runtime_module_helpers()


@pytest.mark.parametrize(
    ("operand_type", "rule", "result_type", "keywords", "message"),
    (
        (
            TENSOR_F32_CPU_1D,
            SOFTMAX_1D_RULE,
            TENSOR_F32_CPU_1D,
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
            ),
            "rank-1 softmax",
        ),
        (
            TENSOR_F32_CPU_2D,
            SOFTMAX_RULE,
            TENSOR_F32_CPU_2D,
            (),
            "rank-2 axis=1",
        ),
        (
            TENSOR_F32_CPU_2D,
            SOFTMAX_1D_RULE,
            TENSOR_F32_CPU_1D,
            (),
            "malformed softmax",
        ),
    ),
)
def test_softmax_lower_rejects_forged_rank_axis_rule_combinations(
    operand_type: str,
    rule: str,
    result_type: str,
    keywords: tuple[KeywordArg, ...],
    message: str,
) -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.nn.softmax",
        operand_types=(operand_type,),
        file_path="",
        line=0,
        column=0,
        rule_id=rule,
        result_type=result_type,
        keywords=keywords,
    )
    with pytest.raises(ValueError, match=message):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("h",),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


def test_rank1_softmax_lower_rejects_forged_default_literal_metadata() -> None:
    claimed = ClaimSite(
        kind="call",
        target="tensorflow.nn.softmax",
        operand_types=(TENSOR_F32_CPU_1D,),
        operand_literals=(ClaimLiteral(is_literal=True, value=0),),
        file_path="",
        line=0,
        column=0,
        rule_id=SOFTMAX_1D_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )
    with pytest.raises(ValueError, match="forged positional literal metadata"):
        PLUGIN.lower(
            claimed,
            LoweringContext(
                operands=("h",),
                target_language="rust",
                fresh_name=_fresh_name,
            ),
        )


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
        operands=("h", "axis_must_not_reach_runtime"),
        target_language="rust",
        fresh_name=_fresh_name,
    )
    lowered = PLUGIN.lower(claimed, ctx)
    assert lowered.rust == f"rextio_tensorflow_runtime::{helper}(&h)?"
    assert "axis_must_not_reach_runtime" not in lowered.rust
    with pytest.raises(ValueError, match="one rendered operand"):
        PLUGIN.lower(claimed, replace(ctx, operands=("h",)))


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
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=0)),
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
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        ),
    )
    with pytest.raises(ValueError, match="axis"):
        PLUGIN.lower(
            claimed,
            LoweringContext(operands=("h",), target_language="rust", fresh_name=_fresh_name),
        )


@pytest.mark.parametrize(
    ("target", "rule", "result_type", "keywords", "message"),
    (
        (
            "tensorflow.reduce_mean",
            MEAN_GENERAL_RULE,
            TENSOR_F32_CPU_1D,
            (
                KeywordArg(
                    name="axis",
                    arg_type=TENSOR_F32_CPU_1D,
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
            "arg_type='int'",
        ),
        (
            "tensorflow.reduce_sum",
            SUM_GENERAL_RULE,
            TENSOR_F32_CPU_2D,
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
                KeywordArg(
                    name="keepdims",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=True),
                ),
            ),
            "arg_type='bool'",
        ),
        (
            "tensorflow.argmax",
            ARGMAX_AXIS0_RULE,
            TENSOR_I64_CPU_1D,
            (
                KeywordArg(
                    name="axis",
                    arg_type=TENSOR_F32_CPU_1D,
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
            "arg_type='int'",
        ),
    ),
)
def test_lower_rejects_axis_keyword_arg_type_literal_contradictions(
    target: str,
    rule: str,
    result_type: str,
    keywords: tuple[KeywordArg, ...],
    message: str,
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
        keywords=keywords,
    )
    with pytest.raises(ValueError, match=message):
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
