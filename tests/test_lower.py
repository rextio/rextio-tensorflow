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

from rextio_tensorflow.claim.activations import RELU_RULE, SIGMOID_RULE, TANH_RULE
from rextio_tensorflow.claim.add import ADD_BINOP_RULE
from rextio_tensorflow.claim.classification import ARGMAX_RULE, SOFTMAX_RULE
from rextio_tensorflow.claim.matmul import MATMUL_RULE
from rextio_tensorflow.claim.reductions import MEAN_RULE
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
    assert ".unwrap()" not in helper
    assert ".expect(" not in helper
    assert "panic!" not in helper


def test_lower_unary_activations() -> None:
    for target, rule, helper in (
        ("tensorflow.nn.relu", RELU_RULE, "relu"),
        ("tensorflow.nn.sigmoid", SIGMOID_RULE, "sigmoid"),
        ("tensorflow.nn.tanh", TANH_RULE, "tanh"),
    ):
        claimed = ClaimSite(
            kind="call",
            target=target,
            operand_types=(TENSOR_F32_CPU_2D,),
            file_path="",
            line=0,
            column=0,
            rule_id=rule,
            result_type=TENSOR_F32_CPU_2D,
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
    with pytest.raises(ValueError, match="axis=1"):
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
