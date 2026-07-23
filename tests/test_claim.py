"""Focused claim tests for the Alpha TF surface."""

from __future__ import annotations

import pytest

from rextio.config.schema import RextioConfig
from rextio.plugins.api import (
    ClaimLiteral,
    Claimed,
    ClaimSite,
    KeywordArg,
    NotCovered,
    Rejected,
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
from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_1D,
    TENSOR_F32_CPU_2D,
    TENSOR_I64_CPU_1D,
)
from rextio_tensorflow.plugin import plugin

PLUGIN = plugin()
CONFIG = RextioConfig()


def _call(
    target: str,
    operands: tuple[str | None, ...],
    *,
    operand_literals: tuple[ClaimLiteral, ...] = (),
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operands,
        file_path="",
        line=0,
        column=0,
        operand_literals=operand_literals,
        keywords=keywords,
    )


def test_claims_matmul() -> None:
    result = PLUGIN.claim(
        _call("tensorflow.matmul", (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)),
        CONFIG,
    )
    assert result == Claimed(rule_id=MATMUL_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_linalg_matmul_alias() -> None:
    result = PLUGIN.claim(
        _call("tensorflow.linalg.matmul", (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)),
        CONFIG,
    )
    assert result == Claimed(rule_id=MATMUL_RULE, result_type=TENSOR_F32_CPU_2D)


@pytest.mark.parametrize(
    ("target", "operand", "rule"),
    (
        ("tensorflow.nn.relu", TENSOR_F32_CPU_1D, RELU_1D_RULE),
        ("tf.nn.relu", TENSOR_F32_CPU_2D, RELU_RULE),
        ("tensorflow.nn.sigmoid", TENSOR_F32_CPU_1D, SIGMOID_1D_RULE),
        ("tf.nn.sigmoid", TENSOR_F32_CPU_2D, SIGMOID_RULE),
        ("tensorflow.nn.tanh", TENSOR_F32_CPU_1D, TANH_1D_RULE),
        ("tf.nn.tanh", TENSOR_F32_CPU_2D, TANH_RULE),
    ),
)
def test_claims_rank1_and_rank2_activations(target: str, operand: str, rule: str) -> None:
    result = PLUGIN.claim(_call(target, (operand,)), CONFIG)
    assert result == Claimed(rule_id=rule, result_type=operand)


def test_claims_add_rank2_bias() -> None:
    result = PLUGIN.claim(
        _call("tensorflow.add", (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)),
        CONFIG,
    )
    assert result == Claimed(rule_id=ADD_CALL_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_binop_add() -> None:
    result = PLUGIN.claim(
        ClaimSite(
            kind="binop",
            target="+",
            operand_types=(TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D),
            file_path="",
            line=0,
            column=0,
        ),
        CONFIG,
    )
    assert result == Claimed(rule_id=ADD_BINOP_RULE, result_type=TENSOR_F32_CPU_2D)


@pytest.mark.parametrize(
    ("target", "rule"),
    (
        ("tensorflow.multiply", MUL_CALL_RULE),
        ("tensorflow.math.multiply", MUL_CALL_RULE),
        ("tf.multiply", MUL_CALL_RULE),
        ("tf.math.multiply", MUL_CALL_RULE),
        ("tensorflow.subtract", SUB_CALL_RULE),
        ("tensorflow.math.subtract", SUB_CALL_RULE),
        ("tf.subtract", SUB_CALL_RULE),
        ("tf.math.subtract", SUB_CALL_RULE),
        ("tensorflow.divide", DIV_CALL_RULE),
        ("tensorflow.math.divide", DIV_CALL_RULE),
        ("tf.divide", DIV_CALL_RULE),
        ("tf.math.divide", DIV_CALL_RULE),
    ),
)
@pytest.mark.parametrize(
    ("left", "right", "result_type"),
    (
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    ),
)
def test_claims_functional_binary_surface(
    target: str, rule: str, left: str, right: str, result_type: str
) -> None:
    result = PLUGIN.claim(_call(target, (left, right)), CONFIG)
    assert result == Claimed(rule_id=rule, result_type=result_type)


@pytest.mark.parametrize(
    ("operator", "rule"),
    (
        ("*", MUL_BINOP_RULE),
        ("-", SUB_BINOP_RULE),
        ("/", DIV_BINOP_RULE),
    ),
)
@pytest.mark.parametrize(
    ("left", "right", "result_type"),
    (
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_1D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D),
        (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
    ),
)
def test_claims_binary_operator_surface(
    operator: str, rule: str, left: str, right: str, result_type: str
) -> None:
    result = PLUGIN.claim(
        ClaimSite(
            kind="binop",
            target=operator,
            operand_types=(left, right),
            file_path="",
            line=0,
            column=0,
        ),
        CONFIG,
    )
    assert result == Claimed(rule_id=rule, result_type=result_type)


def test_binary_surface_rejects_scalars_keywords_and_unlisted_aliases() -> None:
    scalar = PLUGIN.claim(
        ClaimSite(
            kind="binop",
            target="*",
            operand_types=(TENSOR_F32_CPU_2D, "float"),
            file_path="",
            line=0,
            column=0,
        ),
        CONFIG,
    )
    assert isinstance(scalar, Rejected)
    keyword = PLUGIN.claim(
        _call(
            "tensorflow.divide",
            (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D),
            keywords=(
                KeywordArg(
                    name="name",
                    arg_type="str",
                    literal=ClaimLiteral(is_literal=True, value="division"),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(keyword, Rejected)
    for target in (
        "tensorflow.math.truediv",
        "tensorflow.raw_ops.Sub",
        "tensorflow.multiply_extra",
    ):
        result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D)), CONFIG)
        assert isinstance(result, NotCovered)


def test_bias_add_is_explicitly_rejected_until_tfe_contract_is_proven() -> None:
    for target in ("tensorflow.nn.bias_add", "tf.nn.bias_add"):
        result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_1D)), CONFIG)
        assert isinstance(result, Rejected)
        assert result.diagnostic.code == "RXTP-TENSORFLOW-021"
        assert "data_format" in result.diagnostic.message


def test_binop_add_rejection_uses_binop_diagnostic_authority() -> None:
    result = PLUGIN.claim(
        ClaimSite(
            kind="binop",
            target="+",
            operand_types=(TENSOR_F32_CPU_2D,),
            file_path="",
            line=0,
            column=0,
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert result.diagnostic.code == "RXTP-TENSORFLOW-006"


def test_claims_reduce_mean_axis1() -> None:
    keywords = (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=1),
        ),
    )
    result = PLUGIN.claim(
        _call("tensorflow.reduce_mean", (TENSOR_F32_CPU_2D,), keywords=keywords),
        CONFIG,
    )
    assert result == Claimed(rule_id=MEAN_RULE, result_type=TENSOR_F32_CPU_1D)


def test_claims_reduce_sum_axis1_with_keepdims_false() -> None:
    keywords = (
        KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(
            name="keepdims", arg_type="bool", literal=ClaimLiteral(is_literal=True, value=False)
        ),
    )
    result = PLUGIN.claim(
        _call("tensorflow.reduce_sum", (TENSOR_F32_CPU_2D,), keywords=keywords),
        CONFIG,
    )
    assert result == Claimed(rule_id=SUM_RULE, result_type=TENSOR_F32_CPU_1D)


@pytest.mark.parametrize("target", ("tensorflow.reduce_mean", "tensorflow.math.reduce_mean"))
@pytest.mark.parametrize(
    ("axis", "keepdims", "expected_rule", "result_type"),
    (
        (0, False, MEAN_GENERAL_RULE, TENSOR_F32_CPU_1D),
        (1, False, MEAN_RULE, TENSOR_F32_CPU_1D),
        (0, True, MEAN_GENERAL_RULE, TENSOR_F32_CPU_2D),
        (1, True, MEAN_GENERAL_RULE, TENSOR_F32_CPU_2D),
    ),
)
def test_claims_reduce_mean_literal_axis_keepdims_matrix(
    target: str,
    axis: int,
    keepdims: bool,
    expected_rule: str,
    result_type: str,
) -> None:
    keywords = (
        KeywordArg(
            name="axis",
            arg_type="int",
            literal=ClaimLiteral(is_literal=True, value=axis),
        ),
        KeywordArg(
            name="keepdims",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=keepdims),
        ),
    )
    result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D,), keywords=keywords), CONFIG)
    assert result == Claimed(rule_id=expected_rule, result_type=result_type)


@pytest.mark.parametrize("target", ("tensorflow.reduce_sum", "tensorflow.math.reduce_sum"))
@pytest.mark.parametrize(
    ("axis", "keepdims", "expected_rule", "result_type"),
    (
        (0, False, SUM_GENERAL_RULE, TENSOR_F32_CPU_1D),
        (1, False, SUM_RULE, TENSOR_F32_CPU_1D),
        (0, True, SUM_GENERAL_RULE, TENSOR_F32_CPU_2D),
        (1, True, SUM_GENERAL_RULE, TENSOR_F32_CPU_2D),
    ),
)
def test_claims_reduce_sum_positional_axis_keepdims_matrix(
    target: str,
    axis: int,
    keepdims: bool,
    expected_rule: str,
    result_type: str,
) -> None:
    literals = (
        ClaimLiteral(is_literal=False),
        ClaimLiteral(is_literal=True, value=axis),
    )
    keywords = (
        KeywordArg(
            name="keepdims",
            arg_type="bool",
            literal=ClaimLiteral(is_literal=True, value=keepdims),
        ),
    )
    result = PLUGIN.claim(
        _call(
            target,
            (TENSOR_F32_CPU_2D, "int"),
            operand_literals=literals,
            keywords=keywords,
        ),
        CONFIG,
    )
    assert result == Claimed(rule_id=expected_rule, result_type=result_type)


@pytest.mark.parametrize(
    ("target", "axis", "positional", "rule"),
    (
        ("tensorflow.nn.softmax", 1, False, SOFTMAX_RULE),
        ("tf.nn.softmax", 1, True, SOFTMAX_RULE),
        ("tensorflow.argmax", 1, False, ARGMAX_RULE),
        ("tf.argmax", 1, True, ARGMAX_RULE),
        ("tensorflow.argmax", 0, False, ARGMAX_AXIS0_RULE),
        ("tf.argmax", 0, True, ARGMAX_AXIS0_RULE),
    ),
)
def test_claims_classification_literal_axis_forms(
    target: str, axis: int, positional: bool, rule: str
) -> None:
    if positional:
        site = _call(
            target,
            (TENSOR_F32_CPU_2D, "int"),
            operand_literals=(
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=True, value=axis),
            ),
        )
    else:
        site = _call(
            target,
            (TENSOR_F32_CPU_2D,),
            keywords=(
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=axis),
                ),
            ),
        )
    result = PLUGIN.claim(site, CONFIG)
    expected_type = TENSOR_F32_CPU_2D if "softmax" in target else TENSOR_I64_CPU_1D
    assert result == Claimed(rule_id=rule, result_type=expected_type)


@pytest.mark.parametrize("target", ("tensorflow.nn.softmax", "tf.nn.softmax"))
@pytest.mark.parametrize("axis_form", ("default", "keyword", "positional"))
def test_claims_rank1_softmax_final_axis_forms(target: str, axis_form: str) -> None:
    if axis_form == "default":
        site = _call(target, (TENSOR_F32_CPU_1D,))
    elif axis_form == "keyword":
        site = _call(
            target,
            (TENSOR_F32_CPU_1D,),
            keywords=(
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
        )
    else:
        site = _call(
            target,
            (TENSOR_F32_CPU_1D, "int"),
            operand_literals=(
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=True, value=0),
            ),
        )

    assert PLUGIN.claim(site, CONFIG) == Claimed(
        rule_id=SOFTMAX_1D_RULE,
        result_type=TENSOR_F32_CPU_1D,
    )


@pytest.mark.parametrize(
    ("operand_types", "operand_literals", "keywords"),
    (
        (
            (TENSOR_F32_CPU_1D,),
            (),
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
            ),
        ),
        (
            (TENSOR_F32_CPU_1D,),
            (),
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=False),
                ),
            ),
        ),
        (
            (TENSOR_F32_CPU_1D,),
            (),
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
                KeywordArg(
                    name="name",
                    arg_type="str",
                    literal=ClaimLiteral(is_literal=True, value="softmax"),
                ),
            ),
        ),
        (
            (TENSOR_F32_CPU_1D, "int"),
            (
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=False),
            ),
            (),
        ),
        (
            (TENSOR_F32_CPU_1D,),
            (ClaimLiteral(is_literal=True, value=0),),
            (),
        ),
    ),
)
def test_rank1_softmax_near_misses_fail_closed(
    operand_types: tuple[str | None, ...],
    operand_literals: tuple[ClaimLiteral, ...],
    keywords: tuple[KeywordArg, ...],
) -> None:
    result = PLUGIN.claim(
        _call(
            "tensorflow.nn.softmax",
            operand_types,
            operand_literals=operand_literals,
            keywords=keywords,
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)


def test_rank1_softmax_rejects_non_float32_tensor_metadata() -> None:
    result = PLUGIN.claim(
        _call("tensorflow.nn.softmax", (TENSOR_I64_CPU_1D,)),
        CONFIG,
    )
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    "target",
    (
        "tensorflow.nn.softmax",
        "tensorflow.argmax",
        "tensorflow.reduce_mean",
        "tensorflow.reduce_sum",
    ),
)
def test_rejects_duplicate_literal_axis_metadata_at_claim(target: str) -> None:
    duplicate_axis = (
        KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
        KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
    )
    result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D,), keywords=duplicate_axis), CONFIG)
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    "keywords",
    (
        (),
        (
            KeywordArg(
                name="axis", arg_type="int", literal=ClaimLiteral(is_literal=False, value=None)
            ),
        ),
        (
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
            KeywordArg(
                name="keepdims", arg_type="bool", literal=ClaimLiteral(is_literal=False, value=None)
            ),
        ),
        (
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
            KeywordArg(
                name="keepdims", arg_type="bool", literal=ClaimLiteral(is_literal=True, value=False)
            ),
            KeywordArg(
                name="keepdims", arg_type="bool", literal=ClaimLiteral(is_literal=True, value=False)
            ),
        ),
        (
            KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
            KeywordArg(
                name="name", arg_type="str", literal=ClaimLiteral(is_literal=True, value="bad")
            ),
        ),
    ),
)
def test_reduce_sum_near_misses_remain_fallback(keywords: tuple[KeywordArg, ...]) -> None:
    result = PLUGIN.claim(
        _call("tensorflow.reduce_sum", (TENSOR_F32_CPU_2D,), keywords=keywords), CONFIG
    )
    assert isinstance(result, Rejected)


def test_reduce_sum_positional_axis_remains_fallback() -> None:
    result = PLUGIN.claim(_call("tensorflow.reduce_sum", (TENSOR_F32_CPU_2D, "int")), CONFIG)
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    ("operand_types", "operand_literals", "keywords"),
    (
        ((TENSOR_F32_CPU_2D, "int"), (), ()),
        (
            (TENSOR_F32_CPU_2D, "int"),
            (ClaimLiteral(is_literal=False),),
            (),
        ),
        (
            (TENSOR_F32_CPU_2D, "int"),
            (
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=False),
            ),
            (),
        ),
        (
            (TENSOR_F32_CPU_2D, "int"),
            (
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=True, value=True),
            ),
            (),
        ),
        (
            (TENSOR_F32_CPU_2D, "int"),
            (
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=True, value=1),
            ),
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
            ),
        ),
        (
            (TENSOR_F32_CPU_2D, "int", "bool"),
            (
                ClaimLiteral(is_literal=False),
                ClaimLiteral(is_literal=True, value=0),
                ClaimLiteral(is_literal=True, value=True),
            ),
            (),
        ),
    ),
)
def test_reduction_positional_axis_requires_exact_aligned_literal_metadata(
    operand_types: tuple[str | None, ...],
    operand_literals: tuple[ClaimLiteral, ...],
    keywords: tuple[KeywordArg, ...],
) -> None:
    result = PLUGIN.claim(
        _call(
            "tensorflow.reduce_mean",
            operand_types,
            operand_literals=operand_literals,
            keywords=keywords,
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    ("target", "axis"),
    (
        ("tensorflow.nn.softmax", 0),
        ("tf.nn.softmax", 0),
    ),
)
def test_softmax_axis0_remains_explicit_fallback(target: str, axis: int) -> None:
    result = PLUGIN.claim(
        _call(
            target,
            (TENSOR_F32_CPU_2D,),
            keywords=(
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=axis),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)
    assert "last-axis-only" in result.diagnostic.message


@pytest.mark.parametrize("target", ("tensorflow.argmax", "tensorflow.nn.softmax"))
def test_classification_rejects_forged_keepdims_keyword(target: str) -> None:
    result = PLUGIN.claim(
        _call(
            target,
            (TENSOR_F32_CPU_2D,),
            keywords=(
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
                KeywordArg(
                    name="keepdims",
                    arg_type="bool",
                    literal=ClaimLiteral(is_literal=True, value=True),
                ),
            ),
        ),
        CONFIG,
    )
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    ("target", "keywords"),
    (
        (
            "tensorflow.reduce_mean",
            (
                KeywordArg(
                    name="axis",
                    arg_type=TENSOR_F32_CPU_1D,
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
        ),
        (
            "tensorflow.reduce_sum",
            (
                KeywordArg(
                    name="axis",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
                KeywordArg(
                    name="keepdims",
                    arg_type="int",
                    literal=ClaimLiteral(is_literal=True, value=True),
                ),
            ),
        ),
        (
            "tensorflow.argmax",
            (
                KeywordArg(
                    name="axis",
                    arg_type=TENSOR_F32_CPU_1D,
                    literal=ClaimLiteral(is_literal=True, value=0),
                ),
            ),
        ),
        (
            "tensorflow.nn.softmax",
            (
                KeywordArg(
                    name="axis",
                    arg_type="bool",
                    literal=ClaimLiteral(is_literal=True, value=1),
                ),
            ),
        ),
    ),
)
def test_axis_keyword_arg_type_literal_contradictions_are_rejected(
    target: str, keywords: tuple[KeywordArg, ...]
) -> None:
    result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D,), keywords=keywords), CONFIG)
    assert isinstance(result, Rejected)


@pytest.mark.parametrize(
    ("target", "keywords"),
    (
        ("tensorflow.nn.softmax", ()),
        (
            "tensorflow.nn.softmax",
            (
                KeywordArg(
                    name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=0)
                ),
            ),
        ),
        (
            "tensorflow.argmax",
            (
                KeywordArg(
                    name="axis", arg_type="int", literal=ClaimLiteral(is_literal=False, value=None)
                ),
            ),
        ),
        (
            "tensorflow.argmax",
            (
                KeywordArg(
                    name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)
                ),
                KeywordArg(
                    name="output_type",
                    arg_type="dtype",
                    literal=ClaimLiteral(is_literal=True, value="int32"),
                ),
            ),
        ),
    ),
)
def test_classification_head_near_misses_remain_fallback(
    target: str, keywords: tuple[KeywordArg, ...]
) -> None:
    result = PLUGIN.claim(_call(target, (TENSOR_F32_CPU_2D,), keywords=keywords), CONFIG)
    assert isinstance(result, Rejected)


def test_rejects_matmul_wrong_rank() -> None:
    result = PLUGIN.claim(
        _call("tensorflow.matmul", (TENSOR_F32_CPU_1D, TENSOR_F32_CPU_2D)),
        CONFIG,
    )
    assert isinstance(result, Rejected)


def test_not_covered_unknown_call() -> None:
    result = PLUGIN.claim(_call("tensorflow.cos", (TENSOR_F32_CPU_2D,)), CONFIG)
    assert isinstance(result, NotCovered)
