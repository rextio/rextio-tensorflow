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

from rextio_tensorflow.claim.activations import RELU_RULE, SIGMOID_RULE
from rextio_tensorflow.claim.add import ADD_BINOP_RULE, ADD_CALL_RULE
from rextio_tensorflow.claim.classification import ARGMAX_RULE, SOFTMAX_RULE
from rextio_tensorflow.claim.matmul import MATMUL_RULE
from rextio_tensorflow.claim.reductions import MEAN_RULE
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
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operands,
        file_path="",
        line=0,
        column=0,
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


def test_claims_relu() -> None:
    result = PLUGIN.claim(_call("tensorflow.nn.relu", (TENSOR_F32_CPU_2D,)), CONFIG)
    assert result == Claimed(rule_id=RELU_RULE, result_type=TENSOR_F32_CPU_2D)


def test_claims_sigmoid() -> None:
    result = PLUGIN.claim(_call("tensorflow.nn.sigmoid", (TENSOR_F32_CPU_2D,)), CONFIG)
    assert result == Claimed(rule_id=SIGMOID_RULE, result_type=TENSOR_F32_CPU_2D)


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


def test_claims_classification_head_steps() -> None:
    axis = (
        KeywordArg(name="axis", arg_type="int", literal=ClaimLiteral(is_literal=True, value=1)),
    )
    softmax = PLUGIN.claim(_call("tensorflow.nn.softmax", (TENSOR_F32_CPU_2D,), keywords=axis), CONFIG)
    assert softmax == Claimed(rule_id=SOFTMAX_RULE, result_type=TENSOR_F32_CPU_2D)
    argmax = PLUGIN.claim(_call("tensorflow.argmax", (TENSOR_F32_CPU_2D,), keywords=axis), CONFIG)
    assert argmax == Claimed(rule_id=ARGMAX_RULE, result_type=TENSOR_I64_CPU_1D)


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
