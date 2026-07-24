"""Contracts for the bounded, non-certifying TensorFlow CUDA E3 candidate."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rextio.config.schema import RextioConfig
from rextio.devices import DeviceLoweringAuthorization, derive_device_requirements
from rextio.plugins.api import (
    ClaimLiteral,
    Claimed,
    ClaimSite,
    KeywordArg,
    LoweringContext,
    Rejected,
)

from rextio_tensorflow.claim.cuda import (
    CUDA_BIAS_ADD_RULE,
    CUDA_MATMUL_RULE,
    CUDA_MEAN_AXIS1_RULE,
    CUDA_RELU_RULE,
)
from rextio_tensorflow.diagnostics import (
    TENSOR_F32_CPU_2D,
    TENSOR_F32_CUDA0_1D,
    TENSOR_F32_CUDA0_2D,
)
from rextio_tensorflow.lower.cuda import CUDA_CAPABILITY_ID, CUDA_PROVIDER_ID
from rextio_tensorflow.plugin import plugin
from rextio_tensorflow.plugin_types import CUDA_RUNTIME_REQUIREMENTS, plugin_type
from rextio_tensorflow.rust_snippets.cuda_runtime import cuda_runtime_module_helpers
from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

PLUGIN = plugin()
CONFIG = RextioConfig()


def _site(
    target: str,
    operands: tuple[str, ...],
    *,
    keywords: tuple[KeywordArg, ...] = (),
) -> ClaimSite:
    return ClaimSite(
        kind="call",
        target=target,
        operand_types=operands,
        keywords=keywords,
        file_path="",
        line=0,
        column=0,
    )


def _keyword(name: str, arg_type: str, value: object) -> KeywordArg:
    return KeywordArg(
        name=name,
        arg_type=arg_type,
        literal=ClaimLiteral(is_literal=True, value=value),
    )


def _authorization(
    *,
    provider_id: str = CUDA_PROVIDER_ID,
    capability_id: str = CUDA_CAPABILITY_ID,
    runtime: str = "tensorflow-tfe",
    features: tuple[str, ...] = ("eager", "inference", "no-grad"),
    layouts: tuple[str, ...] = ("dense",),
    memory_spaces: tuple[str, ...] = ("device",),
) -> DeviceLoweringAuthorization:
    return DeviceLoweringAuthorization(
        provider_id=provider_id,
        capability_id=capability_id,
        logical_device="cuda:0",
        backend="cuda",
        runtime=runtime,
        reuse_domain_runtime=True,
        features=features,
        layouts=layouts,
        memory_spaces=memory_spaces,
        artifact_profile_sha256="0" * 64,
    )


def _ctx(
    operands: tuple[str, ...],
    authorization: DeviceLoweringAuthorization | None = None,
    *,
    backend: str = "pyo3",
) -> LoweringContext:
    return LoweringContext(
        operands=operands,
        target_language="rust",
        fresh_name=lambda prefix: f"{prefix}_0",
        device_authorization=authorization,
        backend=backend,
    )


def test_cuda_types_have_exact_api_16_metadata_and_distinct_boundary_type() -> None:
    values = []
    for key, rank in (
        (TENSOR_F32_CUDA0_2D, 2),
        (TENSOR_F32_CUDA0_1D, 1),
    ):
        item = plugin_type(key)
        metadata = item.device_value_metadata
        assert metadata is not None
        values.append(metadata)
        assert metadata.logical_device == "gpu:0"
        assert metadata.backend == "cuda"
        assert metadata.dtype == "float32"
        assert metadata.rank == rank
        assert metadata.layout == "dense"
        assert metadata.runtime == "tensorflow-tfe"
        assert metadata.runtime_version == "2.21.0"
        assert metadata.reuse_domain_runtime is True
        assert metadata.features == ("eager", "inference", "no-grad")
        assert metadata.memory_spaces == ("device",)
        assert set(metadata.runtime_requirements) == set(CUDA_RUNTIME_REQUIREMENTS)
        assert item.rust_type == "rextio_tensorflow_cuda_runtime::RxtTfCudaTensor"
        assert item.conversion is not None
        assert f"extract_f32_cuda0_{rank}d" in item.conversion.param_expr
        assert f"materialize_f32_cuda0_{rank}d" in item.conversion.return_expr
    requirements = derive_device_requirements(tuple(values))
    assert len(requirements) == 1
    assert requirements[0].logical_device == "gpu:0"
    assert requirements[0].runtime == "tensorflow-tfe"


def test_claims_exact_cuda_vertical_slice() -> None:
    cases = (
        (
            _site(
                "tf.matmul",
                (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D),
            ),
            CUDA_MATMUL_RULE,
            TENSOR_F32_CUDA0_2D,
        ),
        (
            _site(
                "tf.nn.bias_add",
                (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D),
                keywords=(_keyword("data_format", "str", "NHWC"),),
            ),
            CUDA_BIAS_ADD_RULE,
            TENSOR_F32_CUDA0_2D,
        ),
        (
            _site("tf.nn.relu", (TENSOR_F32_CUDA0_2D,)),
            CUDA_RELU_RULE,
            TENSOR_F32_CUDA0_2D,
        ),
        (
            _site(
                "tf.reduce_mean",
                (TENSOR_F32_CUDA0_2D,),
                keywords=(_keyword("axis", "int", 1),),
            ),
            CUDA_MEAN_AXIS1_RULE,
            TENSOR_F32_CUDA0_1D,
        ),
    )
    for site, rule, result_type in cases:
        assert PLUGIN.claim(site, CONFIG) == Claimed(rule, result_type)


@pytest.mark.parametrize(
    "site",
    (
        _site("tf.matmul", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CPU_2D)),
        _site("tf.linalg.matmul", (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D)),
        _site(
            "tf.nn.bias_add",
            (TENSOR_F32_CUDA0_1D, TENSOR_F32_CUDA0_2D),
        ),
        _site(
            "tf.nn.bias_add",
            (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_1D),
            keywords=(_keyword("data_format", "str", "NCHW"),),
        ),
        _site("tf.nn.relu", (TENSOR_F32_CUDA0_1D,)),
        _site(
            "tf.reduce_mean",
            (TENSOR_F32_CUDA0_2D,),
            keywords=(_keyword("axis", "int", 0),),
        ),
        _site(
            "tf.reduce_mean",
            (TENSOR_F32_CUDA0_2D,),
            keywords=(
                _keyword("axis", "int", 1),
                _keyword("keepdims", "bool", True),
            ),
        ),
    ),
)
def test_cuda_sites_outside_slice_fail_closed_before_cpu_lanes(site: ClaimSite) -> None:
    assert isinstance(PLUGIN.claim(site, CONFIG), Rejected)


def test_cuda_lower_requires_exact_authorization_and_revalidates_claim() -> None:
    site = _site(
        "tf.matmul",
        (TENSOR_F32_CUDA0_2D, TENSOR_F32_CUDA0_2D),
    )
    claimed = PLUGIN.claim(site, CONFIG)
    assert isinstance(claimed, Claimed)
    with pytest.raises(ValueError, match="exact PyO3 authorization"):
        PLUGIN.lower(
            replace(
                site,
                rule_id=claimed.rule_id,
                result_type=claimed.result_type,
            ),
            _ctx(("x", "w")),
        )

    claimed_site = replace(
        site,
        rule_id=claimed.rule_id,
        result_type=claimed.result_type,
    )
    lowered = PLUGIN.lower(claimed_site, _ctx(("x", "w"), _authorization()))
    assert lowered.rust == "rextio_tensorflow_cuda_runtime::matmul(&x, &w)?"
    assert lowered.helpers == (cuda_runtime_module_helpers(),)
    with pytest.raises(ValueError, match="exact PyO3 authorization"):
        PLUGIN.lower(
            claimed_site,
            _ctx(
                ("x", "w"),
                _authorization(capability_id="cuda-libtorch-linux-x86_64"),
            ),
        )


def test_cuda_runtime_is_separate_and_contains_exact_safety_anchors() -> None:
    cpu = runtime_module_helpers()
    helper = cuda_runtime_module_helpers()
    assert "rextio_tensorflow_cuda_runtime" not in cpu
    assert "RxtTfCudaTensor" not in cpu
    assert "mod rextio_tensorflow_cuda_runtime" in helper
    assert "pub struct RxtTfCudaTensor" in helper
    for symbol in (
        "TFE_ContextListDevices",
        "TF_DeleteDeviceList",
        "TF_DeviceListCount",
        "TF_DeviceListName",
        "TF_DeviceListType",
        "TFE_Py_TapeSetPossibleGradientTypes",
        "TFE_TensorHandleCopySharingTensor",
        "TFE_TensorHandleBackingDeviceName",
        "TFE_OpSetDevice",
        "TFE_Execute",
    ):
        assert symbol in helper
    assert 'device_type == "GPU"' in helper
    assert 'name.ends_with("/device:GPU:0")' in helper
    assert "if actual != expected" in helper
    assert "possible != 0" in helper
    assert "TFE_TensorHandleResolve" not in helper
    assert "TFE_TensorHandleCopyToDevice" not in helper
    assert ".numpy()" not in helper
