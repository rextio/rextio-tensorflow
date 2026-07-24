"""Generate and compile the GPU-free TensorFlow CUDA E3 candidate.

This harness uses real Core orchestration and the real CUDA provider with one
deterministic synthetic probe.  It generates and links one cdylib but never
imports TensorFlow and never loads or executes the resulting extension.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from rextio.analyzer.project_scanner import analyze_project
from rextio.build.orchestrator import generate_source_artifact
from rextio.config.schema import PluginConfig, RextioConfig
from rextio.devices import (
    DEVICE_PROVIDER_ENTRY_POINT,
    DeviceProviderOptions,
    DeviceProviderSelection,
)
from rextio.plugins.loader import load_plugin_registry
from rextio.targets.models import TargetSpec
from rextio.targets.plan import TargetPlan

from rextio_tensorflow.plugin import PLUGIN_ID, plugin

PROVIDER_ID = "rextio-device-cuda"
CAPABILITY_ID = "cuda-tensorflow-tfe-linux-x86_64"
TARGET = "x86_64-unknown-linux-gnu"
E3_RULES = (
    "rextio-tensorflow/cuda0-matmul-f32-2d",
    "rextio-tensorflow/cuda0-bias-add-nhwc-f32-2d-1d",
    "rextio-tensorflow/cuda0-relu-f32-2d",
    "rextio-tensorflow/cuda0-reduce-mean-axis1-f32-2d",
)

KERNELS = """\
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cuda0_1D, TensorF32Cuda0_2D


def inference(
    x: TensorF32Cuda0_2D,
    weight: TensorF32Cuda0_2D,
    bias: TensorF32Cuda0_1D,
) -> TensorF32Cuda0_1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
"""


@dataclass
class FixedRunner:
    """Return a reviewed synthetic report and count calls."""

    report: object
    calls: int = 0

    def run(self) -> object:
        """Return the fixed report."""
        self.calls += 1
        return self.report


@dataclass(frozen=True)
class _Distribution:
    name: str = "rextio-device-cuda"
    version: str = "0.1.0"


class _DeviceEntryPoint:
    group = DEVICE_PROVIDER_ENTRY_POINT
    name = PROVIDER_ID
    value = "rextio_device_cuda.provider:provider"
    dist = _Distribution()

    def __init__(self, provider: object) -> None:
        self._provider = provider

    def load(self) -> object:
        return self._provider


class _PluginEntryPoint:
    name = PLUGIN_ID

    def load(self):
        return plugin


def _probe_report() -> object:
    from rextio_device_cuda.probe import (
        CudaDeviceRecord,
        CudaProbeReport,
        ProbeTarget,
    )

    return CudaProbeReport(
        target=ProbeTarget(os="linux", arch="x86_64", environment="gnu"),
        platform_supported=True,
        status="probe-complete",
        reason_code=None,
        driver_loaded=True,
        driver_version=12_080,
        device_count=1,
        cuda_result=0,
        devices=(
            CudaDeviceRecord(
                ordinal=0,
                name="Synthetic NVIDIA Build-Only GPU",
                compute_major=8,
                compute_minor=0,
                sm="sm_80",
            ),
        ),
    )


def _write_fixture(root: Path) -> None:
    package = root / "src" / "cuda_app"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")


def _assert_orchestration(result: object, runner: FixedRunner) -> Path:
    if runner.calls != 1:
        raise RuntimeError(f"expected one synthetic probe, got {runner.calls}")
    [profile] = result.plan.artifact_profiles
    if profile.target_triple != TARGET:
        raise RuntimeError(f"unexpected target: {profile.target_triple}")
    [device] = profile.device_requirements
    if device.to_dict() != {
        "logical_device": "gpu:0",
        "backend": "cuda",
        "runtime": "tensorflow-tfe",
        "features": ["eager", "inference", "no-grad"],
        "layouts": ["dense"],
        "memory_spaces": ["device"],
        "architectures": ["sm_80"],
        "reuse_domain_runtime": True,
    }:
        raise RuntimeError(f"unexpected device requirement: {device.to_dict()}")
    runtime_rows = {
        (item.name, item.version, item.features)
        for item in profile.runtime_requirements
    }
    required = {
        ("tensorflow", "2.21.0", ("cuda", "python-wheel", "tfe-c-api")),
        ("cpython", "3.11", ("private-eager-abi",)),
    }
    if not required.issubset(runtime_rows):
        raise RuntimeError(f"runtime pins missing: {runtime_rows}")

    [provider_plan] = result.device_provider_plans
    authorization = provider_plan["lowering_authorization"]
    if (
        authorization["provider_id"] != PROVIDER_ID
        or authorization["capability_id"] != CAPABILITY_ID
        or authorization["logical_device"] != "gpu:0"
        or authorization["runtime"] != "tensorflow-tfe"
        or authorization["features"] != ["eager", "inference", "no-grad"]
        or authorization["layouts"] != ["dense"]
        or authorization["memory_spaces"] != ["device"]
    ):
        raise RuntimeError(f"unexpected authorization: {authorization}")
    if not re.fullmatch(r"[0-9a-f]{64}", authorization["artifact_profile_sha256"]):
        raise RuntimeError("authorization profile hash missing")
    if (
        authorization["artifact_profile_sha256"]
        != provider_plan["lock"]["artifact_profile_sha256"]
    ):
        raise RuntimeError("authorization/profile lock mismatch")
    report = provider_plan["report"]
    if report["support_claim"] is not False or report["certification_tier"] != "build-only":
        raise RuntimeError(f"provider overclaim: {report}")

    contribution = provider_plan["contribution"]
    resources = contribution["resource_contracts"]
    if {item["resource_kind"] for item in resources} != {
        "framework.tensor",
        "framework.eager-context",
    }:
        raise RuntimeError(f"unexpected resource contracts: {resources}")
    if any(
        item["owner"] != "framework"
        or item["access"] != "borrow-validate"
        or item["may_allocate"]
        or item["may_replace"]
        or item["may_synchronize"]
        for item in resources
    ):
        raise RuntimeError(f"resource contract overclaim: {resources}")

    rust_dir = result.layout.rust_dir
    rust = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8")
    required_tokens = (
        "mod rextio_tensorflow_cuda_runtime",
        "RxtTfCudaTensor",
        "TFE_ContextListDevices",
        "TFE_Py_TapeSetPossibleGradientTypes",
        "rextio_tensorflow_cuda_runtime::matmul",
        "rextio_tensorflow_cuda_runtime::bias_add",
        "rextio_tensorflow_cuda_runtime::relu",
        "rextio_tensorflow_cuda_runtime::reduce_mean_axis1",
    )
    for token in required_tokens:
        if token not in rust:
            raise RuntimeError(f"generated Rust omitted {token}")
    forbidden = (
        "TFE_TensorHandleResolve",
        "TFE_TensorHandleCopyToDevice",
        ".numpy()",
    )
    if any(token in rust for token in forbidden):
        raise RuntimeError("generated Rust contains a forbidden transfer/resolve path")
    return rust_dir


def generate_candidate(root: Path) -> Path:
    """Run analyzer, provider preflight, authorization, and Core codegen."""
    from rextio_device_cuda.config import CudaProviderConfig
    from rextio_device_cuda.provider import CudaDeviceProvider

    _write_fixture(root)
    config = RextioConfig()
    registry = load_plugin_registry(
        PluginConfig(enabled=(PLUGIN_ID,)),
        TargetSpec(),
        entry_points=(_PluginEntryPoint(),),
        full_config=config,
    )
    analysis = analyze_project(
        root,
        active_plugins=registry.active,
        plugin_registry=registry,
        plugin_config=config,
    )
    [function] = analysis.accepted_native_functions
    if tuple(claim.rule_id for claim in function.plugin_claims) != E3_RULES:
        raise RuntimeError("analyzer did not accept the exact CUDA E3 chain")

    runner = FixedRunner(_probe_report())
    provider = CudaDeviceProvider(CudaProviderConfig(), probe_runner=runner)
    result = generate_source_artifact(
        root,
        analysis,
        "cpython",
        target_plan=TargetPlan(TargetSpec(), registry),
        device_selection=DeviceProviderSelection(PROVIDER_ID, CAPABILITY_ID),
        device_options=DeviceProviderOptions(
            values=(("device_ordinal", "0"), ("sm", "sm_80"))
        ),
        device_entry_points=(_DeviceEntryPoint(provider),),
    )
    if result.native_source.status != "generated":
        raise RuntimeError(f"Core source generation failed: {result.native_source}")
    return _assert_orchestration(result, runner)


def main() -> int:
    """Generate once, build once, and never load or execute the artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("CUDA E3 build candidate requires CPython 3.11")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise SystemExit("CUDA E3 build candidate requires Linux x86_64")
    if any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules):
        raise SystemExit("hosted build-only harness must not import TensorFlow")

    root = args.output.resolve()
    rust_dir = generate_candidate(root)
    environment = dict(os.environ)
    environment["RUSTUP_TOOLCHAIN"] = "1.93.1"
    completed = subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")],
        check=False,
        env=environment,
    )
    if completed.returncode != 0:
        return completed.returncode
    linked = tuple((rust_dir / "target" / "release").glob("*_rextio_native*.so"))
    if len(linked) != 1 or linked[0].stat().st_size == 0:
        raise RuntimeError(f"expected one nonempty linked cdylib: {linked}")
    if any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules):
        raise RuntimeError("TensorFlow was imported during the build-only harness")
    print(
        json.dumps(
            {
                "cargo_builds": 1,
                "cuda_executed": False,
                "extension_loaded": False,
                "linked_cdylib": linked[0].name,
                "support_claim": False,
                "synthetic_probe": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
