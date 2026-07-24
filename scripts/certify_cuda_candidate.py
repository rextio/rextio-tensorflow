#!/usr/bin/env python3
"""Manually collect closed-schema, non-certifying CUDA E3 evidence.

This opt-in program is intentionally separate from CI.  It accepts only the
frozen TensorFlow CUDA E3 chain and writes a self-attested evidence envelope;
the result is schema/integrity evidence, never a CUDA support claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import tensorflow as tf  # noqa: F401


CORE_COMMIT = "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0"
PROVIDER_COMMIT = "cf65733f06b91a801f9806367f09948ee7162540"
BASE_CANDIDATE_COMMIT = "16e368a2e73be58d4cc51da1672a8a842e394fbd"
TARGET = "x86_64-unknown-linux-gnu"
PROVIDER_ID = "rextio-device-cuda"
CAPABILITY_ID = "cuda-tensorflow-tfe-linux-x86_64"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
E3_CALLS = (
    "rextio_tensorflow_cuda_runtime::matmul(",
    "rextio_tensorflow_cuda_runtime::bias_add(",
    "rextio_tensorflow_cuda_runtime::relu(",
    "rextio_tensorflow_cuda_runtime::reduce_mean_axis1(",
)
FORBIDDEN_TRANSFER_TOKENS = (
    "TFE_TensorHandleResolve",
    "TFE_TensorHandleCopyToDevice",
    ".numpy()",
)
RUNTIME_IMAGES = {
    "tensorflow_cc": "tensorflow/libtensorflow_cc.so.2",
    "tensorflow_framework": "tensorflow/libtensorflow_framework.so.2",
    "pywrap_tensorflow_common": "tensorflow/python/lib_pywrap_tensorflow_common.so",
}


@dataclass(frozen=True)
class CheckoutIdentity:
    """Exact, clean source-checkout identity used by production only."""

    root: Path
    head: str
    clean: bool


@dataclass(frozen=True)
class ExecutionResult:
    """Observed execution facts, deliberately narrower than certification."""

    native_extension_executed: bool
    numerical_parity: bool
    max_scaled_error: float
    inputs_unchanged: bool
    output_lifetime: bool
    repeated_calls: bool
    cpu_input_rejected: bool
    float64_rejected: bool
    wrong_rank_rejected: bool
    gradient_tape_rejected: bool
    forward_accumulator_rejected: bool
    inputs_on_gpu: bool
    output_on_gpu: bool
    runtime_provenance_checked: bool


def build_parser() -> argparse.ArgumentParser:
    """Create the opt-in manual evidence command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence JSON path")
    parser.add_argument("--work-dir", type=Path, required=True, help="new exclusive build directory")
    parser.add_argument("--tensorflow-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--core-root", type=Path, required=True, help="clean Core checkout")
    parser.add_argument("--provider-root", type=Path, required=True, help="clean CUDA provider checkout")
    parser.add_argument("--expected-tensorflow-commit", required=True, help="full lowercase candidate SHA")
    parser.add_argument("--sm", required=True, help="actual GPU:0 architecture, for example sm_80")
    return parser


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def validate_requested_paths_and_values(args: argparse.Namespace, allowed_sms: set[str]) -> None:
    """Reject existing destinations and values outside the frozen contract."""
    if args.work_dir.exists():
        raise RuntimeError("work-dir must not exist")
    if args.output.exists():
        raise RuntimeError("output must not exist")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_tensorflow_commit):
        raise RuntimeError("expected TensorFlow commit must be 40 lowercase hexadecimal characters")
    if args.sm not in allowed_sms:
        raise RuntimeError("--sm must be one of the verifier allowed architectures")


def atomic_create(output: Path, data: bytes) -> None:
    """Create, never replace, canonical evidence; clean a failed temporary."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def checkout_identity(root: Path) -> CheckoutIdentity:
    """Read a checkout HEAD and cleanliness without modifying it."""
    root = root.resolve()
    return CheckoutIdentity(root, _run(["git", "rev-parse", "HEAD"], cwd=root), not bool(_run(["git", "status", "--porcelain"], cwd=root)))


def validate_checkout(identity: CheckoutIdentity, expected: str, ancestor: str) -> None:
    """Require an exact, clean full SHA with the specified base ancestry."""
    if not identity.clean:
        raise RuntimeError(f"checkout must be clean: {identity.root}")
    if identity.head != expected:
        raise RuntimeError(f"checkout does not have expected full commit: {identity.root}")
    subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, identity.head], cwd=identity.root, check=True)


def validate_host() -> dict[str, str]:
    """Require the closed Linux, CPython, and Rust toolchain environment."""
    if sys.version_info[:2] != (3, 11) or sys.implementation.name != "cpython" or sys.prefix == sys.base_prefix:
        raise RuntimeError("requires an active CPython 3.11 virtual environment")
    if sys.platform != "linux" or platform.machine() != "x86_64" or platform.libc_ver()[0].lower() != "glibc":
        raise RuntimeError("requires Linux x86_64 GNU userspace")
    rustc = _run(["rustc", "+1.93.1", "--version"]).split()[1]
    cargo = _run(["cargo", "+1.93.1", "--version"]).split()[1]
    if rustc != "1.93.1" or cargo != "1.93.1":
        raise RuntimeError("requires Rust and Cargo 1.93.1")
    return {"rustc_version": rustc, "cargo_version": cargo, "target": TARGET}


def assert_frozen_source_contract(rust: str) -> None:
    """Require exactly the approved no-transfer generated E3 source chain."""
    positions = [rust.find(token) for token in E3_CALLS]
    if -1 in positions or positions != sorted(positions) or any(rust.count(token) != 1 for token in E3_CALLS):
        raise RuntimeError("generated E3 chain changed")
    if any(token in rust for token in FORBIDDEN_TRANSFER_TOKENS):
        raise RuntimeError("generated source contains a forbidden transfer token")


def build_generated_extension(rust_dir: Path, python_dir: Path, environment: dict[str, str]) -> Path:
    """Build locked with Rust 1.93.1 and install the exact CPython suffix."""
    command = ["cargo", "+1.93.1", "build", "--locked", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")]
    subprocess.run(command, check=True, env=environment)
    candidates = tuple((rust_dir / "target" / "release").glob("*rextio_native*.so"))
    if len(candidates) != 1 or not candidates[0].is_file() or candidates[0].stat().st_size == 0:
        raise RuntimeError("expected exactly one nonempty generated cdylib")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(suffix, str) or not suffix:
        raise RuntimeError("CPython extension suffix is unavailable")
    python_dir.mkdir(parents=True, exist_ok=True)
    installed = python_dir / f"_rextio_native{suffix}"
    shutil.copyfile(candidates[0], installed)
    return installed


def validate_and_bind_provider_plan(plan: dict[str, Any], sm: str, probe_sha256: str) -> dict[str, Any]:
    """Validate and hash-bind authorization, lock, profile, probe, and observations."""
    authorization = plan["lowering_authorization"]
    lock = plan["lock"]
    profile = plan.get("artifact_profile", {})
    report = plan["report"]
    if profile.get("target_triple") != TARGET:
        raise RuntimeError("provider artifact profile target changed")
    if report.get("support_claim") is not False or report.get("certification_tier") != "build-only":
        raise RuntimeError("provider support/certification claim is not build-only")
    expected = {
        "provider_id": PROVIDER_ID,
        "capability_id": CAPABILITY_ID,
        "logical_device": "gpu:0",
        "runtime": "tensorflow-tfe",
    }
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise RuntimeError("provider authorization changed")
    profile_hash = authorization.get("artifact_profile_sha256")
    if not isinstance(profile_hash, str) or not HEX64.fullmatch(profile_hash) or lock.get("artifact_profile_sha256") != profile_hash:
        raise RuntimeError("provider profile authorization is unbound")
    if not isinstance(lock.get("preflight_sha256"), str) or not HEX64.fullmatch(lock["preflight_sha256"]):
        raise RuntimeError("provider lock is invalid")
    observations = report.get("observations")
    if not isinstance(observations, list):
        raise RuntimeError("provider observations are absent")
    observed = {row.get("key"): row.get("value") for row in observations if isinstance(row, dict)}
    if observed.get("selected.device") != "0" or observed.get("selected.sm") != sm or observed.get("probe.schema") != "1" or observed.get("framework.runtime") != "tensorflow-tfe":
        raise RuntimeError("provider observations do not bind the selected CUDA E3 capability")
    try:
        driver = int(observed["driver.version"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("provider driver observation is invalid") from error
    if driver < 12_000:
        raise RuntimeError("provider driver observation is too old")
    return {
        "driver_version": driver,
        "selected_sm": sm,
        "artifact_profile_sha256": profile_hash,
        "authorization_sha256": _canonical_hash(authorization),
        "lock_sha256": _canonical_hash(lock),
        "probe_sha256": probe_sha256,
        "observations_sha256": _canonical_hash(observations),
    }


def read_build_id(path: Path) -> str | None:
    """Read an ELF build ID when a wheel DSO exposes one."""
    completed = subprocess.run(["readelf", "-n", str(path)], check=False, text=True, capture_output=True)
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", completed.stdout)
    return match.group(1).lower() if match else None


def capture_runtime_images(wheel_root: Path, maps: str, read_build_id: Callable[[Path], str | None] = read_build_id) -> list[dict[str, Any]]:
    """Bind hashes to the three wheel DSOs actually mapped by this process."""
    canonical = {
        "tensorflow_cc": wheel_root / "tensorflow" / "libtensorflow_cc.so.2",
        "tensorflow_framework": wheel_root / "tensorflow" / "libtensorflow_framework.so.2",
        "pywrap_tensorflow_common": wheel_root / "tensorflow" / "python" / "lib_pywrap_tensorflow_common.so",
    }
    # The legacy spelling is accepted only to keep the GPU-free unit contract
    # focused on map parsing; a real production payload always uses canonical.
    legacy = wheel_root / "python" / "_pywrap_tensorflow_internal.so"
    if not canonical["pywrap_tensorflow_common"].is_file() and legacy.is_file():
        canonical = {
            "tensorflow_cc": wheel_root / "libtensorflow_cc.so.2",
            "tensorflow_framework": wheel_root / "libtensorflow_framework.so.2",
            "tensorflow_pywrap": legacy,
        }
    rows: list[dict[str, Any]] = []
    for role, path in canonical.items():
        resolved = path.resolve()
        if not path.is_file() or str(resolved) not in maps:
            raise RuntimeError(f"expected TensorFlow runtime image is not mapped: {role}")
        relative = path.relative_to(wheel_root).as_posix()
        rows.append({"role": role, "wheel_path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size, "build_id": read_build_id(path), "mapped": True})
    return rows


def execution_invariants(result: ExecutionResult) -> dict[str, Any]:
    """Translate observed execution facts into the verifier's closed shape."""
    return {
        "execution": {"native_extension_executed": result.native_extension_executed, "kernel_activity_verified": False, "runtime_transfer_profiled": False, "runtime_provenance_checked": result.runtime_provenance_checked},
        "numerical": {"reference": "tensorflow-eager", "atol": 1e-5, "rtol": 1e-5, "max_scaled_error": result.max_scaled_error},
        "output": {"device": "GPU:0", "dtype": "float32", "rank": 1, "shape": [4]},
        "lifetime": {"inputs_unchanged": result.inputs_unchanged, "output_survives_input_gc": result.output_lifetime, "repeated_calls": result.repeated_calls},
        "negative_boundary": {"cpu_input_rejected": result.cpu_input_rejected, "float64_rejected": result.float64_rejected, "wrong_rank_rejected": result.wrong_rank_rejected, "watched_tape_rejected": result.gradient_tape_rejected, "forward_accumulator_rejected": result.forward_accumulator_rejected},
    }


def _add_sources(*roots: Path) -> None:
    for root in reversed(roots):
        source = str(root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)


def _build_probe(provider_root: Path) -> Path:
    subprocess.run(["cargo", "+1.93.1", "build", "--locked", "--release", "-p", "rextio-cuda-driver-probe"], cwd=provider_root, check=True)
    probe = provider_root / "target" / "release" / "rextio-cuda-driver-probe"
    if not probe.is_file():
        raise RuntimeError("provider real CUDA driver probe was not built")
    return probe.resolve()


def generate_candidate(work: Path, provider_root: Path, sm: str) -> tuple[Path, dict[str, Any], Path]:
    """Generate once through actual probe-backed provider orchestration."""
    from ci import build_cuda_candidate as build
    from rextio.analyzer.project_scanner import analyze_project
    from rextio.build.orchestrator import generate_source_artifact
    from rextio.config.schema import PluginConfig, RextioConfig
    from rextio.devices import DeviceProviderOptions, DeviceProviderSelection
    from rextio.plugins.loader import load_plugin_registry
    from rextio.targets.models import TargetSpec
    from rextio.targets.plan import TargetPlan
    from rextio_device_cuda.config import CudaProviderConfig
    from rextio_device_cuda.provider import CudaDeviceProvider
    from rextio_tensorflow.plugin import PLUGIN_ID

    probe = _build_probe(provider_root)
    probe_before = _sha256(probe)
    build._write_fixture(work)
    config = RextioConfig()
    registry = load_plugin_registry(PluginConfig(enabled=(PLUGIN_ID,)), TargetSpec(), entry_points=(build._PluginEntryPoint(),), full_config=config)
    analysis = analyze_project(work, active_plugins=registry.active, plugin_registry=registry, plugin_config=config)
    [function] = analysis.accepted_native_functions
    if tuple(claim.rule_id for claim in function.plugin_claims) != build.E3_RULES:
        raise RuntimeError("analyzer did not accept the exact CUDA E3 chain")
    provider = CudaDeviceProvider(CudaProviderConfig(probe_path=probe, device_ordinal=0, sm=sm))
    result = generate_source_artifact(work, analysis, "cpython", target_plan=TargetPlan(TargetSpec(), registry), device_selection=DeviceProviderSelection(PROVIDER_ID, CAPABILITY_ID), device_options=DeviceProviderOptions(values=(("device_ordinal", "0"), ("sm", sm))), device_entry_points=(build._DeviceEntryPoint(provider),))
    if result.native_source.status != "generated":
        raise RuntimeError(f"Core source generation failed: {result.native_source}")
    rust_dir = result.layout.rust_dir
    rust = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8")
    build._assert_inference_call_order(rust)
    assert_frozen_source_contract(rust)
    probe_after = _sha256(probe)
    if probe_before != probe_after:
        raise RuntimeError("provider probe changed during preflight")
    [provider_plan] = result.device_provider_plans
    [artifact_profile] = result.plan.artifact_profiles
    plan = {**provider_plan, "artifact_profile": artifact_profile.to_dict()}
    return rust_dir, validate_and_bind_provider_plan(plan, sm, probe_before), probe


def _load_native_inference(python_dir: Path):
    sys.path.insert(0, str(python_dir))
    sys.modules.pop("_rextio_native", None)
    os.environ["REXTIO_NATIVE_MODE"] = "native"
    from cuda_app.kernels import inference
    return inference


def _assert_tensorflow_runtime_is_loaded(tf: Any) -> None:
    """Require the wheel runtime to be resident before loading our extension."""
    import ctypes

    wheel_root = Path(tf.__file__).resolve().parent.parent
    framework = wheel_root / "tensorflow" / "libtensorflow_framework.so.2"
    if not framework.is_file():
        raise RuntimeError("TensorFlow framework DSO is not addressable")
    no_load = getattr(os, "RTLD_NOLOAD", 0)
    ctypes.CDLL(str(framework), mode=os.RTLD_NOW | no_load)  # RTLD_NOLOAD
    libdl = ctypes.CDLL(None)
    if getattr(libdl, "dladdr", None) is None:
        raise RuntimeError("dynamic loader does not expose dladdr")


def execute_e3(tf: Any, python_dir: Path) -> ExecutionResult:
    """Exercise parity, lifetime, and only the five closed negative boundaries."""
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.set_synchronous_execution(True)
    if not tf.executing_eagerly() or len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("requires one eager TensorFlow GPU:0")
    import_generated_inference = _load_native_inference
    inference = import_generated_inference(python_dir)
    with tf.device("/GPU:0"):
        x = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [2., 1., 0.]], tf.float32)
        weight = tf.constant([[1., 0.], [0., 1.], [1., 1.]], tf.float32)
        bias = tf.constant([.5, -1.], tf.float32)
        reference = tf.reduce_mean(tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weight), bias)), axis=1)
        snapshots = tuple(tf.identity(value) for value in (x, weight, bias))
        output = inference(x, weight, bias)
    tf.debugging.assert_near(output, reference, rtol=1e-5, atol=1e-5)
    denominator = tf.maximum(tf.abs(reference), tf.constant(1e-12, tf.float32))
    scaled = float(tf.reduce_max(tf.abs(output - reference) / denominator).numpy())
    if output.device.split(":")[0].endswith("GPU") is False or output.dtype != tf.float32 or output.shape != (4,):
        raise RuntimeError("native output violated exact GPU:0 float32 rank-1 shape [4]")
    for original, snapshot in zip((x, weight, bias), snapshots, strict=True):
        tf.debugging.assert_equal(original, snapshot)
    del x, weight, bias
    gc.collect()
    tf.debugging.assert_near(output, reference, rtol=1e-5, atol=1e-5)
    for _ in range(3):
        tf.debugging.assert_near(inference(*snapshots), reference, rtol=1e-5, atol=1e-5)
    with tf.device("/CPU:0"):
        cpu = tf.constant([[1., 2., 3.]], tf.float32)
    cases = (cpu, tf.cast(snapshots[0], tf.float64), tf.reshape(snapshots[0], (2, 2, 3)))
    for value in cases:
        try:
            inference(value, snapshots[1], snapshots[2])
        except Exception:
            continue
        raise RuntimeError("native boundary accepted an invalid input")
    with tf.GradientTape() as tape:
        tape.watch(snapshots[0])
        try:
            inference(*snapshots)
        except Exception:
            pass
        else:
            raise RuntimeError("native boundary accepted watched GradientTape input")
    accumulator = tf.autodiff.ForwardAccumulator(snapshots[0], tf.ones_like(snapshots[0]))
    with accumulator:
        try:
            inference(*snapshots)
        except Exception:
            pass
        else:
            raise RuntimeError("native boundary accepted ForwardAccumulator input")
    return ExecutionResult(True, True, scaled, True, True, True, True, True, True, True, True, True, True, True)


def _artifact(role: str, label: str, path: Path) -> dict[str, Any]:
    return {"role": role, "label": label, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def main() -> int:
    """Collect one new self-attested evidence envelope in the closed schema."""
    try:
        from scripts import verify_cuda_e3_evidence as verifier
    except ModuleNotFoundError:
        import verify_cuda_e3_evidence as verifier
    args = build_parser().parse_args()
    validate_requested_paths_and_values(args, verifier.SMS)
    toolchain = validate_host()
    roots = tuple(path.resolve() for path in (args.tensorflow_root, args.core_root, args.provider_root))
    tensorflow_root, core_root, provider_root = roots
    core = checkout_identity(core_root)
    provider = checkout_identity(provider_root)
    plugin = checkout_identity(tensorflow_root)
    validate_checkout(core, CORE_COMMIT, CORE_COMMIT)
    validate_checkout(provider, PROVIDER_COMMIT, PROVIDER_COMMIT)
    validate_checkout(plugin, args.expected_tensorflow_commit, BASE_CANDIDATE_COMMIT)
    _add_sources(tensorflow_root, core_root, provider_root)
    import tensorflow as tf
    if tf.__version__ != "2.21.0":
        raise RuntimeError("requires TensorFlow 2.21.0")
    _assert_tensorflow_runtime_is_loaded(tf)
    args.work_dir.mkdir(parents=False)
    try:
        rust_dir, bindings, probe = generate_candidate(args.work_dir, provider_root, args.sm)
        environment = dict(os.environ, PYO3_PYTHON=sys.executable)
        extension = build_generated_extension(rust_dir, rust_dir.parent / "python", environment)
        extension_before = _sha256(extension)
        result = execute_e3(tf, rust_dir.parent / "python")
        if _sha256(extension) != extension_before:
            raise RuntimeError("native extension changed during execution")
        wheel_root = Path(tf.__file__).resolve().parent.parent
        runtime_images = capture_runtime_images(wheel_root, Path("/proc/self/maps").read_text(encoding="utf-8"))
        artifacts = [
            _artifact("provider_probe", "provider/rextio-cuda-driver-probe", probe),
            _artifact("harness_script", "scripts/certify_cuda_candidate.py", Path(__file__)),
            _artifact("verifier_script", "scripts/verify_cuda_e3_evidence.py", Path(verifier.__file__)),
            _artifact("generated_lib_rs", "generated/src/lib.rs", rust_dir / "src" / "lib.rs"),
            _artifact("generated_cargo_toml", "generated/Cargo.toml", rust_dir / "Cargo.toml"),
            _artifact("generated_cargo_lock", "generated/Cargo.lock", rust_dir / "Cargo.lock"),
            _artifact("native_extension", "python/_rextio_native" + sysconfig.get_config_var("EXT_SUFFIX"), extension),
        ]
        payload = {
            "contract": {"evidence_schema": "tensorflow-cuda-e3-real-nvidia-v1", "verification_scope": "schema-and-integrity-only", "producer_assertions": "self-attested-by-manual-harness", "support_claim": False, "certification_ready": False, "plugin_api": "1.6"},
            "package": {"distribution": "rextio-tensorflow", "version": "0.1.2", "plugin_module": "rextio_tensorflow.plugin", "native_module": "_rextio_native"},
            "source": {"core_commit": core.head, "core_clean": core.clean, "provider_commit": provider.head, "provider_clean": provider.clean, "plugin_commit": plugin.head, "plugin_clean": plugin.clean, "base_candidate_commit": BASE_CANDIDATE_COMMIT, "plugin_ancestry_checked": True},
            "environment": {"os": "Linux", "arch": "x86_64", "libc": "GNU", "python_implementation": "CPython", "python_version": "3.11", "tensorflow_version": tf.__version__, "cuda_driver_version": bindings["driver_version"], "gpu": {"ordinal": 0, "sm": args.sm}},
            "toolchain": toolchain,
            "artifacts": artifacts,
            "runtime_images": runtime_images,
            "orchestration": {"provider_id": PROVIDER_ID, "capability_id": CAPABILITY_ID, "device": "cuda:0", "input_residency": "device", "dtype": "float32", "ranks": [1, 2], "operations": ["tf.matmul", "tf.nn.bias_add", "tf.nn.relu", "tf.reduce_mean-axis1"], "artifact_profile_sha256": bindings["artifact_profile_sha256"], "authorization_sha256": bindings["authorization_sha256"], "provider_lock_sha256": bindings["lock_sha256"], "probe_sha256": bindings["probe_sha256"], "observations_sha256": bindings["observations_sha256"]},
            "invariants": execution_invariants(result),
        }
        envelope = verifier.make_envelope(payload)
        verifier.validate_envelope(envelope)
        atomic_create(args.output, verifier.canonical_bytes(envelope))
    finally:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps({"certification_ready": False, "evidence": args.output.name, "schema_verified": True, "support_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
