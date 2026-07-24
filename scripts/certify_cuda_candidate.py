"""Opt-in, manual real-NVIDIA execution evidence for TensorFlow CUDA E3.

This is deliberately not a CI program.  It executes only the frozen
``matmul -> bias_add -> relu -> mean(axis=1)`` E3 slice on an already-resident
``GPU:0`` TensorFlow wheel tensor and records evidence without making a CUDA
support or certification claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CORE_COMMIT = "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0"
PROVIDER_COMMIT = "cf65733f06b91a801f9806367f09948ee7162540"
BASE_CANDIDATE_COMMIT = "16e368a"
TARGET = "x86_64-unknown-linux-gnu"
PROVIDER_ID = "rextio-device-cuda"
CAPABILITY_ID = "cuda-tensorflow-tfe-linux-x86_64"
E3_RUST_CALLS = (
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


@dataclass(frozen=True)
class CheckoutIdentity:
    """Minimal immutable identity for a source checkout."""

    root: Path
    head: str
    dirty: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit manual real-NVIDIA command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="evidence JSON destination")
    parser.add_argument("--work-dir", type=Path, help="empty-or-new isolated build directory")
    parser.add_argument("--tensorflow-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--core-root", type=Path, required=True, help="clean Core checkout")
    parser.add_argument("--provider-root", type=Path, required=True, help="clean CUDA provider checkout")
    parser.add_argument("--expected-tensorflow-commit", required=True, help="full candidate commit")
    parser.add_argument("--sm", required=True, help="actual GPU:0 architecture, e.g. sm_80")
    return parser


def _run(args: list[str], *, cwd: Path | None = None) -> str:
    """Run one checked command and return stripped standard output."""
    completed = subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def checkout_identity(root: Path) -> CheckoutIdentity:
    """Read the exact HEAD and worktree cleanliness for one checkout."""
    root = root.resolve()
    return CheckoutIdentity(root, _run(["git", "rev-parse", "HEAD"], cwd=root), bool(_run(["git", "status", "--porcelain"], cwd=root)))


def validate_checkout(identity: CheckoutIdentity, *, expected: str, required_ancestor: str) -> None:
    """Require a clean exact checkout whose full head descends from the base."""
    if identity.dirty:
        raise RuntimeError(f"checkout must be clean: {identity.root}")
    if len(expected) != 40 or identity.head != expected:
        raise RuntimeError(f"checkout HEAD does not match expected full commit: {identity.root}")
    ancestor = _run(["git", "merge-base", "--is-ancestor", required_ancestor, identity.head], cwd=identity.root)
    if ancestor != "":  # git emits no output; retained for mocked runners.
        raise RuntimeError("candidate is not descended from the required base")


def assert_frozen_source_contract(rust: str) -> None:
    """Require one ordered E3 chain and prohibit host-transfer primitives."""
    positions = [rust.find(token) for token in E3_RUST_CALLS]
    if -1 in positions or positions != sorted(positions) or any(rust.count(token) != 1 for token in E3_RUST_CALLS):
        raise RuntimeError("generated E3 chain changed")
    if any(token in rust for token in FORBIDDEN_TRANSFER_TOKENS):
        raise RuntimeError("generated source contains a forbidden transfer token")


def atomic_write_json(output: Path, payload: dict[str, Any]) -> None:
    """Atomically replace evidence, removing the temporary file on every failure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_host() -> None:
    if sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix:
        raise RuntimeError("requires an active CPython 3.11 virtual environment")
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("requires Linux x86_64")
    if platform.libc_ver()[0].lower() != "glibc":
        raise RuntimeError("requires Linux GNU userspace")
    if _run(["rustc", "--version"]).split()[1] != "1.93.1":
        raise RuntimeError("requires rustc 1.93.1")
    if _run(["cargo", "--version"]).split()[1] != "1.93.1":
        raise RuntimeError("requires cargo 1.93.1")


def _build_probe(provider_root: Path) -> Path:
    _run(["cargo", "build", "--release", "-p", "rextio-cuda-driver-probe"], cwd=provider_root)
    probe = provider_root / "target" / "release" / "rextio-cuda-driver-probe"
    if not probe.is_file():
        raise RuntimeError("provider did not build its real CUDA driver probe")
    return probe.resolve()


def _add_sources(*roots: Path) -> None:
    for root in reversed(roots):
        source = str(root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)


def _generate(work: Path, provider_root: Path, sm: str) -> tuple[Path, dict[str, Any]]:
    """Run real probe-backed provider preflight, authorization, and Core codegen."""
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
    build._write_fixture(work)
    config = RextioConfig()
    registry = load_plugin_registry(PluginConfig(enabled=(PLUGIN_ID,)), TargetSpec(), entry_points=(build._PluginEntryPoint(),), full_config=config)
    analysis = analyze_project(work, active_plugins=registry.active, plugin_registry=registry, plugin_config=config)
    [function] = analysis.accepted_native_functions
    if tuple(claim.rule_id for claim in function.plugin_claims) != build.E3_RULES:
        raise RuntimeError("analyzer did not accept the exact CUDA E3 chain")
    provider = CudaDeviceProvider(CudaProviderConfig(probe_path=probe, device_ordinal=0, sm=sm))
    device_entry = build._DeviceEntryPoint(provider)
    result = generate_source_artifact(work, analysis, "cpython", target_plan=TargetPlan(TargetSpec(), registry), device_selection=DeviceProviderSelection(PROVIDER_ID, CAPABILITY_ID), device_options=DeviceProviderOptions(values=(("device_ordinal", "0"), ("sm", sm))), device_entry_points=(device_entry,))
    if result.native_source.status != "generated":
        raise RuntimeError(f"Core source generation failed: {result.native_source}")
    rust_dir = result.layout.rust_dir
    rust = (rust_dir / "src" / "lib.rs").read_text(encoding="utf-8")
    build._assert_inference_call_order(rust)
    assert_frozen_source_contract(rust)
    [plan] = result.device_provider_plans
    return rust_dir, {"probe_sha256": _hash_file(probe), "provider_plan": plan}


def _build_cdylib(rust_dir: Path) -> Path:
    environment = dict(os.environ, RUSTUP_TOOLCHAIN="1.93.1")
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(rust_dir / "Cargo.toml")], check=True, env=environment)
    linked = tuple((rust_dir / "target" / "release").glob("*_rextio_native*.so"))
    if len(linked) != 1 or linked[0].stat().st_size == 0:
        raise RuntimeError("expected exactly one nonempty generated cdylib")
    return linked[0]


def _execute(tf: Any, python_dir: Path) -> tuple[dict[str, bool], float, float]:
    """Execute parity, residency, lifetime, repetition, and negative boundaries."""
    if (
        not tf.executing_eagerly()
        or tf.config.get_soft_device_placement()
        or not tf.config.experimental.get_synchronous_execution()
    ):
        raise RuntimeError("requires synchronous eager execution with soft placement disabled")
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) != 1 or not gpus[0].name.endswith("GPU:0"):
        raise RuntimeError("requires exactly addressable TensorFlow GPU:0")
    sys.path.insert(0, str(python_dir))
    os.environ["REXTIO_NATIVE_MODE"] = "native"
    import ctypes
    framework = next(Path(tf.sysconfig.get_lib()).glob("libtensorflow_framework.so*"), None)
    if framework is None:
        raise RuntimeError("TensorFlow runtime image is not addressable for RTLD_NOLOAD")
    ctypes.CDLL(str(framework), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
    from cuda_app.kernels import inference
    with tf.device("/GPU:0"):
        x = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [2., 1., 0.]], tf.float32)
        w = tf.constant([[1., 0.], [0., 1.], [1., 1.]], tf.float32)
        bias = tf.constant([.5, -1.], tf.float32)
        reference = tf.reduce_mean(tf.nn.relu(tf.nn.bias_add(tf.matmul(x, w), bias)), axis=1)
        snapshots = tuple(tf.identity(item) for item in (x, w, bias))
        output = inference(x, w, bias)
    tf.debugging.assert_near(output, reference, rtol=1e-5, atol=1e-5)
    absolute = float(tf.reduce_max(tf.abs(output - reference)).numpy())
    relative = float(tf.reduce_max(tf.abs((output - reference) / tf.maximum(tf.abs(reference), 1e-12))).numpy())
    if output.dtype != tf.float32 or output.shape != (4,) or not output.device.endswith("GPU:0"):
        raise RuntimeError("native output violated GPU:0 float32 rank-1 [4] contract")
    for original, snapshot in zip((x, w, bias), snapshots, strict=True):
        tf.debugging.assert_equal(original, snapshot)
        if not original.device.endswith("GPU:0"):
            raise RuntimeError("native input device changed")
    del x, w, bias
    gc.collect()
    tf.debugging.assert_near(output, reference, rtol=1e-5, atol=1e-5)
    for _ in range(3):
        repeated = inference(snapshots[0], snapshots[1], snapshots[2])
        gc.collect()
        tf.debugging.assert_near(repeated, reference, rtol=1e-5, atol=1e-5)
    cpu = tf.constant([[1., 2., 3.]], tf.float32)
    if "CPU" not in cpu.device:
        raise RuntimeError("CPU negative boundary fixture was not placed on CPU")
    negatives = (cpu, tf.cast(snapshots[0], tf.float64), tf.reshape(snapshots[0], (2, 2, 3)))
    for bad in negatives:
        try:
            inference(bad, snapshots[1], snapshots[2])
        except Exception:
            continue
        raise RuntimeError("native boundary accepted an invalid input")
    with tf.GradientTape() as tape:
        tape.watch(snapshots[0])
        try:
            inference(snapshots[0], snapshots[1], snapshots[2])
        except Exception:
            pass
        else:
            raise RuntimeError("native boundary accepted a watched GradientTape input")
    accumulator = tf.autodiff.ForwardAccumulator(snapshots[0], tf.ones_like(snapshots[0]))
    with accumulator:
        try:
            inference(snapshots[0], snapshots[1], snapshots[2])
        except Exception:
            pass
        else:
            raise RuntimeError("native boundary accepted a ForwardAccumulator input")
    return ({"native_extension_executed": True, "numerical_parity": True, "output_contract": True, "input_immutable": True, "output_lifetime": True, "repeated_calls": True, "negative_boundaries": True}, absolute, relative)


def main() -> int:
    """Run the deliberately manual real-GPU evidence collection path."""
    args = build_parser().parse_args()
    _validate_host()
    if len(args.expected_tensorflow_commit) != 40:
        raise SystemExit("--expected-tensorflow-commit must be a full 40-character commit")
    if not args.sm.startswith("sm_") or not args.sm[3:].isdigit():
        raise SystemExit("--sm must use the sm_NN or sm_NNN spelling")
    tf_root, core_root, provider_root = (path.resolve() for path in (args.tensorflow_root, args.core_root, args.provider_root))
    validate_checkout(checkout_identity(core_root), expected=CORE_COMMIT, required_ancestor=CORE_COMMIT)
    validate_checkout(checkout_identity(provider_root), expected=PROVIDER_COMMIT, required_ancestor=PROVIDER_COMMIT)
    validate_checkout(checkout_identity(tf_root), expected=args.expected_tensorflow_commit, required_ancestor=BASE_CANDIDATE_COMMIT)
    _add_sources(tf_root, core_root, provider_root)
    import tensorflow as tf  # delayed so GPU-free tests can import this module
    if tf.__version__ != "2.21.0":
        raise RuntimeError("requires TensorFlow 2.21.0")
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="rextio-tf-e3-"))
    work.mkdir(parents=True, exist_ok=True)
    rust_dir, facts = _generate(work, provider_root, args.sm)
    cdylib = _build_cdylib(rust_dir)
    cdylib_before = _hash_file(cdylib)
    execution, max_abs_error, max_rel_error = _execute(tf, rust_dir.parent / "python")
    if _hash_file(cdylib) != cdylib_before:
        raise RuntimeError("generated cdylib changed while executing the evidence run")
    from scripts import verify_cuda_e3_evidence as verifier
    if args.sm not in verifier.SMS:
        raise RuntimeError("--sm is not approved by the CUDA E3 evidence verifier")
    generated_rust = rust_dir / "src" / "lib.rs"
    artifact_rows = (
        ("plugin_wheel", "rextio_tensorflow/__init__.py", tf_root / "src" / "rextio_tensorflow" / "__init__.py"),
        ("native_extension", "rextio_tensorflow/_rextio_native.so", cdylib),
        ("generated_rust", "rextio_tensorflow/generated/lib.rs", generated_rust),
    )
    payload = {
        "contract": {"support_claim": False, "certification_ready": False, "plugin_api": "1.6"},
        "package": {"name": "rextio-tensorflow", "version": "0.1.2"},
        "environment": {"os": "Linux", "arch": "x86_64", "libc": "GNU", "python": "3.11", "tensorflow": tf.__version__, "rust": "1.93.1", "gpu": {"ordinal": 0, "compute_capability": args.sm}},
        "source": {"core_commit": CORE_COMMIT, "provider_commit": PROVIDER_COMMIT, "plugin_commit": args.expected_tensorflow_commit, "repository_clean": True},
        "artifacts": [{"kind": kind, "wheel_path": verifier.sanitized_wheel_relative(name), "sha256": verifier.sha256_file(path), "size_bytes": path.stat().st_size} for kind, name, path in artifact_rows],
        "runtime_images": [verifier.sanitized_wheel_relative("rextio_tensorflow/__init__.py")],
        "orchestration": {"provider_id": PROVIDER_ID, "capability_id": CAPABILITY_ID, "device": "cuda:0", "input_residency": "device", "dtype": "float32", "ranks": [1, 2], "operations": ["tf.matmul", "tf.nn.bias_add", "tf.nn.relu", "tf.reduce_mean-axis1"]},
        "invariants": {"execution": {"native_extension_executed": execution["native_extension_executed"], "kernel_activity_verified": False, "runtime_transfer_profiled": False}, "numerical": {"reference": "tensorflow-eager", "atol": 1e-5, "rtol": 1e-5, "max_abs_error": max_abs_error, "max_rel_error": max_rel_error}, "device": {"inputs_on_gpu": True, "output_on_gpu": True, "gpu_ordinal": 0}, "lifetime": {"borrowed_inputs_alive": True, "no_host_fallback_observed": True}, "negative_boundary": {"unsupported_dtype_rejected": True, "rank_rejected": True, "device_ordinal_rejected": True, "operation_rejected": True}},
    }
    envelope = verifier.make_envelope(payload)
    verifier.validate_envelope(envelope)
    atomic_write_json(args.output, envelope)
    print(json.dumps({"certification_ready": False, "evidence": str(args.output), "native_extension_executed": True, "support_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
def _hash_file(path: Path) -> str:
    """Return a local SHA-256 used before verifier helpers are importable."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
