"""GPU-free contracts for the opt-in real-NVIDIA CUDA E3 harness."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "certify_cuda_candidate.py"


def _module():
    sys.modules.pop("certify_cuda_candidate", None)
    spec = importlib.util.spec_from_file_location("certify_cuda_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_module_import_is_tensorflow_free_and_all_helpers_precede_main_guard() -> None:
    before = set(sys.modules)
    _module()
    added = set(sys.modules) - before
    assert "tensorflow" not in added
    assert not any(name.startswith("tensorflow.") for name in added)

    source = SCRIPT.read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__":')
    assert "\ndef " not in source[guard:]
    assert "\nclass " not in source[guard:]


def test_cli_requires_exclusive_work_and_output_paths(tmp_path: Path) -> None:
    module = _module()
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    help_text = parser.format_help()
    for option in (
        "--output",
        "--work-dir",
        "--core-root",
        "--provider-root",
        "--expected-tensorflow-commit",
        "--sm",
    ):
        assert option in help_text

    work = tmp_path / "work"
    output = tmp_path / "evidence.json"
    args = argparse.Namespace(
        work_dir=work,
        output=output,
        expected_tensorflow_commit="a" * 40,
        sm="sm_80",
    )
    module.validate_requested_paths_and_values(args, {"sm_80"})
    work.mkdir()
    with pytest.raises(RuntimeError, match="work-dir.*must not exist"):
        module.validate_requested_paths_and_values(args, {"sm_80"})
    work.rmdir()
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(RuntimeError, match="output.*must not exist"):
        module.validate_requested_paths_and_values(args, {"sm_80"})
    output.unlink()
    args.expected_tensorflow_commit = "A" * 40
    with pytest.raises(RuntimeError, match="lowercase"):
        module.validate_requested_paths_and_values(args, {"sm_80"})
    args.expected_tensorflow_commit = "a" * 40
    args.sm = "sm_99"
    with pytest.raises(RuntimeError, match="allowed"):
        module.validate_requested_paths_and_values(args, {"sm_80"})


def test_canonical_atomic_create_is_exclusive_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "evidence.json"
    canonical = b'{"a":1,"b":2}\n'
    module.atomic_create(output, canonical)
    assert output.read_bytes() == canonical
    with pytest.raises(FileExistsError):
        module.atomic_create(output, b"replacement\n")
    assert output.read_bytes() == canonical
    assert not list(tmp_path.glob(".evidence.json.*"))

    second = tmp_path / "second.json"
    monkeypatch.setattr(module.os, "link", lambda *_: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        module.atomic_create(second, canonical)
    assert not second.exists()
    assert not list(tmp_path.glob(".second.json.*"))


def test_cargo_build_is_locked_pinned_and_installs_exact_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    rust_dir = tmp_path / "rust"
    release = rust_dir / "target" / "release"
    release.mkdir(parents=True)
    (rust_dir / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
    built = release / "lib_rextio_native.so"
    built.write_bytes(b"native")
    python_dir = tmp_path / "python"
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.sysconfig, "get_config_var", lambda name: ".cpython-311-x86_64-linux-gnu.so")
    environment = {
        "VIRTUAL_ENV": "/venv",
        "PATH": "/venv/bin:/usr/bin",
        "PYO3_PYTHON": "/venv/bin/python",
    }
    installed = module.build_generated_extension(rust_dir, python_dir, environment)
    assert calls[0][0][:4] == ["cargo", "+1.93.1", "build", "--locked"]
    assert "--release" in calls[0][0]
    assert calls[0][1] == environment
    assert installed == python_dir / "_rextio_native.cpython-311-x86_64-linux-gnu.so"
    assert installed.read_bytes() == b"native"


def test_provider_observations_are_validated_and_bound_to_hashes() -> None:
    module = _module()
    profile_hash = "1" * 64
    plan = {
        "artifact_profile": {"target_triple": module.TARGET},
        "lock": {
            "artifact_profile_sha256": profile_hash,
            "preflight_sha256": "2" * 64,
        },
        "lowering_authorization": {
            "provider_id": module.PROVIDER_ID,
            "capability_id": module.CAPABILITY_ID,
            "logical_device": "gpu:0",
            "runtime": "tensorflow-tfe",
            "artifact_profile_sha256": profile_hash,
        },
        "report": {
            "status": "ready",
            "support_claim": False,
            "certification_tier": "build-only",
            "reason_codes": [],
            "observations": [
                {"key": "driver.version", "value": "12080"},
                {"key": "selected.device", "value": "0"},
                {"key": "selected.sm", "value": "sm_80"},
                {"key": "probe.schema", "value": "1"},
                {"key": "framework.runtime", "value": "tensorflow-tfe"},
            ],
        },
    }
    result = module.validate_and_bind_provider_plan(plan, "sm_80", "3" * 64)
    assert result["driver_version"] == 12080
    assert result["selected_sm"] == "sm_80"
    for name in (
        "artifact_profile_sha256",
        "authorization_sha256",
        "lock_sha256",
        "probe_sha256",
        "observations_sha256",
    ):
        assert len(result[name]) == 64
    plan["report"]["support_claim"] = True
    with pytest.raises(RuntimeError, match="support"):
        module.validate_and_bind_provider_plan(plan, "sm_80", "3" * 64)


def test_runtime_dso_capture_requires_expected_mapped_wheel_images(tmp_path: Path) -> None:
    module = _module()
    wheel = tmp_path / "tensorflow"
    pywrap = wheel / "python" / "_pywrap_tensorflow_internal.so"
    cc = wheel / "libtensorflow_cc.so.2"
    framework = wheel / "libtensorflow_framework.so.2"
    for path in (pywrap, cc, framework):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    maps = "\n".join(f"7f-8 r-xp 0 00:00 0 {path}" for path in (pywrap, cc, framework))
    identities = module.capture_runtime_images(
        wheel,
        maps,
        read_build_id=lambda path: f"build-{path.name}",
    )
    assert {row["role"] for row in identities} == {
        "tensorflow_pywrap",
        "tensorflow_cc",
        "tensorflow_framework",
    }
    assert all(row["mapped"] is True for row in identities)
    assert all(not row["wheel_path"].startswith("/") for row in identities)
    with pytest.raises(RuntimeError, match="mapped"):
        module.capture_runtime_images(wheel, maps.replace(str(cc), ""), read_build_id=lambda _: "x")


def test_execution_payload_records_only_observed_boundaries_and_no_profiler_claims() -> None:
    module = _module()
    result = module.ExecutionResult(
        native_extension_executed=True,
        numerical_parity=True,
        max_scaled_error=0.25,
        inputs_unchanged=True,
        output_lifetime=True,
        repeated_calls=True,
        cpu_input_rejected=True,
        float64_rejected=True,
        wrong_rank_rejected=True,
        gradient_tape_rejected=True,
        forward_accumulator_rejected=True,
        inputs_on_gpu=True,
        output_on_gpu=True,
        runtime_provenance_checked=True,
    )
    invariants = module.execution_invariants(result)
    assert invariants["execution"] == {
        "native_extension_executed": True,
        "kernel_activity_verified": False,
        "runtime_transfer_profiled": False,
        "runtime_provenance_checked": True,
    }
    assert invariants["numerical"]["max_scaled_error"] == 0.25
    assert invariants["negative_boundary"] == {
        "cpu_input_rejected": True,
        "float64_rejected": True,
        "wrong_rank_rejected": True,
        "watched_tape_rejected": True,
        "forward_accumulator_rejected": True,
    }
    encoded = json.dumps(invariants, sort_keys=True)
    assert "device_ordinal_rejected" not in encoded
    assert "operation_rejected" not in encoded
    assert "no_host_fallback_observed" not in encoded


def test_producer_payload_is_accepted_by_the_offline_verifier_without_tensorflow() -> None:
    module = _module()
    from scripts import verify_cuda_e3_evidence as verifier

    digest = "a" * 64
    result = module.ExecutionResult(
        native_extension_executed=True,
        numerical_parity=True,
        max_scaled_error=0.25,
        inputs_unchanged=True,
        output_lifetime=True,
        repeated_calls=True,
        cpu_input_rejected=True,
        float64_rejected=True,
        wrong_rank_rejected=True,
        gradient_tape_rejected=True,
        forward_accumulator_rejected=True,
        inputs_on_gpu=True,
        output_on_gpu=True,
        runtime_provenance_checked=True,
    )
    payload = module.build_payload(
        source={
            "core_commit": verifier.CORE_COMMIT,
            "core_clean": True,
            "provider_commit": verifier.PROVIDER_COMMIT,
            "provider_clean": True,
            "plugin_commit": "b" * 40,
            "plugin_clean": True,
            "base_candidate_commit": verifier.BASE_CANDIDATE_COMMIT,
            "plugin_ancestry_checked": True,
        },
        environment={
            "os": "Linux",
            "arch": "x86_64",
            "libc": "GNU",
            "python_implementation": "CPython",
            "python_version": "3.11",
            "tensorflow_version": "2.21.0",
            "cuda_driver_version": 12080,
            "gpu": {"ordinal": 0, "sm": "sm_80"},
        },
        toolchain={"rustc_version": "1.93.1", "cargo_version": "1.93.1", "target": module.TARGET},
        artifacts=[
            {"role": role, "label": f"evidence/{role}", "sha256": digest, "size_bytes": 1}
            for role in sorted(verifier.ARTIFACT_ROLES)
        ],
        runtime_images=[
            {"role": role, "wheel_path": path, "sha256": digest, "size_bytes": 1, "build_id": None, "mapped": True}
            for role, path in verifier.RUNTIME_IMAGES.items()
        ],
        bindings={
            "artifact_profile_sha256": digest,
            "authorization_sha256": digest,
            "lock_sha256": digest,
            "probe_sha256": digest,
            "observations_sha256": digest,
        },
        result=result,
    )
    assert payload["package"]["native_module"] == "_rextio_native"
    assert verifier.validate_envelope(verifier.make_envelope(payload)) == payload


def test_gpu_device_and_tolerance_helpers_are_exact_and_gpu_free() -> None:
    module = _module()
    assert module.is_gpu0_device("/job:localhost/replica:0/task:0/device:GPU:0")
    assert not module.is_gpu0_device("/job:localhost/replica:0/task:0/device:GPU:1")
    assert module.tolerance_scaled_error(2e-5, 1.0) == pytest.approx(1.0)


def test_source_contains_explicit_tensorflow_before_extension_and_provenance_guards() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("import tensorflow as tf") < source.index("import_generated_inference(")
    for token in (
        'tf.config.set_soft_device_placement(False)',
        'tf.config.experimental.set_synchronous_execution(True)',
        'tf.device("/CPU:0")',
        "RTLD_NOLOAD",
        'Path("/proc/self/maps")',
        '"readelf"',
        "dladdr",
        "sysconfig.get_config_var(\"EXT_SUFFIX\")",
        'sys.modules.pop("_rextio_native"',
    ):
        assert token in source
