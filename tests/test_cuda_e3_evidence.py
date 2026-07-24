"""Tests for the offline CUDA E3 schema and integrity verifier."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_cuda_e3_evidence import (
    BASE_CANDIDATE_COMMIT,
    EvidenceError,
    canonical_bytes,
    canonical_json,
    make_envelope,
    payload_sha256,
    validate_document,
    validate_envelope,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_cuda_e3_evidence.py"
HASH = "a" * 64
PLUGIN_COMMIT = "b" * 40


def _payload() -> dict[str, object]:
    artifact_roles = (
        "provider_probe",
        "harness_script",
        "verifier_script",
        "generated_lib_rs",
        "generated_cargo_toml",
        "generated_cargo_lock",
        "native_extension",
    )
    runtime_rows = (
        ("tensorflow_cc", "tensorflow/libtensorflow_cc.so.2"),
        ("tensorflow_framework", "tensorflow/libtensorflow_framework.so.2"),
        (
            "pywrap_tensorflow_common",
            "tensorflow/python/lib_pywrap_tensorflow_common.so",
        ),
    )
    return {
        "contract": {
            "evidence_schema": "tensorflow-cuda-e3-real-nvidia-v1",
            "verification_scope": "schema-and-integrity-only",
            "producer_assertions": "self-attested-by-manual-harness",
            "support_claim": False,
            "certification_ready": False,
            "plugin_api": "1.6",
        },
        "package": {
            "distribution": "rextio-tensorflow",
            "version": "0.1.2",
            "plugin_module": "rextio_tensorflow.plugin",
            "native_module": "cuda_app._rextio_native",
        },
        "source": {
            "core_commit": "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0",
            "core_clean": True,
            "provider_commit": "cf65733f06b91a801f9806367f09948ee7162540",
            "provider_clean": True,
            "plugin_commit": PLUGIN_COMMIT,
            "plugin_clean": True,
            "base_candidate_commit": BASE_CANDIDATE_COMMIT,
            "plugin_ancestry_checked": True,
        },
        "environment": {
            "os": "Linux",
            "arch": "x86_64",
            "libc": "GNU",
            "python_implementation": "CPython",
            "python_version": "3.11",
            "tensorflow_version": "2.21.0",
            "cuda_driver_version": 12_000,
            "gpu": {"ordinal": 0, "sm": "sm_80"},
        },
        "toolchain": {
            "rustc_version": "1.93.1",
            "cargo_version": "1.93.1",
            "target": "x86_64-unknown-linux-gnu",
        },
        "artifacts": [
            {
                "role": role,
                "label": f"evidence/{role}",
                "sha256": HASH,
                "size_bytes": 1,
            }
            for role in artifact_roles
        ],
        "runtime_images": [
            {
                "role": role,
                "wheel_path": wheel_path,
                "sha256": HASH,
                "size_bytes": 1,
                "build_id": None,
                "mapped": True,
            }
            for role, wheel_path in runtime_rows
        ],
        "orchestration": {
            "provider_id": "rextio-device-cuda",
            "capability_id": "cuda-tensorflow-tfe-linux-x86_64",
            "device": "cuda:0",
            "input_residency": "device",
            "dtype": "float32",
            "ranks": [1, 2],
            "operations": [
                "tf.matmul",
                "tf.nn.bias_add",
                "tf.nn.relu",
                "tf.reduce_mean-axis1",
            ],
            "artifact_profile_sha256": HASH,
            "authorization_sha256": HASH,
            "provider_lock_sha256": HASH,
            "probe_sha256": HASH,
            "observations_sha256": HASH,
        },
        "invariants": {
            "execution": {
                "native_extension_executed": True,
                "kernel_activity_verified": False,
                "runtime_transfer_profiled": False,
                "runtime_provenance_checked": True,
            },
            "numerical": {
                "reference": "tensorflow-eager",
                "atol": 1e-5,
                "rtol": 1e-5,
                "max_scaled_error": 0.5,
            },
            "output": {
                "device": "GPU:0",
                "dtype": "float32",
                "rank": 1,
                "shape": [4],
            },
            "lifetime": {
                "inputs_unchanged": True,
                "output_survives_input_gc": True,
                "repeated_calls": True,
            },
            "negative_boundary": {
                "cpu_input_rejected": True,
                "float64_rejected": True,
                "wrong_rank_rejected": True,
                "watched_tape_rejected": True,
                "forward_accumulator_rejected": True,
            },
        },
    }


def _envelope() -> dict[str, object]:
    return make_envelope(_payload())


def _rehash(envelope: dict[str, object]) -> None:
    envelope["payload_sha256"] = payload_sha256(envelope["payload"])


def _mutated(path: tuple[object, ...], value: object) -> dict[str, object]:
    envelope = copy.deepcopy(_envelope())
    cursor: object = envelope
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    _rehash(envelope)
    return envelope


def test_canonical_roundtrip_and_non_circular_hash() -> None:
    envelope = _envelope()
    raw = canonical_bytes(envelope)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert canonical_json(json.loads(raw)) + "\n" == raw.decode("ascii")
    assert validate_document(raw) == envelope["payload"]
    assert validate_envelope(envelope) == envelope["payload"]


def test_tampering_without_rehash_is_rejected() -> None:
    envelope = _envelope()
    envelope["payload"]["environment"]["gpu"]["sm"] = "sm_90"
    with pytest.raises(EvidenceError, match="payload_sha256"):
        validate_envelope(envelope)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("payload", "contract", "support_claim"), True, "support_claim"),
        (("payload", "contract", "certification_ready"), True, "support_claim"),
        (
            ("payload", "contract", "producer_assertions"),
            "independently-verified",
            "self-attested",
        ),
        (("payload", "invariants", "numerical", "rtol"), 1e-4, "tolerances"),
        (
            ("payload", "invariants", "numerical", "max_scaled_error"),
            1.00001,
            "max_scaled_error",
        ),
        (
            ("payload", "invariants", "execution", "kernel_activity_verified"),
            True,
            "must not claim",
        ),
        (
            ("payload", "invariants", "execution", "runtime_transfer_profiled"),
            True,
            "must not claim",
        ),
        (
            ("payload", "invariants", "negative_boundary", "watched_tape_rejected"),
            False,
            "incomplete",
        ),
        (("schema_version",), True, "schema_version"),
        (("payload", "orchestration", "ranks"), [True, 2], "orchestration"),
    ),
)
def test_rejects_rehashed_overclaims_and_weakened_invariants(
    path: tuple[object, ...],
    value: object,
    message: str,
) -> None:
    with pytest.raises(EvidenceError, match=message):
        validate_envelope(_mutated(path, value))


def test_unknown_fields_and_duplicate_roles_are_rejected() -> None:
    envelope = copy.deepcopy(_envelope())
    envelope["payload"]["extra"] = True
    _rehash(envelope)
    with pytest.raises(EvidenceError, match="unknown or missing"):
        validate_envelope(envelope)

    envelope = copy.deepcopy(_envelope())
    envelope["payload"]["artifacts"][1]["role"] = envelope["payload"]["artifacts"][0]["role"]
    _rehash(envelope)
    with pytest.raises(EvidenceError, match="duplicate artifact"):
        validate_envelope(envelope)

    envelope = copy.deepcopy(_envelope())
    envelope["payload"]["runtime_images"][1]["role"] = "tensorflow_cc"
    _rehash(envelope)
    with pytest.raises(EvidenceError, match="duplicate runtime"):
        validate_envelope(envelope)


@pytest.mark.parametrize(
    "leak",
    (
        "/tmp/native.so",
        "file:///tmp/native.so",
        "https://example.test/native.so",
        "token=abc",
        "ghp_abcdefghijklmnopqrstuvwxyz1234",
        "C:\\secret\\native.dll",
    ),
)
def test_recursively_rejects_path_url_and_credential_leaks(leak: str) -> None:
    envelope = _mutated(("payload", "artifacts", 0, "label"), leak)
    with pytest.raises(EvidenceError, match="absolute path|URL|credential"):
        validate_envelope(envelope)


def test_runtime_images_are_exact_and_build_id_is_nullable_bounded() -> None:
    envelope = _mutated(
        ("payload", "runtime_images", 0, "wheel_path"),
        "tensorflow/libtensorflow_cc.so",
    )
    with pytest.raises(EvidenceError, match="wrong wheel-relative"):
        validate_envelope(envelope)
    envelope = _mutated(("payload", "runtime_images", 0, "build_id"), "abc")
    with pytest.raises(EvidenceError, match="invalid format"):
        validate_envelope(envelope)
    envelope = _mutated(("payload", "runtime_images", 0, "mapped"), False)
    with pytest.raises(EvidenceError, match="mapped=true"):
        validate_envelope(envelope)


def test_offline_portability_has_no_checkout_or_subprocess_dependency(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(canonical_bytes(_envelope()))
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(evidence)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["schema_verified"] is True
    assert result["support_claim"] is False
    assert result["certification_ready"] is False
    assert result["payload_sha256"] == _envelope()["payload_sha256"]


@pytest.mark.parametrize(
    "transform",
    (
        lambda raw: raw.rstrip(b"\n"),
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b'{"payload":', b'{ "payload":', 1),
        lambda raw: raw.replace(b'"arch":"x86_64"', b'"arch": "x86_64"', 1),
    ),
)
def test_noncanonical_raw_bytes_are_rejected(transform) -> None:
    with pytest.raises(EvidenceError, match="not canonical"):
        validate_document(transform(canonical_bytes(_envelope())))


def test_malformed_nonfinite_oversize_and_depth_are_rejected(tmp_path: Path) -> None:
    for raw in (b'{"x":NaN}\n', b"{broken}\n", b"\xff\n"):
        with pytest.raises(EvidenceError):
            validate_document(raw)
    with pytest.raises(EvidenceError, match="maximum size"):
        validate_document(b" " * 131_073)
    value: object = {}
    cursor = value
    for _ in range(14):
        next_value: dict[str, object] = {}
        cursor["x"] = next_value
        cursor = next_value
    with pytest.raises(EvidenceError, match="nesting depth"):
        validate_envelope(value)

    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"{broken}\n")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(malformed)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "schema/integrity verification failed" in completed.stderr


def test_cli_help_names_schema_and_integrity_scope() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "schema and integrity" in completed.stdout
