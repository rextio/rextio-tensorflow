"""Tests for the strict, TensorFlow-free CUDA E3 evidence verifier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.verify_cuda_e3_evidence import (
    EvidenceError,
    canonical_json,
    make_envelope,
    payload_sha256,
    validate_envelope,
)


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _payload() -> dict[str, object]:
    return {
        "contract": {"support_claim": False, "certification_ready": False, "plugin_api": "1.6"},
        "package": {"name": "rextio-tensorflow", "version": "0.1.2"},
        "environment": {
            "os": "Linux",
            "arch": "x86_64",
            "libc": "GNU",
            "python": "3.11",
            "tensorflow": "2.21.0",
            "rust": "1.93.1",
            "gpu": {"ordinal": 0, "compute_capability": "sm_80"},
        },
        "source": {
            "core_commit": "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0",
            "provider_commit": "cf65733f06b91a801f9806367f09948ee7162540",
            "plugin_commit": _commit(),
            "repository_clean": True,
        },
        "artifacts": [
            {
                "kind": kind,
                "wheel_path": f"rextio_tensorflow/{name}",
                "sha256": "a" * 64,
                "size_bytes": 1,
            }
            for kind, name in (
                ("plugin_wheel", "plugin.whl"),
                ("native_extension", "native.so"),
                ("generated_rust", "generated.rs"),
            )
        ],
        "runtime_images": ["rextio_tensorflow/runtime/libtensorflow_framework.so"],
        "orchestration": {
            "provider_id": "rextio-device-cuda",
            "capability_id": "cuda-tensorflow-tfe-linux-x86_64",
            "device": "cuda:0",
            "input_residency": "device",
            "dtype": "float32",
            "ranks": [1, 2],
            "operations": ["tf.matmul", "tf.nn.bias_add", "tf.nn.relu", "tf.reduce_mean-axis1"],
        },
        "invariants": {
            "execution": {
                "native_extension_executed": True,
                "kernel_activity_verified": False,
                "runtime_transfer_profiled": False,
            },
            "numerical": {
                "reference": "tensorflow-eager",
                "atol": 1e-5,
                "rtol": 1e-5,
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
            },
            "device": {"inputs_on_gpu": True, "output_on_gpu": True, "gpu_ordinal": 0},
            "lifetime": {"borrowed_inputs_alive": True, "no_host_fallback_observed": True},
            "negative_boundary": {
                "unsupported_dtype_rejected": True,
                "rank_rejected": True,
                "device_ordinal_rejected": True,
                "operation_rejected": True,
            },
        },
    }


def _envelope() -> dict[str, object]:
    return make_envelope(_payload())


def _rehash(envelope: dict[str, object]) -> None:
    envelope["payload_sha256"] = payload_sha256(envelope["payload"])


def test_canonical_roundtrip() -> None:
    envelope = _envelope()
    assert canonical_json(json.loads(canonical_json(envelope))) == canonical_json(envelope)
    assert validate_envelope(envelope) == envelope["payload"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["payload"]["contract"].__setitem__("support_claim", True),
        lambda e: e["payload"]["invariants"]["numerical"].__setitem__("rtol", 1e-4),
        lambda e: e["payload"]["invariants"]["execution"].__setitem__(
            "kernel_activity_verified", True
        ),
        lambda e: e["payload"]["invariants"]["negative_boundary"].__setitem__(
            "rank_rejected", False
        ),
    ],
)
def test_rejects_tampering_and_overclaims(mutate) -> None:
    envelope = _envelope()
    mutate(envelope)
    _rehash(envelope)
    with pytest.raises(EvidenceError):
        validate_envelope(envelope)


def test_rejects_unknown_path_url_and_credential_leaks() -> None:
    for value in ("/tmp/native.so", "https://example.test/x", "rextio_tensorflow/token=abc"):
        envelope = _envelope()
        envelope["payload"]["runtime_images"] = [value]
        _rehash(envelope)
        with pytest.raises(EvidenceError):
            validate_envelope(envelope)
    envelope = _envelope()
    envelope["payload"]["extra"] = True
    with pytest.raises(EvidenceError):
        validate_envelope(envelope)


def test_rejects_malformed_nonfinite_oversize_and_depth(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/verify_cuda_e3_evidence.py"
    bad = tmp_path / "bad.json"
    bad.write_text('{"x": NaN}', encoding="utf-8")
    assert (
        subprocess.run([sys.executable, str(script), str(bad)], capture_output=True).returncode == 1
    )
    huge = tmp_path / "huge.json"
    huge.write_bytes(b" " * 65537)
    assert (
        subprocess.run([sys.executable, str(script), str(huge)], capture_output=True).returncode
        == 1
    )
    value: object = {}
    cursor = value
    for _ in range(14):
        next_value: dict[str, object] = {}
        cursor["x"] = next_value
        cursor = next_value
    with pytest.raises(EvidenceError):
        validate_envelope(value)


def test_cli_help_and_success(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/verify_cuda_e3_evidence.py"
    assert (
        subprocess.run([sys.executable, str(script), "--help"], capture_output=True).returncode == 0
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(canonical_json(_envelope()), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script), str(evidence)], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["verified"] is True
