#!/usr/bin/env python3
"""Offline verifier for the non-certifying TensorFlow CUDA E3 evidence envelope.

This deliberately verifies evidence metadata only; it never imports TensorFlow,
loads an extension, or attempts to infer CUDA kernel activity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 65_536
MAX_DEPTH = 12
HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+\-]{0,239}$")
FORBIDDEN_TEXT = re.compile(
    r"(?:https?://|file://|(?:token|password|secret|apikey|api_key|authorization)\s*[=:])",
    re.IGNORECASE,
)

CORE = "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0"
PROVIDER = "cf65733f06b91a801f9806367f09948ee7162540"
BASE_PLUGIN = "16e368a"
OPS = ["tf.matmul", "tf.nn.bias_add", "tf.nn.relu", "tf.reduce_mean-axis1"]
SMS = {"sm_60", "sm_61", "sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"}


class EvidenceError(ValueError):
    """Raised when an evidence envelope is not a claim this verifier accepts."""


def canonical_json(value: Any) -> str:
    """Return the sole canonical encoding used for payload hashing."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return SHA-256 of the canonical, non-circular evidence payload."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash an artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_wheel_relative(path: str) -> str:
    """Validate and return a wheel-relative runtime image name, never a host path."""
    _wheel_path(path, "wheel-relative path")
    return path


def make_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Create a canonical, non-circular envelope for a validated-style payload."""
    return {"schema_version": 1, "payload": payload, "payload_sha256": payload_sha256(payload)}


def _closed(value: Any, expected: dict[str, Any], where: str = "payload") -> None:
    if not isinstance(value, dict):
        raise EvidenceError(f"{where} must be an object")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise EvidenceError(f"{where} has unknown or missing fields: {sorted(actual ^ wanted)}")


def _string(value: Any, where: str, *, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{where} must be a non-empty string")
    if len(value) > 512 or FORBIDDEN_TEXT.search(value):
        raise EvidenceError(f"{where} contains a path, URL, or credential-looking value")
    if pattern and not pattern.fullmatch(value):
        raise EvidenceError(f"{where} has invalid format")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceError(f"{where} must be finite")
    return float(value)


def _wheel_path(value: Any, where: str) -> None:
    text = _string(value, where, pattern=SAFE_PATH)
    if text.startswith("/") or ".." in text.split("/") or not text.startswith("rextio_tensorflow/"):
        raise EvidenceError(f"{where} must be a sanitized wheel-relative runtime image")


def _descends_from_base(commit: str) -> None:
    try:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_PLUGIN, commit],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise EvidenceError("cannot verify plugin candidate ancestry") from exc
    if completed.returncode != 0:
        raise EvidenceError(f"plugin candidate is not a descendant of {BASE_PLUGIN}")


def _validate_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise EvidenceError("evidence exceeds maximum nesting depth")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError("JSON object keys must be strings")
            _validate_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_depth(item, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError("evidence contains non-finite numeric value")


def validate_envelope(envelope: Any) -> dict[str, Any]:
    """Strictly validate an E3 evidence envelope and return its payload."""
    _validate_depth(envelope)
    _closed(envelope, {"schema_version": None, "payload": None, "payload_sha256": None}, "envelope")
    if envelope["schema_version"] != 1:
        raise EvidenceError("unsupported evidence schema_version")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise EvidenceError("envelope.payload must be an object")
    claimed_hash = _string(envelope["payload_sha256"], "payload_sha256", pattern=HEX64)
    if claimed_hash != payload_sha256(payload):
        raise EvidenceError("payload_sha256 does not match canonical payload")

    _closed(
        payload,
        {
            "contract": None,
            "package": None,
            "environment": None,
            "source": None,
            "artifacts": None,
            "runtime_images": None,
            "orchestration": None,
            "invariants": None,
        },
    )
    c = payload["contract"]
    _closed(c, {"support_claim": None, "certification_ready": None, "plugin_api": None})
    if c != {"support_claim": False, "certification_ready": False, "plugin_api": "1.6"}:
        raise EvidenceError(
            "contract must retain support_claim=false and certification_ready=false"
        )
    package = payload["package"]
    _closed(package, {"name": None, "version": None})
    if package != {"name": "rextio-tensorflow", "version": "0.1.2"}:
        raise EvidenceError("package binding does not match the E3 candidate")
    e = payload["environment"]
    _closed(
        e,
        {
            "os": None,
            "arch": None,
            "libc": None,
            "python": None,
            "tensorflow": None,
            "rust": None,
            "gpu": None,
        },
    )
    if {k: e[k] for k in ("os", "arch", "libc", "python", "tensorflow", "rust")} != {
        "os": "Linux",
        "arch": "x86_64",
        "libc": "GNU",
        "python": "3.11",
        "tensorflow": "2.21.0",
        "rust": "1.93.1",
    }:
        raise EvidenceError("environment does not match the exact E3 platform contract")
    _closed(e["gpu"], {"ordinal": None, "compute_capability": None})
    if e["gpu"]["ordinal"] != 0 or e["gpu"]["compute_capability"] not in SMS:
        raise EvidenceError("GPU must be ordinal 0 with an allowed SM")
    s = payload["source"]
    _closed(
        s,
        {
            "core_commit": None,
            "provider_commit": None,
            "plugin_commit": None,
            "repository_clean": None,
        },
    )
    if (
        s["core_commit"] != CORE
        or s["provider_commit"] != PROVIDER
        or s["repository_clean"] is not True
    ):
        raise EvidenceError("source bindings are not exact or clean")
    plugin_commit = _string(s["plugin_commit"], "source.plugin_commit", pattern=GIT_SHA)
    _descends_from_base(plugin_commit)
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise EvidenceError("artifacts must contain exactly three hashed artifacts")
    expected_artifacts = {"plugin_wheel", "native_extension", "generated_rust"}
    seen: set[str] = set()
    for artifact in artifacts:
        _closed(
            artifact,
            {"kind": None, "wheel_path": None, "sha256": None, "size_bytes": None},
            "artifact",
        )
        kind = _string(artifact["kind"], "artifact.kind")
        seen.add(kind)
        _wheel_path(artifact["wheel_path"], "artifact.wheel_path")
        _string(artifact["sha256"], "artifact.sha256", pattern=HEX64)
        if (
            isinstance(artifact["size_bytes"], bool)
            or not isinstance(artifact["size_bytes"], int)
            or not 0 < artifact["size_bytes"] <= 2**31
        ):
            raise EvidenceError("artifact.size_bytes must be a bounded positive integer")
    if seen != expected_artifacts:
        raise EvidenceError("artifact kinds are incomplete or duplicated")
    images = payload["runtime_images"]
    if not isinstance(images, list) or not 1 <= len(images) <= 8:
        raise EvidenceError("runtime_images must be a non-empty bounded list")
    for image in images:
        _wheel_path(image, "runtime_images item")
    o = payload["orchestration"]
    _closed(
        o,
        {
            "provider_id": None,
            "capability_id": None,
            "device": None,
            "input_residency": None,
            "dtype": None,
            "ranks": None,
            "operations": None,
        },
    )
    if o != {
        "provider_id": "rextio-device-cuda",
        "capability_id": "cuda-tensorflow-tfe-linux-x86_64",
        "device": "cuda:0",
        "input_residency": "device",
        "dtype": "float32",
        "ranks": [1, 2],
        "operations": OPS,
    }:
        raise EvidenceError("orchestration does not match the exact E3 slice")
    i = payload["invariants"]
    _closed(
        i,
        {
            "execution": None,
            "numerical": None,
            "device": None,
            "lifetime": None,
            "negative_boundary": None,
        },
    )
    _closed(
        i["execution"],
        {
            "native_extension_executed": None,
            "kernel_activity_verified": None,
            "runtime_transfer_profiled": None,
        },
    )
    if i["execution"] != {
        "native_extension_executed": True,
        "kernel_activity_verified": False,
        "runtime_transfer_profiled": False,
    }:
        raise EvidenceError(
            "first-stage evidence may execute the extension but cannot claim kernel or transfer profiling"
        )
    _closed(
        i["numerical"],
        {
            "reference": None,
            "atol": None,
            "rtol": None,
            "max_abs_error": None,
            "max_rel_error": None,
        },
    )
    n = i["numerical"]
    if (
        n["reference"] != "tensorflow-eager"
        or _finite(n["atol"], "atol") != 1e-5
        or _finite(n["rtol"], "rtol") != 1e-5
    ):
        raise EvidenceError("numerical tolerances must be the exact approved values")
    if (
        not 0 <= _finite(n["max_abs_error"], "max_abs_error") <= n["atol"]
        or not 0 <= _finite(n["max_rel_error"], "max_rel_error") <= n["rtol"]
    ):
        raise EvidenceError("numerical errors exceed declared tolerances")
    _closed(i["device"], {"inputs_on_gpu": None, "output_on_gpu": None, "gpu_ordinal": None})
    if i["device"] != {"inputs_on_gpu": True, "output_on_gpu": True, "gpu_ordinal": 0}:
        raise EvidenceError("device invariant is incomplete")
    _closed(i["lifetime"], {"borrowed_inputs_alive": None, "no_host_fallback_observed": None})
    if i["lifetime"] != {"borrowed_inputs_alive": True, "no_host_fallback_observed": True}:
        raise EvidenceError("lifetime invariant is incomplete")
    _closed(
        i["negative_boundary"],
        {
            "unsupported_dtype_rejected": None,
            "rank_rejected": None,
            "device_ordinal_rejected": None,
            "operation_rejected": None,
        },
    )
    if any(value is not True for value in i["negative_boundary"].values()):
        raise EvidenceError("negative boundary checks are incomplete")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the evidence verifier command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="path to a CUDA E3 evidence JSON envelope")
    args = parser.parse_args(argv)
    try:
        raw = args.evidence.read_bytes()
        if len(raw) > MAX_BYTES:
            raise EvidenceError("evidence exceeds maximum size")
        envelope = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON value {value}")
            ),
        )
        payload = validate_envelope(envelope)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, EvidenceError) as exc:
        print(f"evidence verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {"sha256": payload_sha256(payload), "support_claim": False, "verified": True}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
