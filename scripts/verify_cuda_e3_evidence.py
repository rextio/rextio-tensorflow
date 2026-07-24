#!/usr/bin/env python3
"""Offline schema and integrity verifier for TensorFlow CUDA E3 evidence.

Verification proves only that a document is canonical, internally untampered,
and conforms to this closed first-stage evidence schema. The payload SHA-256 is
an integrity checksum; it is not authentication, execution proof, hardware
certification, or independent validation of the producer's self-attestations.
The verifier never imports TensorFlow, loads artifacts, shells out, or reads a
source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 131_072
MAX_DEPTH = 12
MAX_STRING = 512
MAX_ARTIFACT_BYTES = 2**40
ATOL = 1e-5
RTOL = 1e-5

HEX64 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+\-]{0,239}$")
BUILD_ID = re.compile(r"^[0-9a-f]{8,128}$")
URL = re.compile(r"(?:https?|file)://", re.IGNORECASE)
WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
CREDENTIAL = re.compile(
    r"(?:"
    r"(?:token|password|passwd|secret|api[_-]?key|authorization|bearer)\s*[:=]"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[A-Z0-9]{16}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r")",
    re.IGNORECASE,
)

CORE_COMMIT = "7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0"
PROVIDER_COMMIT = "cf65733f06b91a801f9806367f09948ee7162540"
BASE_CANDIDATE_COMMIT = "16e368a2e73be58d4cc51da1672a8a842e394fbd"
OPERATIONS = [
    "tf.matmul",
    "tf.nn.bias_add",
    "tf.nn.relu",
    "tf.reduce_mean-axis1",
]
SMS = {
    "sm_60",
    "sm_61",
    "sm_70",
    "sm_72",
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_87",
    "sm_89",
    "sm_90",
}
ARTIFACT_ROLES = {
    "provider_probe",
    "harness_script",
    "verifier_script",
    "generated_lib_rs",
    "generated_cargo_toml",
    "generated_cargo_lock",
    "native_extension",
}
RUNTIME_IMAGES = {
    "tensorflow_cc": "tensorflow/libtensorflow_cc.so.2",
    "tensorflow_framework": "tensorflow/libtensorflow_framework.so.2",
    "pywrap_tensorflow_common": "tensorflow/python/lib_pywrap_tensorflow_common.so",
}


class EvidenceError(ValueError):
    """Raised when evidence fails canonical schema and integrity verification."""


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text used by the evidence format."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_bytes(envelope: Any) -> bytes:
    """Return the one accepted evidence encoding, including one final newline."""
    return canonical_json(envelope).encode("ascii") + b"\n"


def payload_sha256(payload: dict[str, Any]) -> str:
    """Hash only the canonical payload, avoiding a circular envelope hash."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a producer-selected file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized_relative_label(value: str) -> str:
    """Validate and return a non-secret relative artifact label."""
    text = _string(value, "relative label", pattern=SAFE_RELATIVE)
    if text.startswith("/") or ".." in text.split("/") or text.endswith("/"):
        raise EvidenceError("relative label must be a sanitized relative value")
    return text


def sanitized_wheel_relative(value: str) -> str:
    """Validate and return a path relative to a TensorFlow wheel root."""
    text = sanitized_relative_label(value)
    if not text.startswith("tensorflow/"):
        raise EvidenceError("runtime image path must be relative to the tensorflow/ wheel root")
    return text


def make_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Create the non-circular evidence envelope around a payload."""
    return {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": payload_sha256(payload),
    }


def _closed(value: Any, fields: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{where} must be an object")
    difference = set(value) ^ fields
    if difference:
        raise EvidenceError(f"{where} has unknown or missing fields: {sorted(difference)}")
    return value


def _strict_equal(value: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercions."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _strict_equal(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _strict_equal(actual, item) for actual, item in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _safe_text(value: str, where: str) -> None:
    if len(value) > MAX_STRING:
        raise EvidenceError(f"{where} exceeds the string-size bound")
    if (
        value.startswith("/")
        or WINDOWS_ABSOLUTE.search(value)
        or URL.search(value)
        or CREDENTIAL.search(value)
    ):
        raise EvidenceError(f"{where} contains an absolute path, URL, or credential-looking value")


def _string(
    value: Any,
    where: str,
    *,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{where} must be a non-empty string")
    _safe_text(value, where)
    if pattern is not None and pattern.fullmatch(value) is None:
        raise EvidenceError(f"{where} has invalid format")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise EvidenceError(f"{where} must be finite")
    return float(value)


def _positive_size(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_ARTIFACT_BYTES:
        raise EvidenceError(f"{where} must be a bounded positive integer")


def _walk_constraints(value: Any, where: str = "evidence", depth: int = 0) -> None:
    if depth > MAX_DEPTH:
        raise EvidenceError("evidence exceeds maximum nesting depth")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError("JSON object keys must be strings")
            _safe_text(key, f"{where} key")
            _walk_constraints(item, f"{where}.{key}", depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_constraints(item, f"{where}[{index}]", depth + 1)
    elif isinstance(value, str):
        _safe_text(value, where)
    elif isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError(f"{where} contains a non-finite number")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise EvidenceError(f"{where} contains a non-JSON value")


def _validate_contract(payload: dict[str, Any]) -> None:
    contract = _closed(
        payload["contract"],
        {
            "evidence_schema",
            "verification_scope",
            "producer_assertions",
            "support_claim",
            "certification_ready",
            "plugin_api",
        },
        "payload.contract",
    )
    expected = {
        "evidence_schema": "tensorflow-cuda-e3-real-nvidia-v1",
        "verification_scope": "schema-and-integrity-only",
        "producer_assertions": "self-attested-by-manual-harness",
        "support_claim": False,
        "certification_ready": False,
        "plugin_api": "1.6",
    }
    if not _strict_equal(contract, expected):
        raise EvidenceError(
            "contract must identify self-attested schema/integrity evidence "
            "with support_claim=false and certification_ready=false"
        )


def _validate_identity(payload: dict[str, Any]) -> None:
    package = _closed(
        payload["package"],
        {"distribution", "version", "plugin_module", "native_module"},
        "payload.package",
    )
    if not _strict_equal(
        package,
        {
            "distribution": "rextio-tensorflow",
            "version": "0.1.2",
            "plugin_module": "rextio_tensorflow.plugin",
            "native_module": "_rextio_native",
        },
    ):
        raise EvidenceError("package or module identity does not match the E3 candidate")

    source = _closed(
        payload["source"],
        {
            "core_commit",
            "core_clean",
            "provider_commit",
            "provider_clean",
            "plugin_commit",
            "plugin_clean",
            "base_candidate_commit",
            "plugin_ancestry_checked",
        },
        "payload.source",
    )
    if (
        source["core_commit"] != CORE_COMMIT
        or source["provider_commit"] != PROVIDER_COMMIT
        or source["base_candidate_commit"] != BASE_CANDIDATE_COMMIT
    ):
        raise EvidenceError("source commit bindings do not match the frozen E3 contract")
    _string(source["plugin_commit"], "payload.source.plugin_commit", pattern=GIT_SHA)
    for field in (
        "core_clean",
        "provider_clean",
        "plugin_clean",
        "plugin_ancestry_checked",
    ):
        if source[field] is not True:
            raise EvidenceError(f"payload.source.{field} must be self-attested true")


def _validate_environment(payload: dict[str, Any]) -> None:
    environment = _closed(
        payload["environment"],
        {
            "os",
            "arch",
            "libc",
            "python_implementation",
            "python_version",
            "tensorflow_version",
            "cuda_driver_version",
            "gpu",
        },
        "payload.environment",
    )
    fixed = {
        "os": "Linux",
        "arch": "x86_64",
        "libc": "GNU",
        "python_implementation": "CPython",
        "python_version": "3.11",
        "tensorflow_version": "2.21.0",
    }
    if any(environment[key] != value for key, value in fixed.items()):
        raise EvidenceError("environment does not match the exact E3 platform/runtime contract")
    driver = environment["cuda_driver_version"]
    if isinstance(driver, bool) or not isinstance(driver, int) or not 12_000 <= driver <= 999_999:
        raise EvidenceError("cuda_driver_version must be an integer at least 12000")
    gpu = _closed(environment["gpu"], {"ordinal", "sm"}, "payload.environment.gpu")
    if type(gpu["ordinal"]) is not int or gpu["ordinal"] != 0 or gpu["sm"] not in SMS:
        raise EvidenceError("GPU must be ordinal 0 with an allowed SM")

    toolchain = _closed(
        payload["toolchain"],
        {"rustc_version", "cargo_version", "target"},
        "payload.toolchain",
    )
    if not _strict_equal(
        toolchain,
        {
            "rustc_version": "1.93.1",
            "cargo_version": "1.93.1",
            "target": "x86_64-unknown-linux-gnu",
        },
    ):
        raise EvidenceError("toolchain does not match the exact E3 contract")


def _validate_artifacts(payload: dict[str, Any]) -> None:
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(ARTIFACT_ROLES):
        raise EvidenceError("artifacts must contain the exact seven roles")
    roles: set[str] = set()
    for index, artifact_value in enumerate(artifacts):
        artifact = _closed(
            artifact_value,
            {"role", "label", "sha256", "size_bytes"},
            f"payload.artifacts[{index}]",
        )
        role = _string(artifact["role"], f"payload.artifacts[{index}].role")
        if role in roles:
            raise EvidenceError(f"duplicate artifact role: {role}")
        roles.add(role)
        sanitized_relative_label(artifact["label"])
        _string(
            artifact["sha256"],
            f"payload.artifacts[{index}].sha256",
            pattern=HEX64,
        )
        _positive_size(artifact["size_bytes"], f"payload.artifacts[{index}].size_bytes")
    if roles != ARTIFACT_ROLES:
        raise EvidenceError("artifact roles do not match the exact required set")

    images = payload["runtime_images"]
    if not isinstance(images, list) or len(images) != len(RUNTIME_IMAGES):
        raise EvidenceError("runtime_images must contain exactly three TensorFlow DSOs")
    image_roles: set[str] = set()
    for index, image_value in enumerate(images):
        image = _closed(
            image_value,
            {"role", "wheel_path", "sha256", "size_bytes", "build_id", "mapped"},
            f"payload.runtime_images[{index}]",
        )
        role = _string(image["role"], f"payload.runtime_images[{index}].role")
        if role in image_roles:
            raise EvidenceError(f"duplicate runtime image role: {role}")
        image_roles.add(role)
        if role not in RUNTIME_IMAGES:
            raise EvidenceError(f"unknown runtime image role: {role}")
        path = sanitized_wheel_relative(image["wheel_path"])
        if path != RUNTIME_IMAGES[role]:
            raise EvidenceError(f"runtime image {role} has the wrong wheel-relative path")
        _string(
            image["sha256"],
            f"payload.runtime_images[{index}].sha256",
            pattern=HEX64,
        )
        _positive_size(
            image["size_bytes"],
            f"payload.runtime_images[{index}].size_bytes",
        )
        build_id = image["build_id"]
        if build_id is not None:
            _string(
                build_id,
                f"payload.runtime_images[{index}].build_id",
                pattern=BUILD_ID,
            )
        if image["mapped"] is not True:
            raise EvidenceError(f"runtime image {role} must be self-attested mapped=true")
    if image_roles != set(RUNTIME_IMAGES):
        raise EvidenceError("runtime image roles do not match the exact required set")


def _validate_orchestration(payload: dict[str, Any]) -> None:
    orchestration = _closed(
        payload["orchestration"],
        {
            "provider_id",
            "capability_id",
            "device",
            "input_residency",
            "dtype",
            "ranks",
            "operations",
            "artifact_profile_sha256",
            "authorization_sha256",
            "provider_lock_sha256",
            "probe_sha256",
            "observations_sha256",
        },
        "payload.orchestration",
    )
    fixed = {
        "provider_id": "rextio-device-cuda",
        "capability_id": "cuda-tensorflow-tfe-linux-x86_64",
        "device": "cuda:0",
        "input_residency": "device",
        "dtype": "float32",
        "ranks": [1, 2],
        "operations": OPERATIONS,
    }
    if any(not _strict_equal(orchestration[key], value) for key, value in fixed.items()):
        raise EvidenceError("orchestration does not match the exact E3 slice")
    for field in (
        "artifact_profile_sha256",
        "authorization_sha256",
        "provider_lock_sha256",
        "probe_sha256",
        "observations_sha256",
    ):
        _string(orchestration[field], f"payload.orchestration.{field}", pattern=HEX64)


def _validate_invariants(payload: dict[str, Any]) -> None:
    invariants = _closed(
        payload["invariants"],
        {"execution", "numerical", "output", "lifetime", "negative_boundary"},
        "payload.invariants",
    )
    execution = _closed(
        invariants["execution"],
        {
            "native_extension_executed",
            "kernel_activity_verified",
            "runtime_transfer_profiled",
            "runtime_provenance_checked",
        },
        "payload.invariants.execution",
    )
    if not _strict_equal(
        execution,
        {
            "native_extension_executed": True,
            "kernel_activity_verified": False,
            "runtime_transfer_profiled": False,
            "runtime_provenance_checked": True,
        },
    ):
        raise EvidenceError("first-stage execution must not claim kernel or transfer profiling")

    numerical = _closed(
        invariants["numerical"],
        {"reference", "atol", "rtol", "max_scaled_error"},
        "payload.invariants.numerical",
    )
    if (
        numerical["reference"] != "tensorflow-eager"
        or _finite(numerical["atol"], "payload.invariants.numerical.atol") != ATOL
        or _finite(numerical["rtol"], "payload.invariants.numerical.rtol") != RTOL
    ):
        raise EvidenceError("numerical tolerances must be the exact approved values")
    scaled = _finite(
        numerical["max_scaled_error"],
        "payload.invariants.numerical.max_scaled_error",
    )
    if not 0 <= scaled <= 1:
        raise EvidenceError("max_scaled_error must be in the closed interval [0, 1]")

    output = _closed(
        invariants["output"],
        {"device", "dtype", "rank", "shape"},
        "payload.invariants.output",
    )
    if not _strict_equal(
        output,
        {"device": "GPU:0", "dtype": "float32", "rank": 1, "shape": [4]},
    ):
        raise EvidenceError("output must be exact GPU:0 float32 rank-1 shape [4]")

    lifetime = _closed(
        invariants["lifetime"],
        {"inputs_unchanged", "output_survives_input_gc", "repeated_calls"},
        "payload.invariants.lifetime",
    )
    if not _strict_equal(
        lifetime,
        {
            "inputs_unchanged": True,
            "output_survives_input_gc": True,
            "repeated_calls": True,
        },
    ):
        raise EvidenceError("lifetime and repetition invariants are incomplete")

    negatives = _closed(
        invariants["negative_boundary"],
        {
            "cpu_input_rejected",
            "float64_rejected",
            "wrong_rank_rejected",
            "watched_tape_rejected",
            "forward_accumulator_rejected",
        },
        "payload.invariants.negative_boundary",
    )
    if any(value is not True for value in negatives.values()):
        raise EvidenceError("negative boundary self-attestations are incomplete")


def validate_envelope(envelope: Any) -> dict[str, Any]:
    """Verify the closed schema and payload checksum; return the payload.

    This verifies schema and internal integrity only. It intentionally does not
    authenticate the producer, recompute artifact hashes, or prove execution.
    """
    _walk_constraints(envelope)
    document = _closed(
        envelope,
        {"schema_version", "payload", "payload_sha256"},
        "envelope",
    )
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise EvidenceError("unsupported evidence schema_version")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise EvidenceError("envelope.payload must be an object")
    claimed_hash = _string(
        document["payload_sha256"],
        "envelope.payload_sha256",
        pattern=HEX64,
    )
    if claimed_hash != payload_sha256(payload):
        raise EvidenceError("payload_sha256 does not match the canonical payload")

    _closed(
        payload,
        {
            "contract",
            "package",
            "source",
            "environment",
            "toolchain",
            "artifacts",
            "runtime_images",
            "orchestration",
            "invariants",
        },
        "payload",
    )
    _validate_contract(payload)
    _validate_identity(payload)
    _validate_environment(payload)
    _validate_artifacts(payload)
    _validate_orchestration(payload)
    _validate_invariants(payload)
    return payload


def validate_document(raw: bytes) -> dict[str, Any]:
    """Verify canonical raw bytes plus the closed schema and integrity checksum."""
    if len(raw) > MAX_BYTES:
        raise EvidenceError("evidence exceeds maximum size")
    try:
        text = raw.decode("utf-8")
        envelope = json.loads(
            text,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvidenceError(f"non-finite JSON value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"malformed evidence JSON: {exc}") from exc
    try:
        expected = canonical_bytes(envelope)
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"evidence cannot be canonicalized: {exc}") from exc
    if raw != expected:
        raise EvidenceError("evidence bytes are not canonical JSON with exactly one final newline")
    return validate_envelope(envelope)


def main(argv: list[str] | None = None) -> int:
    """Run the offline schema and integrity verifier CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evidence",
        type=Path,
        help="canonical CUDA E3 evidence JSON path",
    )
    args = parser.parse_args(argv)
    try:
        payload = validate_document(args.evidence.read_bytes())
    except (OSError, EvidenceError) as exc:
        print(f"evidence schema/integrity verification failed: {exc}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "certification_ready": False,
                "payload_sha256": payload_sha256(payload),
                "schema_verified": True,
                "support_claim": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
