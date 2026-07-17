"""Machine-readable platform truth contract for the public Alpha."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "ci" / "platform-contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
PROFILES = CONTRACT["profiles"]

EXPECTED_PROFILE_IDS = {
    "linux-x86_64",
    "linux-aarch64",
    "linux-i686",
    "linux-armv7",
    "macos-arm64",
    "macos-x86_64",
    "macos-i686",
    "macos-armv7",
}

SUPPORTED_CFGS = {
    ("macos", "aarch64", ""),
    ("linux", "x86_64", "gnu"),
    ("linux", "aarch64", "gnu"),
}


def _selected_profiles() -> list[dict[str, object]]:
    requested = os.environ.get("REXTIO_PLATFORM_PROFILE")
    if requested is None:
        return PROFILES
    selected = [profile for profile in PROFILES if profile["id"] == requested]
    if not selected:
        raise AssertionError(f"unknown REXTIO_PLATFORM_PROFILE: {requested}")
    return selected


def test_contract_has_exact_runtime_and_requested_matrix() -> None:
    """All requested OS/architecture cells exist under the exact ABI pin."""
    assert CONTRACT["schema_version"] == 1
    assert CONTRACT["runtime_pin"] == {
        "python_implementation": "CPython",
        "python_version": "3.11",
        "tensorflow_version": "2.21.0",
    }
    assert {profile["id"] for profile in PROFILES} == EXPECTED_PROFILE_IDS
    assert len(PROFILES) == len(EXPECTED_PROFILE_IDS)
    assert len({profile["target_triple"] for profile in PROFILES}) == len(PROFILES)


@pytest.mark.parametrize("profile", _selected_profiles(), ids=lambda item: str(item["id"]))
def test_profile_matches_generated_runtime_or_expected_fail_closed(
    profile: dict[str, object],
) -> None:
    """Runtime-backed cells are explicit; every other cell hits the hard guard."""
    helper = runtime_module_helpers()
    profile_id = str(profile["id"])
    support_class = str(profile["support_class"])
    required_result = str(profile["required_result"])
    rust_cfg = profile["rust_cfg"]
    cfg_key = (
        str(rust_cfg["target_os"]),
        str(rust_cfg["target_arch"]),
        str(rust_cfg["target_env"]),
    )

    assert profile["ci_class"] in {
        "hosted-native-e2e",
        "manual-availability-gated-native-e2e",
        "expected-unsupported",
    }
    if profile["runtime_profile"]:
        assert cfg_key in SUPPORTED_CFGS
        assert f'id: "{profile_id}"' in helper
        assert f'support_class: "{support_class}"' in helper
        assert f'target_os = "{cfg_key[0]}"' in helper
        assert f'target_arch = "{cfg_key[1]}"' in helper
        if cfg_key[2]:
            assert f'target_env = "{cfg_key[2]}"' in helper
        assert required_result.startswith("real-cargo-e2e")
    else:
        assert cfg_key not in SUPPORTED_CFGS
        assert f'id: "{profile_id}"' not in helper
        assert required_result.startswith("native-build-fail-closed")
        assert "compile_error!" in helper
        assert "unsupported compile target" in helper
        assert "fail closed at native build" in helper
        # The hard guard is the negation of exactly the three supported cfgs;
        # this profile's concrete cfg therefore reaches compile_error!.
        assert "#[cfg(not(any(" in helper
        assert 'all(target_os = "macos", target_arch = "aarch64")' in helper
        assert (
            'all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")'
            in helper
        )
        assert (
            'all(target_os = "linux", target_arch = "aarch64", target_env = "gnu")'
            in helper
        )


def test_no_unsupported_cell_is_mislabeled_as_certified() -> None:
    """Availability and compile-only evidence never become a support claim."""
    for profile in PROFILES:
        if not profile["runtime_profile"]:
            assert profile["support_class"] != "certified"
            assert profile["ci_class"] == "expected-unsupported"
