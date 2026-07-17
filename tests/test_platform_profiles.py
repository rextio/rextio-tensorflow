"""Source-contract tests for platform ABI profiles and unsupported targets.

Wheel artifact evidence (inspected offline, not installed into the host):

| Wheel | Images | Private bridge symbols (nm / dynsym) |
| --- | --- | --- |
| ``tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl`` | ``libtensorflow_cc.2.dylib``, ``libtensorflow_framework.2.dylib``, ``python/lib_pywrap_tensorflow_common.dylib`` | same three Itanium mangled EagerTensor_* exports |
| ``tensorflow-2.21.0-cp311-cp311-manylinux_2_27_x86_64.whl`` | ``libtensorflow_cc.so.2``, ``libtensorflow_framework.so.2``, ``python/lib_pywrap_tensorflow_common.so`` | same three exports |
| ``tensorflow-2.21.0-cp311-cp311-manylinux_2_27_aarch64.whl`` | same Linux basenames as x86_64 | same three exports |

Verified mangled names (identical across the three wheels above):

- ``_Z18EagerTensor_HandlePK7_object``
- ``_Z21EagerTensorFromHandleP16TFE_TensorHandleb``
- ``_Z22EagerTensor_CheckExactPK7_object``

Linux profiles require ``target_env = "gnu"`` (official manylinux / glibc wheels).
Musl, Windows, and other triples fail closed at **native build** via
``compile_error!`` (not a runtime dlfcn success path).

Linux status/tensor symbols (``TF_NewStatus``, ``TF_AllocateTensor``, …) export
from ``libtensorflow_framework.so.2``; TFE ops (``TFE_NewOp``, ``TFE_Execute``,
…) export from ``libtensorflow_cc.so.2`` — matching the existing resolve map.
"""

from __future__ import annotations

from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

PRIVATE_BRIDGE_SYMBOLS = (
    "_Z18EagerTensor_HandlePK7_object",
    "_Z21EagerTensorFromHandleP16TFE_TensorHandleb",
    "_Z22EagerTensor_CheckExactPK7_object",
)

# Official tensorflow==2.21.0 wheel filenames inspected for this Alpha drop.
WHEEL_ARTIFACTS = (
    "tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl",
    "tensorflow-2.21.0-cp311-cp311-manylinux_2_27_x86_64.whl",
    "tensorflow-2.21.0-cp311-cp311-manylinux_2_27_aarch64.whl",
)


def test_platform_profiles_are_explicit_not_darwin_only_literals() -> None:
    helper = runtime_module_helpers()
    assert "struct PlatformAbiProfile" in helper
    assert 'id: "macos-arm64"' in helper
    assert 'support_class: "certified"' in helper
    assert 'id: "linux-x86_64"' in helper
    assert 'id: "linux-aarch64"' in helper
    assert 'support_class: "experimental"' in helper
    assert "PLATFORM_ABI_PROFILE" in helper
    # Active-wheel paths come from the profile, not hard-coded Darwin joins.
    assert "tensorflow_root.join(profile.cc_library)" in helper
    assert "tensorflow_root.join(profile.framework_library)" in helper
    assert "tensorflow_root.join(profile.pywrap_library)" in helper
    assert 'join("libtensorflow_cc.2.dylib")' not in helper


def test_macos_certified_profile_paths_and_rtld_flags() -> None:
    helper = runtime_module_helpers()
    assert 'cc_library: "libtensorflow_cc.2.dylib"' in helper
    assert 'framework_library: "libtensorflow_framework.2.dylib"' in helper
    assert 'pywrap_library: "python/lib_pywrap_tensorflow_common.dylib"' in helper
    # Darwin dlfcn.h values
    assert "rtld_now: 0x2" in helper
    assert "rtld_local: 0x4" in helper
    assert "rtld_noload: 0x10" in helper
    assert 'python_machines: &["arm64"]' in helper


def test_linux_experimental_profiles_require_target_env_gnu() -> None:
    helper = runtime_module_helpers()
    gnu_x64 = 'all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")'
    gnu_arm = 'all(target_os = "linux", target_arch = "aarch64", target_env = "gnu")'
    assert gnu_x64 in helper
    assert gnu_arm in helper
    # Linux profile consts must not be gated on arch alone (musl must not match).
    assert '#[cfg(all(target_os = "linux", target_arch = "x86_64"))]' not in helper
    assert '#[cfg(all(target_os = "linux", target_arch = "aarch64"))]' not in helper
    assert 'cc_library: "libtensorflow_cc.so.2"' in helper
    assert 'framework_library: "libtensorflow_framework.so.2"' in helper
    assert 'pywrap_library: "python/lib_pywrap_tensorflow_common.so"' in helper
    # Linux glibc values: RTLD_LOCAL is 0; RTLD_NOLOAD is 0x4 (≠ Darwin 0x10).
    assert "rtld_local: 0x0" in helper
    assert "rtld_noload: 0x4" in helper
    assert 'python_machines: &["x86_64"]' in helper
    assert 'python_machines: &["aarch64", "arm64"]' in helper
    assert "glibc" in helper
    assert "musl" in helper


def test_unsupported_targets_use_compile_error_not_runtime_dlfcn_promise() -> None:
    helper = runtime_module_helpers()
    assert "compile_error!" in helper
    assert "unsupported compile target" in helper
    assert "musl" in helper
    assert "Windows" in helper
    assert "native build" in helper
    # No Option::None runtime fallback profile for unsupported triples.
    assert "PLATFORM_ABI_PROFILE: Option<PlatformAbiProfile> = None" not in helper
    # Runtime validate_platform documents compile-time fail-closed.
    assert "compile_error!" in helper
    assert "fail closed at native build" in helper


def test_linux_gnu_links_libdl_for_dlfcn() -> None:
    helper = runtime_module_helpers()
    assert 'link(name = "dl")' in helper
    assert 'all(target_os = "linux", target_env = "gnu")' in helper
    assert "cfg_attr" in helper
    assert "link:linux-gnu:libdl" in helper


def test_private_bridge_symbols_shared_across_verified_profiles() -> None:
    helper = runtime_module_helpers()
    for symbol in PRIVATE_BRIDGE_SYMBOLS:
        assert symbol in helper
    assert "SYM_EAGER_TENSOR_HANDLE" in helper
    assert "SYM_EAGER_TENSOR_FROM_HANDLE" in helper
    assert "SYM_EAGER_TENSOR_CHECK_EXACT" in helper


def test_same_wheel_reuse_never_uses_rtld_default() -> None:
    helper = runtime_module_helpers()
    assert "RTLD_DEFAULT" not in helper
    assert "RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD" in helper
    assert "dlopen(c_path.as_ptr(), flags)" in helper
    assert "verify_provenance" in helper
    assert "dladdr" in helper


def test_source_contract_records_wheel_artifact_evidence() -> None:
    helper = runtime_module_helpers()
    assert "PLATFORM_PROFILE_SOURCE_CONTRACT" in helper
    assert "profiles:macos-arm64:certified:" in helper
    assert "profiles:linux-x86_64:experimental:target_env=gnu:" in helper
    assert "profiles:linux-aarch64:experimental:target_env=gnu:" in helper
    assert "unsupported:compile_error:windows-musl-and-other-targets-native-build-fail-closed" in helper
    assert len(WHEEL_ARTIFACTS) == 3
    assert all(name.startswith("tensorflow-2.21.0-") for name in WHEEL_ARTIFACTS)
