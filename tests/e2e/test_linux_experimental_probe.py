"""Opt-in Linux GNU experimental probe (not real-Cargo certification).

Enable with::

    REXTIO_TF_LINUX_PROBE=1 pytest tests/e2e/test_linux_experimental_probe.py -q

On non-Linux hosts the module still loads and documents the artifact contract.
The live probe runs only when the env var is set **and** the process is Linux
GNU/glibc x86_64 or aarch64 with ``tensorflow==2.21.0`` importable.

The probe opens the three expected active-wheel images with
``RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD``, resolves representative symbols from
their owning images (including the three private bridge mangled names), and
calls **only** ``TF_Version`` (never private bridge functions). This is not a
certified real-Cargo E2E.
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path

import pytest

from rextio_tensorflow.rust_snippets.runtime import runtime_module_helpers

_LINUX_PROBE = os.environ.get("REXTIO_TF_LINUX_PROBE", "") == "1"

# Linux glibc (target_env=gnu) values from manylinux / bits/dlfcn.h evidence.
# Do not use Darwin RTLD_NOLOAD=0x10 here.
_LINUX_RTLD_NOW = 0x2
_LINUX_RTLD_LOCAL = 0x0
_LINUX_RTLD_NOLOAD = 0x4
_LINUX_DLOPEN_FLAGS = _LINUX_RTLD_NOW | _LINUX_RTLD_LOCAL | _LINUX_RTLD_NOLOAD

_PRIVATE_BRIDGE = (
    "_Z18EagerTensor_HandlePK7_object",
    "_Z21EagerTensorFromHandleP16TFE_TensorHandleb",
    "_Z22EagerTensor_CheckExactPK7_object",
)

_CC_REL = "libtensorflow_cc.so.2"
_FRAMEWORK_REL = "libtensorflow_framework.so.2"
_PYWRAP_REL = "python/lib_pywrap_tensorflow_common.so"


def _is_linux_gnu() -> bool:
    """Official experimental profiles are glibc/manylinux only (not musl)."""
    if sys.platform != "linux":
        return False
    try:
        ctypes.CDLL("libc.so.6")
    except OSError:
        return False
    return True


def test_linux_profile_present_in_helper_without_running_cargo() -> None:
    """Always-on contract: Linux GNU experimental profiles exist in helper text."""
    helper = runtime_module_helpers()
    assert 'id: "linux-x86_64"' in helper
    assert 'id: "linux-aarch64"' in helper
    assert 'support_class: "experimental"' in helper
    assert 'cc_library: "libtensorflow_cc.so.2"' in helper
    assert "rtld_noload: 0x4" in helper  # Linux glibc, not Darwin 0x10
    assert 'target_env = "gnu"' in helper
    assert 'link(name = "dl")' in helper
    assert "compile_error!" in helper


@pytest.mark.skipif(
    not _LINUX_PROBE,
    reason="Set REXTIO_TF_LINUX_PROBE=1 to run the Linux experimental smoke probe",
)
@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only runtime smoke")
@pytest.mark.skipif(not _is_linux_gnu(), reason="Experimental probe is glibc/gnu only (musl unsupported)")
def test_linux_experimental_same_wheel_noload_symbol_probe() -> None:
    """Opt-in live probe: RTLD_NOLOAD open, resolve, TF_Version only.

    Does **not** call private bridge functions and does **not** claim real-Cargo
    certification.
    """
    import tensorflow as tf

    assert tf.__version__ == "2.21.0"
    machine = platform.machine()
    assert machine in {"x86_64", "aarch64", "arm64"}

    helper = runtime_module_helpers()
    if machine == "x86_64":
        assert 'id: "linux-x86_64"' in helper
    else:
        assert 'id: "linux-aarch64"' in helper

    module_file = Path(tf.__file__).resolve()
    wheel_root = module_file.parent
    assert wheel_root.name == "tensorflow"

    cc_path = (wheel_root / _CC_REL).resolve()
    framework_path = (wheel_root / _FRAMEWORK_REL).resolve()
    pywrap_path = (wheel_root / _PYWRAP_REL).resolve()
    for path, label in (
        (cc_path, "libtensorflow_cc"),
        (framework_path, "libtensorflow_framework"),
        (pywrap_path, "lib_pywrap_tensorflow_common"),
    ):
        assert path.is_file(), f"missing active-wheel {label} at {path}"
        assert path == path.resolve()
        # Stay under the active wheel root (same-wheel gate).
        assert str(path).startswith(str(wheel_root))

    # Same-wheel reuse: RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD only (no RTLD_DEFAULT).
    assert _LINUX_DLOPEN_FLAGS == 0x6
    cc = ctypes.CDLL(str(cc_path), mode=_LINUX_DLOPEN_FLAGS)
    framework = ctypes.CDLL(str(framework_path), mode=_LINUX_DLOPEN_FLAGS)
    pywrap = ctypes.CDLL(str(pywrap_path), mode=_LINUX_DLOPEN_FLAGS)

    # Representative public symbols from owning images (resolve only).
    assert getattr(cc, "TF_Version") is not None
    assert getattr(cc, "TFE_NewOp") is not None
    assert getattr(cc, "TFE_Execute") is not None
    assert getattr(framework, "TF_NewStatus") is not None
    assert getattr(framework, "TF_AllocateTensor") is not None

    # Private bridge: resolve all three mangled names; never call them.
    for symbol in _PRIVATE_BRIDGE:
        resolved = getattr(pywrap, symbol)
        assert resolved is not None, f"missing private bridge symbol {symbol}"

    # Safely call only TF_Version.
    tf_version = cc.TF_Version
    tf_version.restype = ctypes.c_char_p
    tf_version.argtypes = []
    version_bytes = tf_version()
    assert version_bytes is not None
    assert version_bytes.decode("utf-8") == "2.21.0"
