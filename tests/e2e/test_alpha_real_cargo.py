"""One real-Cargo E2E certification for the Alpha TF inference slice.

Uses the core certification kit with the entry-point-discoverable plugin.
The invoking interpreter must be CPython 3.11 with TensorFlow 2.21.0.  CI and
local callers may set ``REXTIO_TF_E2E_PYTHON`` explicitly; otherwise the
current interpreter is used. TensorFlow must be imported before the generated
native extension loads so TFE symbols resolve from the already-loaded wheel
images.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from rextio.plugins.testing import CertifiedProject, build_certification_project

from rextio_tensorflow.diagnostics import RUNTIME_ERRORS

E2E_PYTHON = Path(os.environ.get("REXTIO_TF_E2E_PYTHON", sys.executable)).expanduser().absolute()
SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"

pytestmark = [
    pytest.mark.needs_cargo,
    pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="real-Cargo certification requires cargo on PATH",
    ),
    pytest.mark.skipif(
        not E2E_PYTHON.is_file(),
        reason="configured E2E Python interpreter is unavailable",
    ),
]

KERNELS = """
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import tensorflow as tf


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
    depth: int,
    phase: int,
) -> TensorI64Cpu1D:
    h = tf.matmul(x, weight)
    h = tf.nn.relu(h)
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            h = tf.nn.sigmoid(h)
        else:
            h = tf.nn.relu(h)
    probabilities = tf.nn.softmax(h + bias, axis=1)
    return tf.argmax(probabilities, axis=1)


def classify_with_class_input(
    logits: TensorF32Cpu2D,
    classes: TensorI64Cpu1D,
) -> TensorI64Cpu1D:
    # The classes parameter intentionally exercises its exact native boundary;
    # returning a newly computed tensor preserves Python/native identity semantics.
    return tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
"""


def _configure_e2e_env() -> None:
    """Pin build subprocesses to the invoking CPython 3.11 + TF 2.21 environment."""
    assert sys.version_info[:2] == (3, 11), "real-Cargo E2E requires CPython 3.11"
    assert Path(sys.executable).absolute() == E2E_PYTHON, (
        "run pytest with the same interpreter named by REXTIO_TF_E2E_PYTHON"
    )
    e2e_bin = E2E_PYTHON.parent
    os.environ["VIRTUAL_ENV"] = str(Path(sys.prefix).resolve())
    os.environ["PATH"] = f"{e2e_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["PYO3_PYTHON"] = str(E2E_PYTHON)
    # Fail closed if the configured interpreter resolves a mismatched runtime.
    import subprocess

    probe = subprocess.run(
        [
            str(E2E_PYTHON),
            "-c",
            "import platform, sys, tensorflow as tf; "
            "print(sys.implementation.name); print(sys.version_info[0], sys.version_info[1]); "
            "print(tf.__version__); print(platform.system()); print(platform.machine())",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert probe.returncode == 0, probe.stderr
    lines = [line.strip() for line in probe.stdout.strip().splitlines() if line.strip()]
    assert lines[0] == "cpython", f"E2E requires CPython; got {lines[0]!r}"
    assert lines[1] == "3 11", f"E2E requires CPython 3.11; got {lines[1]!r}"
    assert lines[2] == "2.21.0", f"E2E requires TF 2.21.0; got {lines[2]!r}"
    assert (lines[3], lines[4]) in {
        ("Darwin", "arm64"),
        ("Linux", "x86_64"),
        ("Linux", "aarch64"),
        ("Linux", "arm64"),
    }, f"unsupported E2E platform: {(lines[3], lines[4])!r}"


def _import_tf():
    import tensorflow as tf

    return tf


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> CertifiedProject:
    _configure_e2e_env()
    if os.environ.get("REXTIO_TF_REQUIRE_INSTALLED_WHEEL") == "1":
        import rextio_tensorflow

        package_file = Path(rextio_tensorflow.__file__).resolve()
        assert SOURCE_ROOT.resolve() not in package_file.parents, (
            f"native E2E must import the installed wheel, not {SOURCE_ROOT}"
        )
    # Import TF in this process before the native extension is ever loaded.
    _import_tf()

    root = tmp_path_factory.mktemp("tf_alpha")
    (root / "rextio.toml").write_text(
        '[rust]\nbuild_tool = "cargo"\n\n[plugins]\nenabled = ["rextio-tensorflow"]\n',
        encoding="utf-8",
    )
    package = root / "src" / "tf_app"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "kernels.py").write_text(KERNELS, encoding="utf-8")

    # Plugin discovery must use the distribution entry point. CI installs the
    # freshly built wheel; local development may use an editable install.
    return build_certification_project(root)


def _route_of(project: CertifiedProject, qualname: str) -> dict:
    report = json.loads(
        (project.project_root / ".rextio" / "reports" / "check.json").read_text(encoding="utf-8")
    )
    for module in report["modules"]:
        for function in module["functions"]:
            if function["qualname"] == qualname:
                return function
    raise AssertionError(f"{qualname} not found in check.json")


def _tensor_equal(left: object, right: object) -> bool:
    tf = _import_tf()
    if type(left).__name__ != "EagerTensor" and not isinstance(left, tf.Tensor):
        return False
    if type(right).__name__ != "EagerTensor" and not isinstance(right, tf.Tensor):
        return False
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype == tf.int64:
        return bool(tf.reduce_all(tf.equal(left, right)).numpy())
    return bool(tf.reduce_all(tf.abs(left - right) <= 1e-5).numpy())


def _args_unmutated(left: object, right: object) -> bool:
    tf = _import_tf()
    if isinstance(left, tf.Tensor) or isinstance(right, tf.Tensor):
        return _tensor_equal(left, right)
    return left == right


def _eager_reference(
    x: object,
    weight: object,
    bias: object,
    depth: int,
    phase: int,
) -> object:
    tf = _import_tf()
    hidden = tf.nn.relu(tf.matmul(x, weight))
    for layer in range(depth):
        if (layer + phase) % 2 == 0:
            hidden = tf.nn.sigmoid(hidden)
        else:
            hidden = tf.nn.relu(hidden)
    probabilities = tf.nn.softmax(hidden + bias, axis=1)
    return tf.argmax(probabilities, axis=1)


def _copy_tensor_args(args: tuple[object, ...]) -> tuple[object, ...]:
    tf = _import_tf()
    copies: list[object] = []
    for arg in args:
        if isinstance(arg, tf.Tensor):
            copies.append(tf.identity(arg))
        else:
            copies.append(arg)
    return tuple(copies)


def test_alpha_chain_real_cargo_certification(project: CertifiedProject) -> None:
    """Single serialized build: route evidence, native≈eager, contracts, lifetime."""
    tf = _import_tf()
    assert tf.__version__ == "2.21.0"
    assert sys.version_info[:2] == (3, 11)
    tf_path = tf.__file__
    assert tf_path is not None

    build = json.loads(
        (project.project_root / ".rextio" / "reports" / "build.json").read_text(encoding="utf-8")
    )
    assert build["status"] == "built"
    native_build = build.get("native_build") or {}
    assert native_build.get("status") == "built"

    record = _route_of(project, "tf_app.kernels.inference")
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-tensorflow"
    claims = record.get("plugin_claims") or []
    assert len(claims) == 7
    claim_rules = {claim["rule_id"] for claim in claims}
    assert "rextio-tensorflow/matmul-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/relu-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/sigmoid-f32-cpu-2d" in claim_rules
    assert (
        "rextio-tensorflow/add-call-f32-cpu" in claim_rules
        or "rextio-tensorflow/add-binop-f32-cpu" in claim_rules
    )
    assert "rextio-tensorflow/softmax-axis1-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/argmax-axis1-i64-cpu-2d" in claim_rules

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "mod rextio_tensorflow_runtime" in rust
    assert "rextio_tensorflow_runtime::matmul" in rust
    assert "rextio_tensorflow_runtime::relu" in rust
    assert "rextio_tensorflow_runtime::add" in rust
    assert "rextio_tensorflow_runtime::softmax_axis1" in rust
    assert "rextio_tensorflow_runtime::argmax_axis1" in rust
    assert "EagerTensor_Handle" in rust or "_Z18EagerTensor_Handle" in rust
    assert "TFE_Execute" in rust
    assert "RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD" in rust
    assert "PlatformAbiProfile" in rust
    assert "dladdr" in rust
    # Certified macOS profile still present; Linux experimental profiles too.
    assert "libtensorflow_cc.2.dylib" in rust
    assert "libtensorflow_framework.2.dylib" in rust
    assert "lib_pywrap_tensorflow_common.dylib" in rust
    assert "libtensorflow_cc.so.2" in rust
    assert "lib_pywrap_tensorflow_common.so" in rust
    assert 'support_class: "certified"' in rust
    assert 'support_class: "experimental"' in rust
    assert "TFE_TensorHandleBackingDeviceName" in rust
    assert "TFE_OpSetDevice" in rust
    assert 'set_bool("grad_a", false)' in rust
    assert 'set_bool("grad_b", false)' in rust
    assert "TF_AllocateTensor" in rust
    assert "TF_TensorData" in rust
    assert "TF_INT64" in rust
    assert "axis_one_scalar" in rust
    assert "Rc<OwnedTensorHandle>" in rust
    assert "unsafe impl Send" not in rust
    assert "RTLD_DEFAULT" not in rust
    assert ".unwrap()" not in rust
    assert ".expect(" not in rust
    assert "panic!" not in rust
    assert "TFE_NewContext" not in rust  # no duplicate Context
    assert "TF_Session" not in rust
    assert "DLPack" not in rust
    assert "TFE_TensorHandleResolve" not in rust

    # Fixed small CPU float32 fixtures (N=4, in=3, out=2).
    x = tf.constant(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [1.0, -1.1, 1.2]],
        dtype=tf.float32,
    )
    weight = tf.constant(
        [[0.2, -0.3], [0.4, 0.5], [-0.1, 0.6]],
        dtype=tf.float32,
    )
    bias = tf.constant([0.05, -0.05], dtype=tf.float32)
    x_snap = tf.identity(x)
    w_snap = tf.identity(weight)
    b_snap = tf.identity(bias)

    checker = project.equivalence_checker(
        "tf_app.kernels.inference",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    native_out = checker(x, weight, bias, 3, 0)

    assert isinstance(native_out, tf.Tensor)
    assert "CPU" in native_out.device
    assert native_out.dtype == tf.int64
    assert len(native_out.shape) == 1
    assert int(native_out.shape[0]) == 4

    eager = _eager_reference(x_snap, w_snap, b_snap, 3, 0)
    assert _tensor_equal(native_out, eager)

    # Inputs not mutated.
    assert _tensor_equal(x, x_snap)
    assert _tensor_equal(weight, w_snap)
    assert _tensor_equal(bias, b_snap)

    # Exact runtime version/path (active wheel, not an alternate runtime).
    assert tf.__version__ == "2.21.0"
    assert Path(tf_path).exists()

    # Output lifetime after inputs released.
    x_live = tf.identity(x_snap)
    w_live = tf.identity(w_snap)
    b_live = tf.identity(b_snap)
    with _native_mode(project, "native"):
        from tf_app.kernels import inference as inference_live

        held = inference_live(x_live, w_live, b_live, 3, 0)
    del x_live, w_live, b_live
    import gc

    gc.collect()
    assert isinstance(held, tf.Tensor)
    assert "CPU" in held.device
    assert held.dtype == tf.int64
    assert _tensor_equal(held, eager)

    # Fail-closed boundary on annotation-violating runtime values.
    w_ok = tf.identity(w_snap)
    b_ok = tf.identity(b_snap)
    with _native_mode(project, "native"):
        from tf_app.kernels import inference as inference_boundary

        x_f64 = tf.constant(
            [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9], [1.0, -1.1, 1.2]],
            dtype=tf.float64,
        )
        with pytest.raises(Exception) as dtype_info:
            inference_boundary(x_f64, w_ok, b_ok, 3, 0)
        assert RUNTIME_ERRORS["dtype"] in str(dtype_info.value)

        x_rank1 = tf.constant([0.1, -0.2, 0.3], dtype=tf.float32)
        with pytest.raises(Exception) as rank_info:
            inference_boundary(x_rank1, w_ok, b_ok, 3, 0)
        assert "rank-2" in str(rank_info.value)

        with pytest.raises(TypeError, match="expected a TensorFlow EagerTensor"):
            inference_boundary(tf.Variable(x_snap), w_ok, b_ok, 3, 0)

        with pytest.raises(TypeError, match="expected a TensorFlow EagerTensor"):
            inference_boundary(x_snap.numpy(), w_ok, b_ok, 3, 0)

    assert _tensor_equal(w_ok, w_snap)
    assert _tensor_equal(b_ok, b_snap)

    # Repeated calls exercise Rc clones/reassignments, alternating scalar
    # control-flow branches, and output ownership after all inputs are gone.
    repeated: list[tuple[object, object]] = []
    with _native_mode(project, "native"):
        from tf_app.kernels import inference as inference_repeated

        for phase in range(8):
            x_call = tf.identity(x_snap)
            w_call = tf.identity(w_snap)
            b_call = tf.identity(b_snap)
            output = inference_repeated(x_call, w_call, b_call, 4, phase % 2)
            reference = _eager_reference(x_call, w_call, b_call, 4, phase % 2)
            repeated.append((output, reference))
    del x_call, w_call, b_call
    import gc

    gc.collect()
    assert all(_tensor_equal(output, reference) for output, reference in repeated)


def test_i64_parameter_boundary_real_cargo(project: CertifiedProject) -> None:
    """The registered int64 rank-1 parameter extractor compiles and fails closed."""
    tf = _import_tf()
    record = _route_of(project, "tf_app.kernels.classify_with_class_input")
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-tensorflow"
    claims = record.get("plugin_claims") or []
    assert {claim["rule_id"] for claim in claims} == {
        "rextio-tensorflow/softmax-axis1-f32-cpu-2d",
        "rextio-tensorflow/argmax-axis1-i64-cpu-2d",
    }

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "extract_i64_cpu_1d" in rust
    assert "validate_i64" in rust

    logits = tf.constant(
        [[0.2, 0.7, 0.1], [0.9, 0.05, 0.05], [0.1, 0.2, 0.7], [0.4, 0.3, 0.3]],
        dtype=tf.float32,
    )
    classes = tf.constant([2, 0, 1, 2], dtype=tf.int64)
    snapshot = tf.identity(classes)
    expected = tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)
    with _native_mode(project, "native"):
        from tf_app.kernels import classify_with_class_input

        held = classify_with_class_input(logits, classes)

        with pytest.raises(Exception, match="expected an int64 tensor"):
            classify_with_class_input(logits, tf.constant([2, 0, 1, 2], dtype=tf.float32))
        with pytest.raises(Exception, match="expected rank-1 tensor"):
            classify_with_class_input(logits, tf.constant([[2, 0], [1, 2]], dtype=tf.int64))
        with pytest.raises(TypeError, match="expected a TensorFlow EagerTensor"):
            classify_with_class_input(logits, tf.Variable(snapshot))

    assert _tensor_equal(classes, snapshot)
    del classes, logits
    import gc

    gc.collect()
    assert isinstance(held, tf.Tensor)
    assert "CPU" in held.device
    assert held.dtype == tf.int64
    assert held.shape == (4,)
    assert _tensor_equal(held, expected)


class _native_mode:
    """Temporarily force REXTIO_NATIVE_MODE and load the generated package."""

    def __init__(self, project: CertifiedProject, mode: str) -> None:
        self.project = project
        self.mode = mode
        self._previous: str | None = None
        self._inserted = False
        self._build_dir = str(project.build_python_dir)

    def __enter__(self) -> None:
        _import_tf()
        self._previous = os.environ.get("REXTIO_NATIVE_MODE")
        os.environ["REXTIO_NATIVE_MODE"] = self.mode
        for name in list(sys.modules):
            if name == "_rextio_native" or name == "tf_app" or name.startswith("tf_app."):
                sys.modules.pop(name, None)
        if self._build_dir not in sys.path:
            sys.path.insert(0, self._build_dir)
            self._inserted = True

    def __exit__(self, *exc: object) -> None:
        if self._inserted and self._build_dir in sys.path:
            sys.path.remove(self._build_dir)
        for name in list(sys.modules):
            if name == "_rextio_native" or name == "tf_app" or name.startswith("tf_app."):
                sys.modules.pop(name, None)
        if self._previous is None:
            os.environ.pop("REXTIO_NATIVE_MODE", None)
        else:
            os.environ["REXTIO_NATIVE_MODE"] = self._previous
