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
import math
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
        h = tf.nn.tanh(h)
    probabilities = tf.nn.softmax(h + bias, axis=1)
    return tf.argmax(probabilities, axis=1)


def classify_with_class_input(
    logits: TensorF32Cpu2D,
    classes: TensorI64Cpu1D,
) -> TensorI64Cpu1D:
    # The classes parameter intentionally exercises its exact native boundary;
    # returning a newly computed tensor preserves Python/native identity semantics.
    return tf.argmax(tf.nn.softmax(logits, axis=1), axis=1)


def reduce_sum_axis1(x: TensorF32Cpu2D) -> TensorF32Cpu1D:
    return tf.reduce_sum(x, axis=1, keepdims=False)


def multiply_1d(left: TensorF32Cpu1D, right: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return left * right


def multiply_2d(left: TensorF32Cpu2D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return left * right


def multiply_2d_1d(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return left * right


def multiply_1d_2d(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return left * right


def rank1_activations(x: TensorF32Cpu1D) -> TensorF32Cpu1D:
    return tf.nn.tanh(tf.nn.sigmoid(tf.nn.relu(x)))


def subtract_2d_1d(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return tf.subtract(left, right)


def subtract_1d_2d(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return left - right


def divide_2d_1d(left: TensorF32Cpu2D, right: TensorF32Cpu1D) -> TensorF32Cpu2D:
    return tf.divide(left, right)


def divide_1d_2d(left: TensorF32Cpu1D, right: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return left / right


def reduce_mean_axis0_keepdims(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.reduce_mean(x, 0, keepdims=True)


def reduce_sum_axis1_keepdims(x: TensorF32Cpu2D) -> TensorF32Cpu2D:
    return tf.reduce_sum(x, axis=1, keepdims=True)


def argmax_axis0(x: TensorF32Cpu2D) -> TensorI64Cpu1D:
    return tf.argmax(x, 0)


def cpu_surface_vertical(
    matrix: TensorF32Cpu2D,
    gate_input: TensorF32Cpu1D,
    bias: TensorF32Cpu1D,
) -> TensorI64Cpu1D:
    gate = tf.nn.tanh(tf.nn.sigmoid(tf.nn.relu(gate_input)))
    multiplied = tf.multiply(matrix, gate)
    shifted = multiplied - bias
    divided = tf.divide(gate, shifted)
    column_summary = tf.reduce_mean(divided, 0, keepdims=True)
    row_summary = tf.reduce_sum(column_summary, axis=1, keepdims=True)
    return tf.argmax(row_summary, 0)
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
        hidden = tf.nn.tanh(hidden)
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
    assert len(claims) == 8
    claim_rules = {claim["rule_id"] for claim in claims}
    assert "rextio-tensorflow/matmul-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/relu-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/sigmoid-f32-cpu-2d" in claim_rules
    assert "rextio-tensorflow/tanh-f32-cpu-2d" in claim_rules
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
    assert "rextio_tensorflow_runtime::tanh" in rust
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


def test_reduce_sum_axis1_real_cargo(project: CertifiedProject) -> None:
    """Certify Sum's axis-handle ownership, edge semantics, and boundary guards."""
    tf = _import_tf()
    record = _route_of(project, "tf_app.kernels.reduce_sum_axis1")
    assert record["native_status"] == "accepted"
    assert record["route"] == "native-plugin:rextio-tensorflow"
    assert [claim["rule_id"] for claim in record.get("plugin_claims") or []] == [
        "rextio-tensorflow/reduce-sum-axis1-f32-cpu-2d"
    ]

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "rextio_tensorflow_runtime::reduce_sum_axis1" in rust
    assert "fn reduce_axis(" in rust
    assert "OwnedOp::new(Rc::clone(&context), op_name, &status)" in rust
    assert 'reduce_axis(input, "Sum", 1, false)' in rust
    assert "TFE_NewTensorHandle(reduction axis)" in rust

    regular = tf.constant([[1.5, -2.0, 3.0], [-4.0, 0.5, 1.0]], dtype=tf.float32)
    regular_snapshot = tf.identity(regular)
    checker = project.equivalence_checker(
        "tf_app.kernels.reduce_sum_axis1",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )
    native_regular = checker(regular)
    eager_regular = tf.reduce_sum(regular_snapshot, axis=1, keepdims=False)
    assert _tensor_equal(native_regular, eager_regular)
    assert _tensor_equal(regular, regular_snapshot)
    assert native_regular.dtype == tf.float32
    assert native_regular.shape == (2,)
    assert "CPU" in native_regular.device

    # Empty axis lanes retain rank-1 output and use TensorFlow's own Sum semantics.
    empty = tf.constant([[], []], shape=(2, 0), dtype=tf.float32)
    with _native_mode(project, "native"):
        from tf_app.kernels import reduce_sum_axis1

        native_empty = reduce_sum_axis1(empty)
    eager_empty = tf.reduce_sum(empty, axis=1, keepdims=False)
    assert _tensor_equal(native_empty, eager_empty)
    assert native_empty.shape == (2,)

    # Compare exceptional floating values by class and preserve zero's sign exactly
    # relative to eager TensorFlow, rather than treating NaN as ordinary equality.
    special = tf.constant(
        [[float("nan"), 1.0], [float("inf"), float("-inf")], [-0.0, 0.0]],
        dtype=tf.float32,
    )
    with _native_mode(project, "native"):
        from tf_app.kernels import reduce_sum_axis1

        native_special = reduce_sum_axis1(special)
    eager_special = tf.reduce_sum(special, axis=1, keepdims=False)
    native_values = native_special.numpy().tolist()
    eager_values = eager_special.numpy().tolist()
    assert math.isnan(native_values[0]) and math.isnan(eager_values[0])
    assert math.isnan(native_values[1]) and math.isnan(eager_values[1])
    assert native_values[2] == eager_values[2] == 0.0
    assert math.copysign(1.0, native_values[2]) == math.copysign(1.0, eager_values[2])

    with _native_mode(project, "native"):
        from tf_app.kernels import reduce_sum_axis1

        with pytest.raises(Exception, match="expected a float32 tensor"):
            reduce_sum_axis1(tf.constant([[1.0]], dtype=tf.float64))
        with pytest.raises(Exception, match="rank-2"):
            reduce_sum_axis1(tf.constant([1.0], dtype=tf.float32))

    held_input = tf.identity(regular_snapshot)
    with _native_mode(project, "native"):
        from tf_app.kernels import reduce_sum_axis1

        held = reduce_sum_axis1(held_input)
    del held_input
    import gc

    gc.collect()
    assert _tensor_equal(held, eager_regular)


def test_multiply_real_cargo(project: CertifiedProject) -> None:
    """Certify all bounded Mul rank pairs and fail-closed runtime behavior."""
    tf = _import_tf()
    rule_id = "rextio-tensorflow/mul-binop-f32-cpu"
    for qualname in (
        "tf_app.kernels.multiply_1d",
        "tf_app.kernels.multiply_2d",
        "tf_app.kernels.multiply_2d_1d",
        "tf_app.kernels.multiply_1d_2d",
    ):
        record = _route_of(project, qualname)
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-tensorflow"
        assert [claim["rule_id"] for claim in record.get("plugin_claims") or []] == [rule_id]

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert "rextio_tensorflow_runtime::mul" in rust
    assert 'binary(left, right, "Mul", false, expected_rank)' in rust

    vector_left = tf.constant([2.0, -3.0, 0.5], dtype=tf.float32)
    vector_right = tf.constant([4.0, -2.0, 8.0], dtype=tf.float32)
    matrix_left = tf.constant([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=tf.float32)
    matrix_right = tf.constant([[2.0, 3.0, -1.0], [0.25, -4.0, 2.0]], dtype=tf.float32)

    cases = (
        ("tf_app.kernels.multiply_1d", (vector_left, vector_right)),
        ("tf_app.kernels.multiply_2d", (matrix_left, matrix_right)),
        ("tf_app.kernels.multiply_2d_1d", (matrix_left, vector_right)),
        ("tf_app.kernels.multiply_1d_2d", (vector_right, matrix_left)),
    )
    for qualname, args in cases:
        snapshots = _copy_tensor_args(args)
        checker = project.equivalence_checker(
            qualname,
            equals=_tensor_equal,
            args_equals=_args_unmutated,
            copy_args=_copy_tensor_args,
        )
        native = checker(*args)
        eager = snapshots[0] * snapshots[1]
        assert _tensor_equal(native, eager)
        assert native.dtype == tf.float32
        assert "CPU" in native.device
        assert all(_tensor_equal(arg, snapshot) for arg, snapshot in zip(args, snapshots))

    # Special values keep TensorFlow's classes and zero signs exactly.
    special_left = tf.constant([0.0, -0.0, float("nan"), 2.0], dtype=tf.float32)
    special_right = tf.constant([float("inf"), 2.0, 1.0, float("inf")], dtype=tf.float32)
    with _native_mode(project, "native"):
        from tf_app.kernels import multiply_1d

        native_special = multiply_1d(special_left, special_right)
    eager_special = special_left * special_right
    native_values = native_special.numpy().tolist()
    eager_values = eager_special.numpy().tolist()
    assert math.isnan(native_values[0]) and math.isnan(eager_values[0])
    assert native_values[1] == eager_values[1] == 0.0
    assert math.copysign(1.0, native_values[1]) == math.copysign(1.0, eager_values[1])
    assert math.isnan(native_values[2]) and math.isnan(eager_values[2])
    assert native_values[3] == eager_values[3] == float("inf")

    with _native_mode(project, "native"):
        from tf_app.kernels import multiply_2d_1d

        with pytest.raises(Exception):
            multiply_2d_1d(matrix_left, tf.constant([1.0, 2.0], dtype=tf.float32))
        with pytest.raises(Exception, match="expected a float32 tensor"):
            multiply_2d_1d(tf.cast(matrix_left, tf.float64), vector_right)
        with pytest.raises(Exception, match="rank-2"):
            multiply_2d_1d(vector_left, vector_right)

    left_live = tf.identity(matrix_left)
    right_live = tf.identity(vector_right)
    with _native_mode(project, "native"):
        from tf_app.kernels import multiply_2d_1d

        held = multiply_2d_1d(left_live, right_live)
    expected_held = matrix_left * vector_right
    del left_live, right_live
    import gc

    gc.collect()
    assert _tensor_equal(held, expected_held)


def test_expanded_cpu_surface_real_cargo(project: CertifiedProject) -> None:
    """Certify the new unary/binary/axis surface in one owned TFE vertical slice."""
    tf = _import_tf()
    expected_rules = {
        "tf_app.kernels.rank1_activations": {
            "rextio-tensorflow/relu-f32-cpu-1d",
            "rextio-tensorflow/sigmoid-f32-cpu-1d",
            "rextio-tensorflow/tanh-f32-cpu-1d",
        },
        "tf_app.kernels.subtract_2d_1d": {
            "rextio-tensorflow/sub-call-f32-cpu"
        },
        "tf_app.kernels.subtract_1d_2d": {
            "rextio-tensorflow/sub-binop-f32-cpu"
        },
        "tf_app.kernels.divide_2d_1d": {
            "rextio-tensorflow/div-call-f32-cpu"
        },
        "tf_app.kernels.divide_1d_2d": {
            "rextio-tensorflow/div-binop-f32-cpu"
        },
        "tf_app.kernels.reduce_mean_axis0_keepdims": {
            "rextio-tensorflow/reduce-mean-literal-axis-f32-cpu-2d"
        },
        "tf_app.kernels.reduce_sum_axis1_keepdims": {
            "rextio-tensorflow/reduce-sum-literal-axis-f32-cpu-2d"
        },
        "tf_app.kernels.argmax_axis0": {
            "rextio-tensorflow/argmax-axis0-i64-cpu-2d"
        },
    }
    for qualname, rules in expected_rules.items():
        record = _route_of(project, qualname)
        assert record["native_status"] == "accepted"
        assert record["route"] == "native-plugin:rextio-tensorflow"
        assert {claim["rule_id"] for claim in record.get("plugin_claims") or []} == rules

    vertical_record = _route_of(project, "tf_app.kernels.cpu_surface_vertical")
    assert vertical_record["native_status"] == "accepted"
    vertical_rules = {
        claim["rule_id"] for claim in vertical_record.get("plugin_claims") or []
    }
    assert {
        "rextio-tensorflow/relu-f32-cpu-1d",
        "rextio-tensorflow/sigmoid-f32-cpu-1d",
        "rextio-tensorflow/tanh-f32-cpu-1d",
        "rextio-tensorflow/mul-call-f32-cpu",
        "rextio-tensorflow/sub-binop-f32-cpu",
        "rextio-tensorflow/div-call-f32-cpu",
        "rextio-tensorflow/reduce-mean-literal-axis-f32-cpu-2d",
        "rextio-tensorflow/reduce-sum-literal-axis-f32-cpu-2d",
        "rextio-tensorflow/argmax-axis0-i64-cpu-2d",
    }.issubset(vertical_rules)

    rust = (project.project_root / ".rextio" / "generated" / "rust" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    assert 'binary(left, right, "Sub", false, expected_rank)' in rust
    assert 'binary(left, right, "RealDiv", false, expected_rank)' in rust
    assert 'reduce_axis(input, "Mean", 0, true)' in rust
    assert 'reduce_axis(input, "Sum", 1, true)' in rust
    assert "argmax(input, 0)" in rust
    assert "input.validate_f32(expected_rank)?" in rust
    assert "TFE_TensorHandleBackingDeviceName" in rust
    assert "TFE_OpSetDevice" in rust
    assert "Rc<OwnedTensorHandle>" in rust
    assert "TFE_TensorHandleResolve" not in rust
    assert "TFE_NewContext" not in rust

    vector = tf.constant([1.0, -2.0, 3.0], dtype=tf.float32)
    divisor = tf.constant([2.0, -4.0, 0.5], dtype=tf.float32)
    matrix = tf.constant(
        [[2.0, -8.0, 1.5], [4.0, 12.0, -0.5]], dtype=tf.float32
    )
    matrix_other = tf.constant(
        [[0.5, -1.0, 2.0], [3.0, 4.0, -2.0]], dtype=tf.float32
    )

    finite_cases = (
        (
            "tf_app.kernels.rank1_activations",
            (vector,),
            lambda args: tf.nn.tanh(tf.nn.sigmoid(tf.nn.relu(args[0]))),
        ),
        (
            "tf_app.kernels.subtract_2d_1d",
            (matrix, vector),
            lambda args: tf.subtract(args[0], args[1]),
        ),
        (
            "tf_app.kernels.subtract_1d_2d",
            (vector, matrix),
            lambda args: args[0] - args[1],
        ),
        (
            "tf_app.kernels.divide_2d_1d",
            (matrix, divisor),
            lambda args: tf.divide(args[0], args[1]),
        ),
        (
            "tf_app.kernels.divide_1d_2d",
            (divisor, matrix),
            lambda args: args[0] / args[1],
        ),
        (
            "tf_app.kernels.reduce_mean_axis0_keepdims",
            (matrix,),
            lambda args: tf.reduce_mean(args[0], 0, keepdims=True),
        ),
        (
            "tf_app.kernels.reduce_sum_axis1_keepdims",
            (matrix,),
            lambda args: tf.reduce_sum(args[0], axis=1, keepdims=True),
        ),
        (
            "tf_app.kernels.argmax_axis0",
            (matrix,),
            lambda args: tf.argmax(args[0], 0),
        ),
    )
    for qualname, args, eager_call in finite_cases:
        snapshots = _copy_tensor_args(args)
        checker = project.equivalence_checker(
            qualname,
            equals=_tensor_equal,
            args_equals=_args_unmutated,
            copy_args=_copy_tensor_args,
        )
        native = checker(*args)
        eager = eager_call(snapshots)
        assert _tensor_equal(native, eager)
        assert "CPU" in native.device
        assert all(
            _tensor_equal(arg, snapshot)
            for arg, snapshot in zip(args, snapshots)
        )

    assert project.equivalence_checker(
        "tf_app.kernels.cpu_surface_vertical",
        equals=_tensor_equal,
        args_equals=_args_unmutated,
        copy_args=_copy_tensor_args,
    )(matrix_other, vector, divisor).dtype == tf.int64

    special_left = tf.constant(
        [0.0, -0.0, float("nan"), 1.0, float("inf")], dtype=tf.float32
    )
    special_right = tf.constant(
        [float("-0.0"), 2.0, 1.0, 0.0, float("inf")], dtype=tf.float32
    )
    with _native_mode(project, "native"):
        from tf_app.kernels import divide_1d_2d, subtract_1d_2d

        native_sub = subtract_1d_2d(
            special_left,
            tf.stack((special_right, special_right)),
        )
        native_div = divide_1d_2d(
            special_left,
            tf.stack((special_right, special_right)),
        )
    eager_sub = special_left - tf.stack((special_right, special_right))
    eager_div = special_left / tf.stack((special_right, special_right))
    for native, eager in ((native_sub, eager_sub), (native_div, eager_div)):
        native_values = native.numpy().ravel().tolist()
        eager_values = eager.numpy().ravel().tolist()
        for native_value, eager_value in zip(
            native_values, eager_values, strict=True
        ):
            if math.isnan(eager_value):
                assert math.isnan(native_value)
                continue
            assert native_value == eager_value
            if eager_value == 0.0:
                assert math.copysign(1.0, native_value) == math.copysign(
                    1.0, eager_value
                )

    with _native_mode(project, "native"):
        from tf_app.kernels import (
            cpu_surface_vertical,
            divide_2d_1d,
            rank1_activations,
        )

        with pytest.raises(Exception):
            divide_2d_1d(
                matrix,
                tf.constant([1.0, 2.0], dtype=tf.float32),
            )
        with pytest.raises(Exception, match="expected a float32 tensor"):
            rank1_activations(tf.cast(vector, tf.float64))
        with pytest.raises(Exception, match="rank-1"):
            rank1_activations(matrix)

        matrix_live = tf.identity(matrix_other)
        gate_live = tf.identity(vector)
        bias_live = tf.identity(divisor)
        held = cpu_surface_vertical(matrix_live, gate_live, bias_live)
        expected_held = tf.argmax(
            tf.reduce_sum(
                tf.reduce_mean(
                    tf.divide(
                        tf.nn.tanh(tf.nn.sigmoid(tf.nn.relu(gate_live))),
                        (
                            tf.multiply(
                                matrix_live,
                                tf.nn.tanh(tf.nn.sigmoid(tf.nn.relu(gate_live))),
                            )
                            - bias_live
                        ),
                    ),
                    0,
                    keepdims=True,
                ),
                axis=1,
                keepdims=True,
            ),
            0,
        )
    del matrix_live, gate_live, bias_live
    import gc

    gc.collect()
    assert isinstance(held, tf.Tensor)
    assert "CPU" in held.device
    assert held.dtype == tf.int64
    assert _tensor_equal(held, expected_held)


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
