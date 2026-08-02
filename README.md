# rextio-tensorflow

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio project icon"></p>
<p align="center"><strong>Bounded TensorFlow inference lowering through Rextio's owned thin TFE C API runtime.</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-tensorflow/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-tensorflow?label=PyPI" alt="rextio-tensorflow on PyPI"></a> <a href="https://github.com/rextio/rextio-tensorflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a></p>
<p align="center"><strong>English</strong> · <a href="https://github.com/rextio/rextio-tensorflow/blob/main/README.ko.md">한국어</a> · <a href="https://github.com/rextio/rextio-tensorflow/blob/main/README.zh-hans.md">简体中文</a> · <a href="https://github.com/rextio/rextio-tensorflow/blob/main/README.zh-hant.md">繁體中文</a> · <a href="https://github.com/rextio/rextio-tensorflow/blob/main/README.ja.md">日本語</a></p>

`rextio-tensorflow` is a public native-AOT Alpha proof of concept for [Rextio](https://github.com/rextio/rextio). It lowers a narrow TensorFlow 2.21.0 inference surface into an owned Rust helper that calls the already-loaded wheel's eager TFE C API. It does not reimplement TensorFlow in pure Rust and has no TensorFlow Rust-crate dependency.

> [!IMPORTANT]
> This is an exact-version, inference-only Alpha. Unsupported sites stay on ordinary Python fallback; claimed native sites fail closed on ABI, symbol, version, dtype, rank, device, or eager-context mismatch. There is no transparent runtime retry and no training or speedup claim.

## See the bounded path

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D


def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

Rextio can claim the exact rank/type route, keep intermediates as owned `TFE_TensorHandle` values, and materialize an ordinary `tf.Tensor` at the boundary. Concrete matrix and broadcast compatibility remains TensorFlow's runtime responsibility.

## How it works

```text
typed Python function
        ↓ Rextio API 1.6 claim + revalidation
owned rextio_tensorflow_runtime helper
        ↓ RTLD_NOLOAD + symbol/provenance checks
already-loaded TensorFlow 2.21.0 wheel / synchronous eager TFE context
```

- The runtime reuses the active wheel; it creates no second TensorFlow context.
- Public TFE C calls own operations, statuses, and tensor handles with RAII.
- Function boundaries use three private EagerTensor bridge symbols from the exact 2.21.0 wheel. That private ABI is the principal Alpha risk.
- Import-free marker classes describe CPU/CUDA, dtype, and rank. Runtime values remain ordinary exact EagerTensors.
- Standalone Rust lowering is rejected. The plugin is a PyO3 host-extension integration.

## Install and try

```bash
python3.11 -m pip install 'rextio>=0.1.6,<0.2' 'rextio-tensorflow==0.1.3'
```

The resolver installs the exact `tensorflow==2.21.0` dependency. Put the example above in a Rextio project and use the normal Rextio analysis/build flow. The same CPython 3.11 environment must import TensorFlow, run Rextio, and load the generated extension.

## Compatibility contract

| Component | Contract |
| --- | --- |
| Package | `rextio-tensorflow==0.1.3` (public Alpha, released 2026-07-27) |
| CPython | `>=3.11,<3.12` — 3.11 only |
| Rextio | `>=0.1.6,<0.2` |
| Plugin API | `1.6` |
| TensorFlow Python and C runtime | exactly `2.21.0`; `TF_Version() == "2.21.0"` |
| CPU device | exact `CPU:0` |
| CPU types | float32 rank 1/2; default ArgMax result/input boundary int64 rank 1 |
| Generated runtime | no `tensorflow`/`tensorflow-sys` Cargo dependency |
| Certified toolchain evidence | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` |
| Mode | synchronous eager inference; no training |

### Platform ABI profiles

| Host | Status |
| --- | --- |
| macOS arm64 | **Certified Alpha** real-Cargo path |
| Linux x86_64 GNU/glibc | **Experimental**, hosted real-Cargo E2E; not certified |
| Linux AArch64 GNU/glibc | **Experimental / availability-gated**, manual native E2E |
| macOS x86_64 | Availability-gated and currently unsupported: no exact TF 2.21.0 wheel |
| Windows, Linux musl, i686, ARMv7, other triples | Native build fails closed; unsupported or deferred |

## Supported CPU surface

All operation inputs are registered float32 CPU rank-1/rank-2 exact EagerTensors. Static arguments must match the forms below.

| Family | Accepted forms and boundaries |
| --- | --- |
| Matmul | `tf.matmul` / `tf.linalg.matmul`; rank 2 × 2; no transpose keywords |
| Activations | `tf.nn.relu/sigmoid/tanh`; rank 1 or 2; no keywords |
| Unary math | `tf.abs`, `tf.negative`, `tf.square`, `tf.exp`, `tf.math.log`, `tf.math.sqrt`; one positional tensor, no keywords |
| Elementwise | `tf.add/multiply/subtract/divide`, matching `tf.math.*`, or `+ * - /`; ranks 1/1, 2/2, 2/1, 1/2; no scalar/options |
| Maximum/minimum | top-level `tf.maximum/minimum`; equal rank 1/1 or 2/2 only |
| Mean/sum | `tf.reduce_mean/sum` or `tf.math.*`; rank 2; literal axis 0/1; `keepdims` omitted or named literal bool |
| Softmax | `tf.nn.softmax`; rank 1 axis omitted/0 or rank 2 explicit axis 1 |
| ArgMax | `tf.argmax`; rank-2 float32, explicit literal axis 0/1, default output type; returns int64 rank 1 |
| Transpose | `tf.transpose`; rank 2; default permutation only, no `perm`, `conjugate`, or `name` |
| Bias add | `tf.nn.bias_add`; rank-2 value + rank-1 bias; data format omitted or literal `NHWC` |

Claims prove rank, not concrete dimensions. TFE validates matrix and broadcasting dimensions. `TensorF32Cpu2D`, `TensorF32Cpu1D`, and `TensorI64Cpu1D` are the CPU markers. Tensor-dependent control flow, variables, async eager, and alternate runtimes are outside the surface.

## Fallback and fail-closed behavior

Unrecognized or unsupported calls remain Python fallback at analysis time. Recognized forms with invalid ranks or options are rejected with `RXTP-TENSORFLOW-*` guidance. Lowering independently revalidates rule IDs, ranks, targets, operand counts, and literal options, raising `ValueError` on drift.

Once a site is native, the generated runtime does not replay Python on failure. It requires an already-imported exact TensorFlow 2.21.0 wheel, synchronous eager mode, exact EagerTensor objects, expected dtype/rank/device, the active context capsule, private bridge symbols, and correct `dladdr` provenance. Any mismatch raises a native exception.

## CUDA: build-only, non-promoting evidence

The only candidate is frozen to Linux `x86_64-unknown-linux-gnu`, CPython 3.11, Rust 1.93.1, TensorFlow 2.21.0, exactly one `GPU:0`, permitted SM values, and float32 rank-1/rank-2 tensors already resident on that GPU:

```text
tf.matmul → tf.nn.bias_add(NHWC) → tf.nn.relu → tf.reduce_mean(axis=1)
```

It requires `rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64` authorization. Hosted CI uses a synthetic probe and only generates/links the extension; it never imports TensorFlow, loads the extension, or executes CUDA. The retained WSL2/RTX 3060 (`sm_86`) manual evidence is self-attested and verifier-valid, but the offline verifier proves schema/payload integrity—not GPU execution or certification. It deliberately records:

```text
support_claim=false
certification_ready=false
kernel_activity_verified=false
runtime_transfer_profiled=false
```

There is no transfer lowering, CPU/CUDA mixing, `GPU:1`, multi-GPU, Variables, GradientTape, forward accumulator, async eager, `tf.function`, XLA, training, Windows/macOS CUDA, kernel-activity claim, transfer-profile claim, performance claim, or CUDA support claim. See the [complete frozen CUDA contract](docs/cuda-build-only-0.1.2.md).

## Further detail

- [CUDA build-only and manual-evidence contract](docs/cuda-build-only-0.1.2.md)
- [Platform truth matrix](ci/platform-contract.json)
- [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE)
