# TensorFlow CUDA E3 build-only candidate

This document describes an **unreleased, non-certifying** candidate. It is
`support_claim=false` and `certification_ready=false`. It is not part of
`rextio-tensorflow==0.1.0` and must not be represented as working CUDA support.

## Frozen environment

- Linux `x86_64-unknown-linux-gnu`
- CPython 3.11
- Rust 1.93.1
- TensorFlow 2.21.0 from the active Python wheel
- `rextio` API 1.6, pinned CI source commit
  `7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0`
- `rextio-device-cuda` commit
  `cf65733f06b91a801f9806367f09948ee7162540`
- exactly one TensorFlow `GPU:0`
- float32 rank-1/rank-2 exact EagerTensor values already resident on that GPU

## Accepted slice

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cuda0_1D, TensorF32Cuda0_2D


def inference(
    x: TensorF32Cuda0_2D,
    weight: TensorF32Cuda0_2D,
    bias: TensorF32Cuda0_1D,
) -> TensorF32Cuda0_1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

`bias_add` also accepts literal `data_format="NHWC"`. `reduce_mean` accepts
omitted or literal `keepdims=False`. No other CUDA spellings are admitted.

## Runtime boundary

Generated Rust uses the separate `rextio_tensorflow_cuda_runtime` module and
`RxtTfCudaTensor` type. It:

1. Reuses only already-loaded TensorFlow 2.21.0 wheel images with
   `RTLD_NOLOAD`.
2. Verifies every resolved symbol with `dladdr`.
3. Borrows and anchors the active synchronous Python eager `TFE_Context`.
4. Enumerates devices with `TFE_ContextListDevices` and the `TF_DeviceList`
   API, selects exactly one fully qualified type `GPU` ordinal 0 name, and
   requires exact backing-device equality on every tensor.
5. Rejects when a reverse-mode tape or forward accumulator may record the
   supplied input through the exact pinned
   `TFE_Py_TapeSetPossibleGradientTypes(PyObject*) -> PyObject*` ABI before
   copying its handle. An active recorder that is not watching these inputs
   does not change this slice's gradient semantics and is not rejected.
6. Sets every operation to the exact enumerated device and validates every
   intermediate/output dtype, rank, and backing device.

The official CPython 3.11 Linux x86_64 TensorFlow 2.21.0 wheel with SHA-256
`9056fbc9ba04235810b71ae6cbd958a196e8804fb53bbcffbf3e23b56155f124`
was inspected with `nm -D`. It defines
`_Z35TFE_Py_TapeSetPossibleGradientTypesP7_object` in
`python/lib_pywrap_tensorflow_common.so`; `_pywrap_tfe.so` references that
owner.

## Explicit exclusions

- Host↔device or device↔device transfer
- CPU/CUDA mixing, `GPU:1`, dynamic devices, or multi-GPU
- Variables, training, GradientTape, forward accumulators, or async eager mode
- `tf.function`, XLA, standalone Rust, Windows, macOS, or Linux non-GNU targets
- Provider-created CUDA contexts, streams, allocators, synchronization, or
  raw CUDA resources
- Performance claims, support claims, certification, release, and PyPI upload

The int32 axis handle for `reduce_mean(axis=1)` is the one bounded host control
input. It is not a user-tensor transfer.

## Hosted CI

Hosted CI uses the real Core/provider orchestration with a deterministic
synthetic probe. It generates and links one cdylib. It never installs or
imports TensorFlow in that job and never loads or executes the extension.

This build-only lane is deliberately not a substitute for a GPU test: it does
not load the extension, execute CUDA, establish numerical parity, or observe
the lifetime of borrowed TensorFlow objects.

## Opt-in manual real-NVIDIA first-stage evidence

`scripts/certify_cuda_candidate.py` is a manual, **first-stage evidence**
producer. It is not a hosted CI job and must be run only on a machine whose
operator has explicitly chosen to use a real NVIDIA GPU. Its output is checked
offline by `scripts/verify_cuda_e3_evidence.py`.

This is a frozen environment, not a portability recipe:

- Linux `x86_64-unknown-linux-gnu` with GNU/glibc; no macOS, Windows, musl, or
  cross-compiled host is accepted.
- CPython 3.11, TensorFlow `2.21.0`, and Rust `1.93.1`.
- Core checkout exactly `7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0` and CUDA
  provider checkout exactly `cf65733f06b91a801f9806367f09948ee7162540`.
- A clean TensorFlow-plugin checkout at candidate commit exactly
  `16e368a2e73be58d4cc51da1672a8a842e394fbd`; pass that value explicitly via
  `--expected-tensorflow-commit`.
- Exactly one usable `GPU:0`, with a permitted architecture from this closed
  set: `sm_60`, `sm_61`, `sm_70`, `sm_72`, `sm_75`, `sm_80`, `sm_86`,
  `sm_87`, `sm_89`, or `sm_90`. Other ordinals and SM values are rejected
  rather than generalized.

The harness deliberately has no `toolkit_root` setting or command-line option.
It reuses the active TensorFlow wheel and its already-loaded images; pointing
at an independent CUDA toolkit would violate the runtime-reuse contract.

Use independent checkout and output directories so neither evidence nor build
products can be confused with a source checkout:

```bash
export E3_ROOT="$HOME/rextio-tf-e3-manual-$(date +%Y%m%d-%H%M%S)"
export E3_OUT="$E3_ROOT/evidence-output"
export E3_BUILD="$E3_ROOT/isolated-build"
mkdir -p "$E3_ROOT/checkouts" "$E3_OUT" "$E3_BUILD"

git clone https://github.com/rextio/rextio.git "$E3_ROOT/checkouts/rextio"
git -C "$E3_ROOT/checkouts/rextio" checkout --detach \
  7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0
git clone https://github.com/rextio/rextio-device-cuda.git \
  "$E3_ROOT/checkouts/rextio-device-cuda"
git -C "$E3_ROOT/checkouts/rextio-device-cuda" checkout --detach \
  cf65733f06b91a801f9806367f09948ee7162540
git clone https://github.com/rextio/rextio-tensorflow.git \
  "$E3_ROOT/checkouts/rextio-tensorflow"
git -C "$E3_ROOT/checkouts/rextio-tensorflow" checkout --detach \
  16e368a2e73be58d4cc51da1672a8a842e394fbd

python3.11 -m venv "$E3_ROOT/venv"
"$E3_ROOT/venv/bin/python" -m pip install --upgrade pip
"$E3_ROOT/venv/bin/python" -m pip install tensorflow==2.21.0
"$E3_ROOT/venv/bin/python" -m pip install --no-deps \
  "$E3_ROOT/checkouts/rextio" "$E3_ROOT/checkouts/rextio-device-cuda" \
  "$E3_ROOT/checkouts/rextio-tensorflow"
rustup toolchain install 1.93.1 --profile minimal
```

Import TensorFlow before invoking the candidate. This is required to establish
the wheel-image reuse boundary, rather than an optional smoke test:

```bash
cd "$E3_ROOT/checkouts/rextio-tensorflow"
"$E3_ROOT/venv/bin/python" -c 'import tensorflow as tf; assert tf.__version__ == "2.21.0"'
"$E3_ROOT/venv/bin/python" scripts/certify_cuda_candidate.py \
  --output "$E3_OUT/cuda-e3-first-stage.json" \
  --work-dir "$E3_BUILD" \
  --core-root "$E3_ROOT/checkouts/rextio" \
  --provider-root "$E3_ROOT/checkouts/rextio-device-cuda" \
  --expected-tensorflow-commit 16e368a2e73be58d4cc51da1672a8a842e394fbd \
  --sm sm_80
"$E3_ROOT/venv/bin/python" scripts/verify_cuda_e3_evidence.py \
  "$E3_OUT/cuda-e3-first-stage.json"
```

The evidence records `native_extension_executed=true`, but intentionally
records `kernel_activity_verified=false` and `runtime_transfer_profiled=false`.
Accordingly it is execution, numerical-parity, and borrowed-object-lifetime
evidence only. It is **not** kernel-activity certification, a transfer/profile
measurement, CUDA support, or a performance claim. A successful harness and
verifier run leave `support_claim=false` and `certification_ready=false`.
