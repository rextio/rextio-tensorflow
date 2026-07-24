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
5. Rejects active reverse-mode tapes and forward accumulators through the
   exact pinned `TFE_Py_TapeSetPossibleGradientTypes(PyObject*) -> PyObject*`
   ABI before copying an input handle.
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

## Hosted CI and manual testing

Hosted CI uses the real Core/provider orchestration with a deterministic
synthetic probe. It generates and links one cdylib. It never installs or
imports TensorFlow in that job and never loads or executes the extension.

Real-NVIDIA execution, numerical parity, kernel activity, lifetime, same-image
runtime identity, and absence of user-tensor transfers remain deferred manual
work. Any future evidence remains non-certifying until a separate review
explicitly changes the contract.
