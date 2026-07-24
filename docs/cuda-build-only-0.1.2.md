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
- A clean TensorFlow-plugin checkout selected by `TF_REF`. After this PR is
  integrated, set `TF_REF=0.1.2`; that is the default runnable path. Until
  then, the current `0.1.2` target branch does not contain these two scripts,
  so a reviewer/operator must set `TF_REF` to the current immutable full
  PR-head SHA instead. Do not use a moving feature-branch name or record that
  self-referential SHA in this document. After checkout, derive the full
  lowercase SHA with `git rev-parse HEAD` and pass it explicitly via
  `--expected-tensorflow-commit`; the harness verifies that it descends from
  the frozen E3 base.
- Exactly one usable `GPU:0`, with a permitted architecture from this closed
  set: `sm_60`, `sm_61`, `sm_70`, `sm_72`, `sm_75`, `sm_80`, `sm_86`,
  `sm_87`, `sm_89`, or `sm_90`. Other ordinals and SM values are rejected
  rather than generalized.
- GNU binutils, including `readelf`, on `PATH`. The harness records GNU build
  IDs from the TensorFlow wheel images and fails closed when an expected image
  has no build ID.

The harness deliberately has no `toolkit_root` setting or command-line option.
It reuses the active TensorFlow wheel and its already-loaded images; pointing
at an independent CUDA toolkit would violate the runtime-reuse contract.

Use independent checkout, output, and work directories. The output file and
the new exclusive work directory must be outside **all three** clean source
checkouts; the harness rejects paths inside any attested checkout. Do not
create the work directory itself: the harness requires it not to exist yet.

```bash
export E3_ROOT="$HOME/rextio-tf-e3-manual-$(date +%Y%m%d-%H%M%S)"
export E3_OUT="$E3_ROOT/evidence-output"
export E3_BUILD="$E3_ROOT/isolated-build"
mkdir -p "$E3_ROOT/checkouts" "$E3_OUT"

git clone https://github.com/rextio/rextio.git "$E3_ROOT/checkouts/rextio"
git -C "$E3_ROOT/checkouts/rextio" checkout --detach \
  7f47f0ce8cea0b6dbeb7fd3c733f65eeaa6bb5e0
git clone https://github.com/rextio/rextio-device-cuda.git \
  "$E3_ROOT/checkouts/rextio-device-cuda"
git -C "$E3_ROOT/checkouts/rextio-device-cuda" checkout --detach \
  cf65733f06b91a801f9806367f09948ee7162540
export TF_ROOT="$E3_ROOT/checkouts/rextio-tensorflow"
export TF_REF=0.1.2
# Before this PR merges, replace 0.1.2 above with the current immutable full
# PR-head SHA. The 0.1.2 default becomes runnable only after integration.
git clone https://github.com/rextio/rextio-tensorflow.git "$TF_ROOT"
git -C "$TF_ROOT" fetch --tags origin "$TF_REF"
git -C "$TF_ROOT" checkout --detach "$TF_REF"
test -f "$TF_ROOT/scripts/certify_cuda_candidate.py"
test -f "$TF_ROOT/scripts/verify_cuda_e3_evidence.py"
export TF_COMMIT="$(git -C "$TF_ROOT" rev-parse HEAD)"
test "${#TF_COMMIT}" -eq 40

python3.11 -m venv "$E3_ROOT/venv"
"$E3_ROOT/venv/bin/python" -m pip install --upgrade pip
"$E3_ROOT/venv/bin/python" -m pip install tensorflow==2.21.0
"$E3_ROOT/venv/bin/python" -m pip install --no-deps \
  "$E3_ROOT/checkouts/rextio" "$E3_ROOT/checkouts/rextio-device-cuda" \
  "$E3_ROOT/checkouts/rextio-tensorflow"
rustup toolchain install 1.93.1 --profile minimal
command -v readelf
readelf --version
```

Import TensorFlow **in the same process that invokes the harness**. This is
required to establish the wheel-image reuse boundary, rather than an optional
smoke test. Set `TF_SM` to the actual permitted architecture of the sole usable
`GPU:0` (the example uses `sm_80` only as a placeholder):

```bash
cd "$TF_ROOT"
export TF_SM=sm_80
E3_OUTPUT="$E3_OUT/cuda-e3-first-stage.json" E3_WORK="$E3_BUILD" \
E3_CORE="$E3_ROOT/checkouts/rextio" \
E3_PROVIDER="$E3_ROOT/checkouts/rextio-device-cuda" \
"$E3_ROOT/venv/bin/python" - <<'PY'
import os
import sys

import tensorflow as tf

assert tf.__version__ == "2.21.0"
from scripts import certify_cuda_candidate

sys.argv = [
    "certify_cuda_candidate.py",
    "--output", os.environ["E3_OUTPUT"],
    "--work-dir", os.environ["E3_WORK"],
    "--core-root", os.environ["E3_CORE"],
    "--provider-root", os.environ["E3_PROVIDER"],
    "--expected-tensorflow-commit", os.environ["TF_COMMIT"],
    "--sm", os.environ["TF_SM"],
]
raise SystemExit(certify_cuda_candidate.main())
PY
"$E3_ROOT/venv/bin/python" scripts/verify_cuda_e3_evidence.py \
  "$E3_OUT/cuda-e3-first-stage.json"
```

The producer self-attests `native_extension_executed=true` only if the bounded
harness reaches that observation. It intentionally records
`kernel_activity_verified=false` and `runtime_transfer_profiled=false`.
The offline verifier checks canonical schema and payload integrity only; it
does not authenticate the producer, prove execution, recompute artifact
hashes, certify hardware, or confer CUDA support. Any evidence remains
self-attested execution, numerical-parity, and borrowed-object-lifetime
evidence, never a GPU-success claim, kernel-activity certification,
transfer/profile measurement, CUDA support, or a performance claim. It always
leaves `support_claim=false` and `certification_ready=false`.
