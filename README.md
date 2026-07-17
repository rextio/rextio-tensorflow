# rextio-tensorflow

**Private Alpha PoC** Rextio plugin that lowers a tiny, statically proven
subset of Python TensorFlow **2.21.0** CPU inference to native Rust AOT code
via an owned thin safe wrapper over the **same** TensorFlow wheel TFE C API
(`dlopen` / `dlsym` of already-loaded dylibs).

Status: **0.1.0 Alpha** — not for PyPI. Package metadata includes
`Private :: Do Not Upload`.

## Compatibility baseline (frozen)

| Component | Pin |
| --- | --- |
| Platform | **macOS arm64** (Alpha PoC) |
| CPython | 3.11 only (`requires-python = ">=3.11,<3.12"`) |
| Rextio | `>=0.1.3,<0.2` (plugin API **1.3**) |
| TensorFlow | `tensorflow==2.21.0` |
| Device / dtype | **CPU**, **float32**, ranks **1** and **2** |
| Mode | **Inference only** |
| Rust TF crates | **None** — do not depend on the abandoned high-level `tensorflow` crate; `tensorflow-sys` is reference-only |

## Alpha surface

```python
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D
import tensorflow as tf

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    h = tf.matmul(x, weight)
    h = tf.nn.relu(h)
    h = h + bias          # or tf.add(h, bias)
    return tf.reduce_mean(h, axis=1)
```

Optional fifth op: `tf.nn.sigmoid` on float32 CPU rank-2.

Lowering emits calls into the canonical generated module
`rextio_tensorflow_runtime` (for example
`rextio_tensorflow_runtime::matmul(&a, &b)?`). Intermediates stay
`TFE_TensorHandle`-native between helpers. Python `for` / `if` remain core
Rust control flow.

### Boundary and ABI contract

- Python boundary types: `rextio_tensorflow.types.TensorF32Cpu2D` /
  `TensorF32Cpu1D` (import-free markers).
- Native type: `rextio_tensorflow_runtime::RxtTfTensor` (owned handle RAII).
- Extract uses private bridge `EagerTensor_Handle(const PyObject*)` then
  `TFE_TensorHandleCopySharingTensor` (no host resolve).
- Materialize uses `EagerTensorFromHandle(TFE_TensorHandle*, bool is_packed=false)`
  which **takes ownership** of the handle.
- Exact Python `tensorflow.__version__` and C `TF_Version()` must both be
  `2.21.0`. The runtime canonicalizes the active wheel paths, opens only those
  already-loaded images with `RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD`, resolves
  each symbol from its owning image, and verifies its `dladdr` provenance.
- A runtime/version/symbol mismatch raises a stable fail-closed
  `rextio-tensorflow:` exception. It does **not** transparently retry the
  Python body: core plugin API 1.3 has no runtime-availability/module-init
  fallback hook. Analysis-time unsupported sites still use ordinary fallback.
- Reuses the existing Python eager context (`context._handle` capsule). **No**
  `TFE_NewContext`, **no** Session, **no** DLPack, **no**
  `TFE_TensorHandleResolve` on the inference path.
- The borrowed context capsule and Python Context are held by strong Python
  references for every owned handle. Tensor wrapper clones share an `Rc`
  owner; they never fall back to an unowned pointer.

### Explicit exclusions

GPU/other devices, training/autograd, graph/Session, dynamic axis/dtype/rank,
in-place ops, the abandoned high-level TensorFlow Rust crate as a dependency,
and performance claims (no benchmark gate for Alpha).

## Install (development only)

```bash
# CPython 3.11 + tensorflow==2.21.0, e.g. the stage0 venv:
#   /tmp/rextio-tensorflow-stage0/venv
python -m pip install -e ".[dev]"
```

## Tests

```bash
# Unit / contract (no Cargo):
pytest tests/test_import_minimal.py tests/test_plugin.py tests/test_claim.py tests/test_lower.py -q

# Real-Cargo E2E (serialized; uses stage0 TF 2.21.0 venv):
pytest tests/e2e/test_alpha_real_cargo.py -q
```

## Package metadata

- Version: `0.1.0`
- Classifier: `Private :: Do Not Upload`
- No GitHub remote / tag / PyPI publish for this Alpha PoC folder unless the
  owner later decides otherwise.
