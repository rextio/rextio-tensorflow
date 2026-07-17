# rextio-tensorflow 0.1.0 implementation plan

Status: private local Alpha PoC

## Product definition

`rextio-tensorflow` is a Rextio plugin that lowers a **statically proven**
subset of Python TensorFlow 2.21.0 CPU inference code to Rust expressions
backed by an owned thin safe wrapper over the **same** TensorFlow wheel’s TFE
C API (process-local `dlsym` after the wheel is loaded).

It is not a whole-project TensorFlow translator. Unsupported sites remain on
the ordinary Python fallback path.

## Compatibility baseline

- **Certified platform**: macOS arm64 (`aarch64-apple-darwin`) — real-Cargo E2E
- **Experimental platforms**: Linux **GNU/glibc** x86_64 and aarch64
  (`target_env=gnu` only; manylinux TF 2.21 wheels) — wheel artifact ABI
  verified; runtime profile wired; no certified real-Cargo E2E in this tree
- **Native-build fail-closed**: Windows (deferred), Linux **musl**, and every
  other triple — clear `compile_error!` (POSIX dlfcn is not a truthful runtime
  contract on those targets)
- Python: CPython 3.11 (`requires-python = ">=3.11,<3.12"`)
- Rextio: `>=0.1.3,<0.2` (plugin API **1.3**)
- TensorFlow: `tensorflow==2.21.0` (CPU)
- Rust binding: **no** high-level `tensorflow` crate; no `tensorflow-sys`
  Cargo dependency (headers/symbols used as reference only)
- Device: CPU only; dtype: float32; ranks: 1 and 2; inference only

## Platform ABI profiles

Loader constants, wheel image basenames, bridge image selection, and platform
validation live in an explicit `PlatformAbiProfile` selected at compile time.
Common logic never assumes Darwin-only paths or Darwin `dlfcn` numeric values.

| Profile | Class | Images | `RTLD_NOLOAD` |
| --- | --- | --- | --- |
| `macos-arm64` | certified | `libtensorflow_cc.2.dylib`, `libtensorflow_framework.2.dylib`, `python/lib_pywrap_tensorflow_common.dylib` | Darwin `0x10` |
| `linux-x86_64` (`target_env=gnu`) | experimental | `libtensorflow_cc.so.2`, `libtensorflow_framework.so.2`, `python/lib_pywrap_tensorflow_common.so` | Linux glibc `0x4` |
| `linux-aarch64` (`target_env=gnu`) | experimental | same Linux basenames | Linux glibc `0x4` |
| Windows / musl / other | **compile_error!** | n/a | n/a |

Linux GNU builds also `#[link(name = "dl")]` the dlfcn extern block so older
manylinux/glibc toolchains do not rely on incidental global `libdl` resolution.

Wheel artifact evidence (official `tensorflow==2.21.0` cp311 wheels, inspected
in a temporary directory without host install):

- `tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl`
- `tensorflow-2.21.0-cp311-cp311-manylinux_2_27_x86_64.whl`
- `tensorflow-2.21.0-cp311-cp311-manylinux_2_27_aarch64.whl`

Private bridge symbols (identical Itanium mangling on all three):

- `_Z18EagerTensor_HandlePK7_object`
- `_Z21EagerTensorFromHandleP16TFE_TensorHandleb`
- `_Z22EagerTensor_CheckExactPK7_object`

## Runtime architecture

### Canonical module

All generated support lives in one exact-text helper block that defines:

```text
mod rextio_tensorflow_runtime { ... }
```

Lowered expressions call:

- `rextio_tensorflow_runtime::matmul`
- `rextio_tensorflow_runtime::relu`
- `rextio_tensorflow_runtime::add`
- `rextio_tensorflow_runtime::reduce_mean_axis1`
- `rextio_tensorflow_runtime::sigmoid` (optional)

Boundary conversion uses:

- `rextio_tensorflow_runtime::extract_f32_cpu_{1,2}d`
- `rextio_tensorflow_runtime::materialize_tensor`

### Bound C / Eager surface (illustrative)

TF status/version/tensor: `TF_Version`, `TF_NewStatus`, `TF_DeleteStatus`,
`TF_GetCode`, `TF_Message`, `TF_AllocateTensor`, `TF_TensorData`,
`TF_TensorByteSize`, `TF_DeleteTensor`, …

TFE: `TFE_NewOp`, `TFE_DeleteOp`, `TFE_OpAddInput`, `TFE_OpSetAttrType`,
`TFE_OpSetAttrBool`, `TFE_OpSetDevice`, `TFE_Execute`,
`TFE_DeleteTensorHandle`, `TFE_TensorHandleCopySharingTensor`,
`TFE_TensorHandleNumDims`, `TFE_TensorHandleDim`,
`TFE_TensorHandleDataType`, `TFE_TensorHandleBackingDeviceName`,
`TFE_NewTensorHandle`, …

Private Python bridge (exact mangled names on verified 2.21.0 wheels):

- `_Z18EagerTensor_HandlePK7_object`
- `_Z21EagerTensorFromHandleP16TFE_TensorHandleb`
- `_Z22EagerTensor_CheckExactPK7_object`

`EagerTensorFromHandle` **takes ownership** of the handle; second argument is
`is_packed` (always `false` for this slice).

### Context policy

Reuse the already-initialized Python eager context
(`tensorflow.python.eager.context.context()._handle` capsule). Do **not** call
`TFE_NewContext` / create a Session.

### Fail-closed version / symbol policy

On first API load:

1. Compile only for certified macOS arm64 or experimental Linux GNU
   (`target_env=gnu`) x86_64/aarch64. Windows, musl, and other triples hit
   `compile_error!` at native build (not a runtime dlfcn success path). At
   runtime, re-check CPython 3.11 and `platform.machine()` for the profile.
2. Canonicalize the three exact active-wheel library paths from the profile and
   open them only with `RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD` using **OS-specific
   numeric values** (never the process-default symbol namespace).
3. Resolve symbols from the owning cc/framework/pywrap handle and verify every
   address with `dladdr`; retain all three handles with the function table.
4. Require `TF_Version() == "2.21.0"` and an existing synchronous unnamed
   Python eager-context capsule.
5. On mismatch or missing symbol, raise a stable fail-closed
   `rextio-tensorflow:` exception. Core API 1.3 cannot yet turn a native-call
   runtime error into a transparent Python fallback; it needs a future
   runtime-availability/module-init hook for that behavior.

## Alpha claim surface

| Python form | Rule id | Result |
| --- | --- | --- |
| `tf.matmul` / `tf.linalg.matmul` | `…/matmul-f32-cpu-2d` | rank-2 |
| `tf.nn.relu` | `…/relu-f32-cpu-2d` | rank-2 |
| `tf.add` / `+` | `…/add-f32-cpu` | max rank |
| `tf.reduce_mean(x, axis=1)` (literal keyword only) | `…/reduce-mean-axis1-f32-cpu-2d` | rank-1 |
| `tf.nn.sigmoid` | `…/sigmoid-f32-cpu-2d` | rank-2 |

Claim decisions are pure functions of API 1.3 site metadata. Lowering
revalidates independently with `ValueError` (not `assert`).

## Explicit exclusions

- Session, Graph, SavedModel, Keras training loops
- GPU / non-CPU devices
- Non-float32 dtypes; rank ≠ {1,2}
- Dynamic reduction axes / transpose keywords
- Host resolve (`TFE_TensorHandleResolve`) on the inference path
- DLPack
- Performance benchmark as a release gate
- Windows platform implementation (deferred; fail-closed only)

## Acceptance

- Focused unit tests: import-minimal, plugin contract, claim, lower, platform
  ABI profiles / unsupported targets
- One real-Cargo E2E against `/tmp/rextio-tensorflow-stage0/venv` (macOS arm64
  certified)
- Opt-in Linux experimental probe (`REXTIO_TF_LINUX_PROBE=1`) when accurate
- Ruff, mypy, package build/check as available
- No remotes/tags/PyPI; no `AGENTS.md`; no commit required for the Alpha drop

## Residual ABI risks

1. **Private bridge stability** — `EagerTensor_*` symbols and EagerTensor
   object layout are private ABI; a patch release of TF may break them even at
   2.21.x. Linux experimental profiles share the same mangled names as macOS
   (artifact-verified) but lack certified real-Cargo E2E in this tree.
2. **`TFE_TensorHandleCopySharingTensor` identity** — some builds return the
   same pointer with a refcount bump; ownership accounting must treat that as
   valid owned share, not a distinct allocation.
3. **Context capsule** — depends on `context._handle` remaining a null-named
   `PyCapsule` of `TFE_Context*`.
4. **Process-local symbols** — native code requires TensorFlow to be imported
   (and its exact images loaded) before `RTLD_NOLOAD` can succeed.
5. **GIL requirement** — `EagerTensorFromHandle` called through
   `ctypes.CDLL` (which releases the GIL) reproduced a SIGSEGV, while a
   GIL-preserving call succeeded. Both private bridge functions stay under a
   PyO3 `Python` token and never run in detached code.
6. **Mean axis tensor construction** — uses `TF_AllocateTensor` /
   `TF_TensorData` + `TFE_NewTensorHandle`, then immediately deletes the
   caller-owned `TF_Tensor` without resolving the primary float activation
   tensors.
7. **OS-specific `dlopen` flags** — Darwin and Linux glibc assign different
   numeric values to `RTLD_NOLOAD` / `RTLD_LOCAL`; using Darwin literals on
   Linux would mis-open images. Profiles encode the correct values.
8. **Experimental Linux GNU only** — manylinux/glibc wheels; musl unsupported
   at compile time; explicit `libdl` link on Linux GNU; no performance claim;
   live probe is opt-in only and is not certified real-Cargo E2E.
9. **Unsupported triples** — fail at native build (`compile_error!`); do not
   promise a stable runtime exception via POSIX dlfcn on Windows.
