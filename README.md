# rextio-tensorflow

**Public native-AOT Alpha** (package version **0.1.0**, unreleased on PyPI).

This is a Rextio **plugin API 1.3** provider that lowers a **tiny, statically
proven** subset of Python **TensorFlow 2.21.0 `CPU:0` inference-oriented code** to native Rust
AOT code. Generated code does **not** reimplement TensorFlow in pure Rust. It
is an **owned thin safe wrapper** over the **same** already-loaded TensorFlow
wheel’s public TFE C API plus a **private** EagerTensor bridge
(`dlopen` / `dlsym` with `RTLD_NOLOAD`).

| Status field | Value |
| --- | --- |
| Version | `0.1.0` (`src/rextio_tensorflow/__about__.py`) |
| Maturity | Public Alpha PoC — source/CI review in progress; **unreleased on PyPI** |
| Upload gate | `Private :: Do Not Upload` remains until owner release approval |
| Performance claim | **None** — no benchmark gate; Alpha does not claim speedups |
| Pure-Rust TensorFlow | **No** — native helpers call into the active wheel |
| Abandoned TF Rust crates | **Not used** as Cargo dependencies (`crate_dependencies() == ()`) |

Unsupported call sites stay on Rextio’s ordinary **Python fallback** at
analysis time. Sites that *are* claimed and lowered native still **fail closed
at runtime** on version / symbol / boundary mismatch — core plugin API 1.3 has
**no** transparent runtime-availability / module-init retry hook.

---

## Version, platform, and ABI contract

Some entries are exact compatibility pins; others are package ranges, MSRV
contracts, or certification evidence. They are enforced by package metadata,
plugin registration, the generated runtime helper
(`rextio_tensorflow_runtime`), or the stated certification environment.

| Component | Contract | Enforcement / evidence |
| --- | --- | --- |
| Package version | `0.1.0` | `__about__.__version__` |
| CPython | **3.11 only** (`requires-python = ">=3.11,<3.12"`) | `pyproject.toml`; runtime rejects other implementations/versions |
| Platform profiles | See **Platform ABI profiles** below | Compile-time `PlatformAbiProfile` + runtime `validate_platform` |
| Rextio package | **`>=0.1.3,<0.2`** | Allowed package range in `pyproject.toml`, not an exact package pin |
| Plugin API | **1.3** (`REQUIRED_PLUGIN_API = "1.3"`) | `plugin.py`; loader contract tests |
| TensorFlow (Python) | **`tensorflow==2.21.0`** | `pyproject.toml` dependency; runtime checks `tf.__version__` |
| TensorFlow (C) | **`TF_Version() == "2.21.0"`** | Runtime `Api::load` |
| Device | **`CPU:0` only** | Boundary requires a backing-device name ending in `/device:CPU:0`; ops reuse that device |
| Dtype | **float32 only** | Plugin types `tensor-f32-cpu-{1,2}d`; runtime dtype checks |
| Ranks | **1 and 2 only** | Type vocabulary + claim/lower rules |
| Execution surface | **Inference-oriented only** | Training and `GradientTape` integration are unsupported. MatMul sets `grad_a` / `grad_b` false, but this is not a general TensorFlow no-grad guarantee. |
| Generated Rust crate | Edition **2021**, `rust-version = "1.83"`, PyO3 **0.29** | Inherited from Rextio 0.1.3's generated Cargo manifest; the Rust version is an MSRV, not an exact toolchain patch pin |
| Certified Rust toolchain | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` | Used for the current real-Cargo Alpha evidence; this repo has no `rust-toolchain.toml` |
| Rust TF crates | **None** | `crate_dependencies() == ()`; helpers must not use `tensorflow-sys` / high-level `tensorflow` crate |

The TensorFlow and CPython pins are intentionally exact because this Alpha
crosses a private eager ABI. A successful install on another version is not a
support claim. The temporary `Private :: Do Not Upload` classifier is an
upload safety gate, not a statement that the source repository is private; it
is removed only from a clean reviewed `main` immediately before an approved
PyPI release.

### Platform ABI profiles

The generated runtime selects an explicit **platform ABI profile** at compile
time (loader flags, wheel image basenames, Python `platform.machine()` tags).
Common load logic never assumes Darwin-only paths or Darwin `dlfcn` numeric
values mixed into Linux.

| Profile id | Class | Target triple (Rust) | Python machines | Wheel images (relative to `tensorflow` package root) | `dlopen` flags (numeric) |
| --- | --- | --- | --- | --- | --- |
| `macos-arm64` | **Certified** | `aarch64-apple-darwin` | `arm64` | `libtensorflow_cc.2.dylib`, `libtensorflow_framework.2.dylib`, `python/lib_pywrap_tensorflow_common.dylib` | Darwin: `RTLD_NOW=0x2`, `RTLD_LOCAL=0x4`, `RTLD_NOLOAD=0x10` |
| `linux-x86_64` | **Experimental** | `x86_64-unknown-linux-gnu` (`target_env=gnu` only) | `x86_64` | `libtensorflow_cc.so.2`, `libtensorflow_framework.so.2`, `python/lib_pywrap_tensorflow_common.so` | Linux glibc: `RTLD_NOW=0x2`, `RTLD_LOCAL=0`, `RTLD_NOLOAD=0x4` |
| `linux-aarch64` | **Experimental** | `aarch64-unknown-linux-gnu` (`target_env=gnu` only) | `aarch64` or `arm64` | same Linux basenames as x86_64 | same Linux glibc values |
| *(compile_error!)* | **Native-build fail-closed** | Windows, Linux **musl**, macOS x86_64/i686/ARMv7, Linux i686/ARMv7, and every other triple | n/a | n/a | n/a — clear `compile_error!` at native build (not a runtime dlfcn path) |

**Certified** means a real-Cargo E2E path has been run on that profile (macOS
arm64 only today). **Experimental** means official `tensorflow==2.21.0`
**manylinux / glibc** wheel archives were inspected offline for image layout
and exported private bridge symbols, and the runtime profile is wired for
Linux **GNU** only (`target_env=gnu`). This tree does **not** claim a certified
real-Cargo Linux E2E or any performance result. Linux **musl** is unsupported.

### Public CI truth matrix

The machine-readable source of truth is
[`ci/platform-contract.json`](ci/platform-contract.json). Every requested
Linux/macOS × x86/x64/ARM32/ARM64 cell is present, but only an upstream runtime
profile can run native E2E. Static platform-contract jobs test the remaining
cells as expected fail-closed outcomes; they are not emulated support claims.

| Requested cell | Public Alpha result | CI treatment |
| --- | --- | --- |
| Linux x86_64 | **Experimental**, release-gated on real Cargo + TF 2.21.0 | Hosted native E2E |
| Linux AArch64 | **Experimental / availability-gated** | Manual `ubuntu-24.04-arm` native E2E; no claim until green evidence exists |
| Linux i686 | **Unsupported** — no pinned upstream runtime | Static expected-unsupported / native-build fail-closed |
| Linux ARMv7 | **Unsupported** — no pinned upstream runtime | Static expected-unsupported / native-build fail-closed |
| macOS ARM64 | **Certified Alpha baseline** | Hosted native E2E plus existing local real-Cargo evidence |
| macOS x86_64 | **Availability-gated, currently unsupported** — no exact TF 2.21.0 wheel | Static expected-unsupported / native-build fail-closed |
| macOS i686 | **Impossible modern target / unsupported** | Static expected-unsupported / native-build fail-closed |
| macOS ARMv7 | **Impossible modern target / unsupported** | Static expected-unsupported / native-build fail-closed |

The main workflow separates `quality`, `platform-contract`, `native-e2e`, and
`package` jobs. Native E2E builds and installs the candidate wheel before
running the real route/lifetime suite; it does not certify an editable source
checkout. Actions have read-only repository permissions and every
third-party Action reference is pinned to an immutable commit. Linux AArch64
is a separate manual experimental workflow so runner availability cannot be
misrepresented as routine certification. A stable aggregate `ci-gate` job is
the intended branch-protection check.

On Linux GNU, the generated helper links **libdl** explicitly
(`#[link(name = "dl")]`) so cdylibs do not rely on incidental global
resolution of `dlopen`/`dlsym` on older manylinux/glibc.

Inspected wheels (download-only; not installed into the host environment):

- `tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl`
- `tensorflow-2.21.0-cp311-cp311-manylinux_2_27_x86_64.whl`
- `tensorflow-2.21.0-cp311-cp311-manylinux_2_27_aarch64.whl`

Windows support is **explicitly deferred**. Unsupported compile targets
(Windows, musl, other) fail closed at **native build** via `compile_error!`
because the POSIX `dlfcn` externs are not a truthful runtime contract there.
Runtime availability failures on supported profiles still never silently retry
the Python body under plugin API 1.3 (no runtime-availability hook).

### Why a private ABI exists

Public TFE C symbols alone are not enough to round-trip Python
`tf.Tensor` / EagerTensor objects at the function boundary without host
resolve. The Alpha runtime therefore also resolves **private** bridge
symbols from the active wheel’s pywrap image
(`python/lib_pywrap_tensorflow_common.{dylib,so}`; Itanium-mangled names).
Artifact-level `nm` of the three official 2.21.0 wheels above confirms the
**same** three exports on macOS arm64 and Linux x86_64/aarch64:

| Private symbol (mangled) | Role |
| --- | --- |
| `_Z18EagerTensor_HandlePK7_object` | `EagerTensor_Handle` — extract underlying `TFE_TensorHandle*` |
| `_Z21EagerTensorFromHandleP16TFE_TensorHandleb` | `EagerTensorFromHandle` — **takes ownership** of the handle (`is_packed=false`) |
| `_Z22EagerTensor_CheckExactPK7_object` | Exact EagerTensor type check |

These are **private ABI**: a TensorFlow patch within 2.21.x can break them.
That is an explicit residual risk of this PoC, not a public stability promise.
The bridge also depends on Python internals in
`tensorflow.python.eager.context`: `context()`, `ensure_initialized()`,
`is_async()`, and the private `context()._handle` **null-named capsule**.
Only the existing synchronous eager context is accepted. These Python-side
details are part of the same exact-2.21.0 private ABI pin even though they are
not C++ symbols.

---

## Supported TensorFlow forms and result ranks

Claim decisions are pure functions of Rextio API 1.3 site metadata (kind,
target, operand types, keyword **literals**). Lowering **revalidates** the
same constraints and fails with `ValueError` (not `assert`).

| Python form | Accepted targets | Operand ranks | Keywords | Result rank | Rule id | Diagnostic |
| --- | --- | --- | --- | --- | --- | --- |
| MatMul | `tf.matmul` / `tf.linalg.matmul` (also `tensorflow.*`) | **2 × 2** only | **None** (no `transpose_*`) | **2** | `rextio-tensorflow/matmul-f32-cpu-2d` | `RXTP-TENSORFLOW-001` |
| ReLU | `tf.nn.relu` | **2** | **None** | **2** | `rextio-tensorflow/relu-f32-cpu-2d` | `RXTP-TENSORFLOW-002` |
| Sigmoid | `tf.nn.sigmoid` | **2** | **None** | **2** | `rextio-tensorflow/sigmoid-f32-cpu-2d` | `RXTP-TENSORFLOW-005` |
| Add (call) | `tf.add` / `tf.math.add` | See add pairs below | **None** | max rank | `rextio-tensorflow/add-call-f32-cpu` | `RXTP-TENSORFLOW-003` |
| Add (binop) | binary `+` | See add pairs below | n/a | max rank | `rextio-tensorflow/add-binop-f32-cpu` | `RXTP-TENSORFLOW-006` |
| Reduce mean | `tf.reduce_mean` / `tf.math.reduce_mean` | **2** | **`axis=1` literal** only; optional `keepdims=False` or omitted | **1** | `rextio-tensorflow/reduce-mean-axis1-f32-cpu-2d` | `RXTP-TENSORFLOW-004` |

### Add operand pairs (call and binop)

| Left | Right | Result |
| --- | --- | --- |
| rank-2 | rank-2 | rank-2 |
| rank-1 | rank-1 | rank-1 |
| rank-2 | rank-1 | rank-2 (trailing bias broadcast) |
| rank-1 | rank-2 | rank-2 (either order) |

Claims prove **ranks only**. Concrete matrix / broadcast dimension
compatibility is checked by TFE (`MatMul`, `AddV2`, …) at runtime.

### Coverage declaration (analyzer routing)

Declared packages/modules/symbols (`rules/coverage.py`):

- packages: `tensorflow`
- modules: `tensorflow`, `tensorflow.linalg`, `tensorflow.nn`, `tensorflow.math`
- symbols: `tensorflow.matmul`, `tensorflow.linalg.matmul`, `tensorflow.nn.relu`,
  `tensorflow.nn.sigmoid`, `tensorflow.add`, `tensorflow.math.add`,
  `tensorflow.reduce_mean`, `tensorflow.math.reduce_mean`

### Boundary annotation types

Import-free markers (`rextio_tensorflow.types` — never import TensorFlow):

| Annotation | Plugin type key | Rust native type |
| --- | --- | --- |
| `TensorF32Cpu2D` | `rextio-tensorflow/tensor-f32-cpu-2d` | `rextio_tensorflow_runtime::RxtTfTensor` |
| `TensorF32Cpu1D` | `rextio-tensorflow/tensor-f32-cpu-1d` | `rextio_tensorflow_runtime::RxtTfTensor` |

Runtime values remain ordinary `tf.Tensor` / EagerTensor objects. Intermediates
between helpers stay `TFE_TensorHandle`-native (`RxtTfTensor` RAII). Python
`for` / `if` that Rextio core can prove from scalar values remain ordinary core
Rust control flow. Tensor-data-dependent branches are not part of this plugin
surface.

### Canonical lowered helpers

Lowering emits calls into the exact generated module
`rextio_tensorflow_runtime` (single helper block; no Cargo TF crates):

| Op | Emitted Rust (shape) |
| --- | --- |
| matmul | `rextio_tensorflow_runtime::matmul(&a, &b)?` |
| relu | `rextio_tensorflow_runtime::relu(&x)?` |
| sigmoid | `rextio_tensorflow_runtime::sigmoid(&x)?` |
| add / `+` | `rextio_tensorflow_runtime::add(&a, &b)?` |
| reduce_mean axis=1 | `rextio_tensorflow_runtime::reduce_mean_axis1(&x)?` |
| boundary extract | `extract_f32_cpu_{1,2}d` |
| boundary materialize | `materialize_tensor` (via `EagerTensorFromHandle`, ownership transfer) |

---

## Static preconditions (must hold at claim/lower time)

All of the following are required for a site to be **Claimed** and lowered:

1. **Annotations** — operands are the plugin float32 CPU types above (not bare
   `tf.Tensor`, not unannotated/`None` types).
2. **Functional style only** — covered calls with a **receiver** are
   `NotCovered` (no method-style receivers on matmul / relu / sigmoid / add /
   reduce_mean). Lowering also rejects claimed/rendered receivers with
   `ValueError`.
3. **Positional operands only** for matmul / relu / sigmoid / add (keywords →
   `Rejected`).
4. **Matmul** — exactly two rank-2 tensors; no transpose keywords.
5. **Activations** — exactly one rank-2 tensor; no keywords.
6. **Add** — exactly two tensors in a supported rank pair (table above).
7. **reduce_mean** — exactly one rank-2 tensor **plus** static literal keyword
   `axis=1`. Positional axis is **not** claimed on Alpha. Optional
   `keepdims=False` only (or omit). Non-literal keywords → `Rejected`.
8. **No dynamic axis/dtype/rank proof** — only the fixed Alpha vocabulary.
9. **Inference-oriented slice** — not training/`GradientTape`, graph/Session,
   `tf.function`/AutoGraph, or non-`CPU:0` execution.

Static claims do **not** prove concrete shapes (e.g. matmul inner dimensions).
Those fail later inside TFE if incompatible.

---

## Unsupported / fallback forms

Anything outside the tables above is either:

| Outcome | Meaning | Typical cases |
| --- | --- | --- |
| **`NotCovered`** | Plugin declines; site may stay on ordinary Python fallback | Unknown symbols (`tf.cos`, …); method receivers on covered targets; untyped (`None`) operands |
| **`Rejected`** | Recognized shape but not lowerable; diagnostic + Python fallback | Wrong ranks; keywords on matmul/relu/add; `reduce_mean` without `axis=1` literal; bad keepdims; non-plugin tensor types on covered ops (`RXTP-TENSORFLOW-010` / per-op codes) |

### Explicit exclusions (not Alpha-supported)

- GPU, `CPU:1` or later, and every device other than `CPU:0`
- Training, `GradientTape`, optimizers, or a general TensorFlow no-grad contract
- `tf.Variable` or any non-exact EagerTensor at the Python boundary (E2E rejects it)
- Graph / Session, `tf.function`, AutoGraph, Keras, or SavedModel
- Tensor-data-dependent Python `if` / `for`, `tf.cond`, or `tf.while_loop`
- Non-float32 dtypes; rank ≠ {1, 2}
- Rank-1 activations or matmul; rank-3+ / batched matmul
- Dynamic reduction axes; positional `axis`; `keepdims=True`
- Matmul transpose / other keywords
- In-place ops
- Host resolve (`TFE_TensorHandleResolve`) on the inference path
- DLPack
- `TFE_NewContext` / second eager context / Session
- Asynchronous eager contexts or mixing tensors from different eager contexts
- Loading alternate dylibs (`RTLD_DEFAULT` not used)
- Cargo dependency on abandoned high-level `tensorflow` crate or `tensorflow-sys`
- **Any performance claim** (no benchmark gate for Alpha)

Fallback rule record: `rextio-tensorflow/unsupported-tensor-surface`
(`RXTP-TENSORFLOW-010`, outcome `fallback`).

---

## Same-wheel runtime reuse (`RTLD_NOLOAD`)

The generated runtime binds **only** images already loaded by the active
`tensorflow==2.21.0` process — it does **not** load a second TensorFlow.
It never uses `RTLD_DEFAULT` or a process-global symbol search as a substitute
for per-image `dlsym` + `dladdr` provenance.

On first API load (`Api::load`):

1. Compile only for a supported `PlatformAbiProfile` (else `compile_error!`).
   At runtime require CPython 3.11, a matching `platform.machine()`, and Python
   `tf.__version__ == "2.21.0"`.
2. Canonicalize the three active-wheel library paths under the package root
   using the profile’s basenames (`.dylib` on certified macOS arm64; `.so.2`
   / `.so` on experimental Linux GNU).
3. Open each path only with
   **`RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD`** using **OS-specific numeric
   values** (Darwin `RTLD_NOLOAD=0x10` vs Linux glibc `RTLD_NOLOAD=0x4`;
   Linux `RTLD_LOCAL=0`). Missing image → error (never an instruction to load
   another copy).
4. Resolve each symbol from its **owning** image (`cc` / `framework` / `pywrap`)
   and verify provenance with **`dladdr`**.
5. Require `TF_Version() == "2.21.0"`, import
   `tensorflow.python.eager.context`, and reuse the existing synchronous
   Python eager context's null-named private capsule (`context()._handle`) —
   **no** `TFE_NewContext`.
6. Retain the three handles with the function table so pointers stay live.

The caller does not have to pre-import TensorFlow: lazy runtime initialization
calls `py.import("tensorflow")`. By the time `RTLD_NOLOAD` runs, however, the
three expected images from that exact active wheel must be mapped. If the
imported wheel does not map them, initialization fails closed rather than
loading a second TensorFlow runtime.

---

## Compile-time fallback vs runtime fail-closed

| Phase | Behavior | Transparent Python retry? |
| --- | --- | --- |
| **Analysis / claim** | `Claimed` → lower to native; `NotCovered` / `Rejected` → ordinary Rextio **Python fallback** for that site | Yes — unsupported sites never leave the fallback path |
| **Lowering** | Revalidates claim metadata; mismatch → `ValueError` (survives `python -O`) | N/A (compile/codegen failure) |
| **Native runtime** (version, symbols, `RTLD_NOLOAD`, dtype/rank/device/boundary) | Raise stable `rextio-tensorflow: …` exceptions (`PyRuntimeError` / `PyValueError` / type errors on extract) | **No** — plugin API **1.3** has **no** runtime-availability / module-init hook to transparently re-run the Python body |

Runtime error string prefixes used in contracts include (see
`diagnostics.RUNTIME_ERRORS` and E2E boundary checks):

- `rextio-tensorflow: expected a TensorFlow EagerTensor`
- `rextio-tensorflow: expected a CPU tensor`
- `rextio-tensorflow: expected a float32 tensor`
- rank mismatches on extract (message includes rank expectation)
- version / symbol / wheel-path mismatches under the same `rextio-tensorflow:` prefix

---

## Accepted and rejected examples

### Accepted (claim → native lower)

```python
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D
import tensorflow as tf

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorF32Cpu1D:
    h = tf.matmul(x, weight)           # rank-2 → rank-2
    h = tf.nn.relu(h)                  # rank-2 → rank-2
    h = tf.nn.sigmoid(h)               # optional; rank-2 → rank-2
    h = h + bias                       # or tf.add(h, bias); rank-2
    return tf.reduce_mean(h, axis=1)   # literal axis=1 → rank-1
```

Also accepted (when types match the tables):

- `tf.linalg.matmul(a, b)` (alias of matmul rule)
- `tf.math.add(x, y)` / `tf.math.reduce_mean(x, axis=1)`
- same-rank `+` / `tf.add` for 1D+1D or 2D+2D
- `tf.reduce_mean(x, axis=1, keepdims=False)`

Core-lowerable scalar Python control flow around claimed ops is supported. The
real-Cargo E2E uses `range(depth)` and an integer condition to choose relu or
sigmoid; tensor-dependent control flow remains unsupported.

### Rejected or not covered (stay on fallback or fail claim)

| Example | Outcome (claim layer) |
| --- | --- |
| `tf.matmul(rank1, rank2)` | `Rejected` (wrong ranks) |
| `tf.matmul(a, b, transpose_b=True)` | `Rejected` (keywords) |
| `tf.nn.relu(rank1)` | `Rejected` (Alpha relu is rank-2 only) |
| `tf.reduce_mean(x)` without `axis=1` | `Rejected` |
| `tf.reduce_mean(x, 1)` positional axis | `Rejected` (not statically proven on Alpha) |
| `tf.reduce_mean(x, axis=0)` | `Rejected` |
| `tf.cos(x)` | `NotCovered` |
| Method-style receiver on a covered call | `NotCovered` |
| Operand types outside plugin vocabulary on a covered symbol | `Rejected` (`RXTP-TENSORFLOW-010` / op diagnostic) |

### Runtime fail-closed (after successful native claim/build)

E2E boundary checks (real Cargo path) assert exceptions when native code is
invoked with annotation-violating values, for example:

| Runtime value at a `TensorF32Cpu2D` parameter | Observed failure |
| --- | --- |
| `float64` tensor | dtype message (`expected a float32 tensor`) |
| rank-1 float32 tensor | rank message |
| `tf.Variable(...)` | `expected a TensorFlow EagerTensor` |
| NumPy array | `expected a TensorFlow EagerTensor` |

These do **not** transparently fall back to the Python body under API 1.3.

---

## Alpha surface (reference sketch)

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
    h = h + bias
    return tf.reduce_mean(h, axis=1)
```

### Boundary and ABI contract (summary)

- Python boundary types: `TensorF32Cpu2D` / `TensorF32Cpu1D` (import-free markers).
- Native type: `rextio_tensorflow_runtime::RxtTfTensor` (owned handle RAII;
  clones share `Rc` owner — never an unowned pointer fallback).
- Extract: private `EagerTensor_Handle` then
  `TFE_TensorHandleCopySharingTensor` (no host resolve).
- Materialize: `EagerTensorFromHandle(..., is_packed=false)` **takes ownership**.
- Exact Python `tensorflow.__version__` and C `TF_Version()` must both be
  `2.21.0`; active-wheel images opened only with profile
  `RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD` (OS-specific numeric values).
- Reuses the existing Python eager context capsule. **No** `TFE_NewContext`,
  **no** Session, **no** DLPack, **no** `TFE_TensorHandleResolve` on the
  inference path.
- Borrowed context capsule and Python Context are held by strong Python
  references for every owned handle.

---

## Install (public Alpha source)

```bash
# Use CPython 3.11; tensorflow==2.21.0 is an exact package dependency.
python -m pip install -e ".[dev]"
```

The package is not yet on PyPI. Maintainer uploads remain blocked by
`Private :: Do Not Upload` until the release owner reviews CI, merges to
`main`, and explicitly removes the gate.

---

## Tests

```bash
# Unit / contract (no Cargo), including all platform truth cells:
pytest tests -m "not needs_cargo" -q

# Real-Cargo E2E (run under CPython 3.11 + TF 2.21.0):
pytest tests/e2e/test_alpha_real_cargo.py -q

# Opt-in Linux experimental probe (skipped unless env set + Linux host):
REXTIO_TF_LINUX_PROBE=1 pytest tests/e2e/test_linux_experimental_probe.py -q

# Lint / types (when the dev extra is installed):
ruff check src tests
mypy src
```

Focused unit tests cover claim accept/reject, lower emission into
`rextio_tensorflow_runtime`, plugin API 1.3 loader contract, empty crate deps,
runtime-helper hardening (`RTLD_NOLOAD`, private bridge symbols, no
`unwrap`/`panic!` in helpers), and **platform ABI profile source contracts**
(certified macOS arm64, experimental Linux x86_64/aarch64, unsupported/
Windows/32-bit fail-closed). The E2E uses the invoking CPython 3.11
environment, requires exact TensorFlow 2.21.0, and fails if the configured
interpreter or platform contract differs. Existing certification is
**macOS arm64 only** until the new hosted jobs produce reviewed evidence. Its
single vertical slice is: rank-2 matmul
→ rank-2 relu → scalar `for`/`if` selecting relu/sigmoid → rank-2 + rank-1
bias → axis-1 mean. Other aliases and supported add rank pairs are covered at
the unit claim/lower layer, not by separate real-Cargo fixtures. The Linux
probe is opt-in and does not claim certification when it has not been run.

---

## Package metadata

| Field | Value |
| --- | --- |
| Name | `rextio-tensorflow` |
| Version | `0.1.0` |
| Entry point | `rextio.plugins` → `rextio_tensorflow.plugin:plugin` |
| Upload gate | `Private :: Do Not Upload` until release approval |
| License | MIT |

The isolated PEP 517 build backend is pinned exactly to `setuptools==82.0.1`
and `wheel==0.47.0`; CI package/test tools are likewise exact-pinned under
`ci/`. Transitive TensorFlow dependencies remain resolved by its exact 2.21.0
wheel metadata.

This is the public Alpha source for `rextio/rextio-tensorflow`. It has no
release tag or PyPI publication yet; repository visibility and package
publishing remain separate owner-reviewed release gates.

For the intended Alpha architecture and staged scope, see the
[0.1.0 implementation plan](docs/implementation-plan-0.1.0.md). Release-facing
changes are recorded in [CHANGELOG.md](CHANGELOG.md); this README is the
current support contract.

---

## What this is not

1. **Not pure-Rust TensorFlow** — ops execute through the active wheel’s TFE C
   API; Rust owns handles and orchestration only.
2. **Not a whole-project TensorFlow translator** — only the tabulated Alpha
   slice is claimable.
3. **Not a performance product** — **no** speedup claim and **no** benchmark
   release gate for 0.1.0.
4. **Not a stable public ABI** — private EagerTensor bridge symbols and exact
   eager-context internals plus exact 2.21.0 / CPython 3.11 pins and the
   certified-vs-experimental platform profiles are intentional Alpha
   constraints. Windows remains deferred.
