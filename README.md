# rextio-tensorflow

**Unreleased 0.1.2 native-AOT Alpha bounded CPU-surface expansion.** The latest released
package remains 0.1.0 (released **2026-07-18**).

This is a Rextio **plugin API 1.3** provider that lowers a **tiny, statically
proven** subset of Python **TensorFlow 2.21.0 `CPU:0` inference-oriented code** to native Rust
AOT code. Generated code does **not** reimplement TensorFlow in pure Rust. It
is an **owned thin safe wrapper** over the **same** already-loaded TensorFlow
wheel’s public TFE C API plus a **private** EagerTensor bridge
(`dlopen` / `dlsym` with `RTLD_NOLOAD`).

| Status field | Value |
| --- | --- |
| Version | `0.1.2` (`src/rextio_tensorflow/__about__.py`) |
| Maturity | Public Alpha PoC — limited, version-pinned native-AOT surface |
| Release state | **Unreleased — do not tag or publish**; latest release is [`rextio-tensorflow==0.1.0`](https://pypi.org/project/rextio-tensorflow/0.1.0/) |
| Performance claim | **None** — no benchmark gate; Alpha does not claim speedups |
| Pure-Rust TensorFlow | **No** — native helpers call into the active wheel |
| Abandoned TF Rust crates | **Not used** as Cargo dependencies (`crate_dependencies() == ()`) |

This hotfix remains a plugin API **1.3** provider and supports Core 0.1.5's
API 1.4 host-extension path. It rejects boundary-free standalone Rust lowering
and does not opt into standalone artifact capability. Provider entry methods
also fail closed unless the host advertises API 1.x minor 3 or newer, protecting
the API-1.3 field boundary even if package dependencies are bypassed.

Final release verification completed on 2026-07-18: GitHub Actions
[run `29597803215`](https://github.com/rextio/rextio-tensorflow/actions/runs/29597803215)
finished **13/13 jobs successfully**, and a no-cache CPython 3.11 install from
PyPI resolved `tensorflow==2.21.0` and exposed the plugin entry point with API
**1.3** metadata. This is release evidence, not a performance claim.

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
| Package version | `0.1.2` (unreleased) | `__about__.__version__` |
| CPython | **3.11 only** (`requires-python = ">=3.11,<3.12"`) | `pyproject.toml`; runtime rejects other implementations/versions |
| Platform profiles | See **Platform ABI profiles** below | Compile-time `PlatformAbiProfile` + runtime `validate_platform` |
| Rextio package | **`>=0.1.3,<0.2`** | Allowed package range in `pyproject.toml`, not an exact package pin |
| Plugin API | **1.3** (`REQUIRED_PLUGIN_API = "1.3"`) | `plugin.py`; loader contract tests |
| TensorFlow (Python) | **`tensorflow==2.21.0`** | `pyproject.toml` dependency; runtime checks `tf.__version__` |
| TensorFlow (C) | **`TF_Version() == "2.21.0"`** | Runtime `Api::load` |
| Device | **`CPU:0` only** | Boundary requires a backing-device name ending in `/device:CPU:0`; ops reuse that device |
| Dtype | **float32 operation inputs/intermediates; default-int64 ArgMax output** | Runtime checks float32 rank-1/2 inputs and exact int64 rank-1 classification output/boundaries |
| Ranks | **float32 rank 1/2; int64 rank 1 only** | Type vocabulary + claim/lower and boundary checks |
| Execution surface | **Inference-oriented only** | Training and `GradientTape` integration are unsupported. MatMul sets `grad_a` / `grad_b` false, but this is not a general TensorFlow no-grad guarantee. |
| Generated Rust crate | Edition **2021**, `rust-version = "1.83"`, PyO3 **0.29** | Inherited from Rextio 0.1.3's generated Cargo manifest; the Rust version is an MSRV, not an exact toolchain patch pin |
| Certified Rust toolchain | `rustc 1.93.1`, `cargo 1.93.1` on `aarch64-apple-darwin` | Used for the current real-Cargo Alpha evidence; this repo has no `rust-toolchain.toml` |
| Rust TF crates | **None** | `crate_dependencies() == ()`; helpers must not use `tensorflow-sys` / high-level `tensorflow` crate |

The TensorFlow and CPython pins are intentionally exact because this Alpha
crosses a private eager ABI. A successful install on another version is not a
support claim. Release metadata uses the standard
`Development Status :: 3 - Alpha` classifier; source availability and a dated
release do not broaden the certified runtime profiles below.

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
| Linux x86_64 | **Experimental**; merged PR #1 passed real Cargo + TF 2.21.0 candidate-wheel E2E | Hosted native E2E remains a release gate |
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
| ReLU | `tf.nn.relu` | **1 or 2** | **None** | preserves rank | `rextio-tensorflow/relu-f32-cpu-{1,2}d` | `RXTP-TENSORFLOW-018` / `002` |
| Sigmoid | `tf.nn.sigmoid` | **1 or 2** | **None** | preserves rank | `rextio-tensorflow/sigmoid-f32-cpu-{1,2}d` | `RXTP-TENSORFLOW-019` / `005` |
| Tanh | `tf.nn.tanh` | **1 or 2** | **None** | preserves rank | `rextio-tensorflow/tanh-f32-cpu-{1,2}d` | `RXTP-TENSORFLOW-020` / `009` |
| Math unary | `tf.abs`, `tf.negative`, `tf.square`, `tf.exp`, `tf.math.log`, `tf.math.sqrt` | **1 or 2** | exactly one positional operand; no keywords | preserves rank | `rextio-tensorflow/{abs,negative,square,exp,log,sqrt}-f32-cpu` | `RXTP-TENSORFLOW-026`–`031` |
| Add (call) | `tf.add` / `tf.math.add` | See add pairs below | **None** | max rank | `rextio-tensorflow/add-call-f32-cpu` | `RXTP-TENSORFLOW-003` |
| Add (binop) | binary `+` | See add pairs below | n/a | max rank | `rextio-tensorflow/add-binop-f32-cpu` | `RXTP-TENSORFLOW-006` |
| Multiply | `tf.multiply` / `tf.math.multiply` or binary `*` | See binary pairs below | calls take exactly two positional operands; no keywords | max rank | `rextio-tensorflow/mul-{call,binop}-f32-cpu` | `RXTP-TENSORFLOW-013` / `012` |
| Subtract | `tf.subtract` / `tf.math.subtract` or binary `-` | See binary pairs below | calls take exactly two positional operands; no keywords | max rank | `rextio-tensorflow/sub-{call,binop}-f32-cpu` | `RXTP-TENSORFLOW-014` / `015` |
| Divide | `tf.divide` / `tf.math.divide` or binary `/` | See binary pairs below | calls take exactly two positional operands; no keywords | max rank | `rextio-tensorflow/div-{call,binop}-f32-cpu` | `RXTP-TENSORFLOW-016` / `017` |
| Maximum / minimum | top-level `tf.maximum` / `tf.minimum` | **1 × 1 or 2 × 2**, equal ranks with TensorFlow-compatible same-rank broadcasting | exactly two positional non-literal tensors; no keywords | preserves rank | `rextio-tensorflow/{maximum,minimum}-call-f32-cpu` | `RXTP-TENSORFLOW-032` / `033` |
| Reduce mean | `tf.reduce_mean` / `tf.math.reduce_mean` | **2** | literal `axis=0\|1`, keyword or positional; named literal `keepdims=True\|False` or omitted | **1 or 2** | legacy axis-1 rule or `rextio-tensorflow/reduce-mean-literal-axis-f32-cpu-2d` | `RXTP-TENSORFLOW-004` / `022` |
| Reduce sum | `tf.reduce_sum` / `tf.math.reduce_sum` | **2** | literal `axis=0\|1`, keyword or positional; named literal `keepdims=True\|False` or omitted | **1 or 2** | legacy axis-1 rule or `rextio-tensorflow/reduce-sum-literal-axis-f32-cpu-2d` | `RXTP-TENSORFLOW-011` / `023` |
| Softmax | `tf.nn.softmax` | **1 or 2** | rank 1: omitted or literal `axis=0`; rank 2: explicit literal `axis=1`; literal axes may be keyword or positional | preserves float32 rank | `rextio-tensorflow/softmax-axis{0-f32-cpu-1d,1-f32-cpu-2d}` | `RXTP-TENSORFLOW-025` / `007` |
| ArgMax | `tf.argmax` | **2** float32 | explicit literal `axis=0\|1`, keyword or positional; default output type only | **1** int64 | `rextio-tensorflow/argmax-axis{0,1}-i64-cpu-2d` | `RXTP-TENSORFLOW-024` / `008` |
| Bias add | `tf.nn.bias_add` | rank-2 value + rank-1 bias | data format omitted or named literal `NHWC`; tensor operands positional | **2** float32 | `rextio-tensorflow/bias-add-nhwc-f32-cpu-2d` | `RXTP-TENSORFLOW-021` |

### Elementwise operand pairs (add, multiply, subtract, and divide)

| Left | Right | Result |
| --- | --- | --- |
| rank-2 | rank-2 | rank-2 |
| rank-1 | rank-1 | rank-1 |
| rank-2 | rank-1 | rank-2 (trailing bias broadcast) |
| rank-1 | rank-2 | rank-2 (either order) |

Claims prove **ranks only**. Concrete matrix / broadcast dimension
compatibility is checked by TFE (`MatMul`, `AddV2`, …) at runtime.
Maximum/minimum are narrower than the general binary matrix because mixed
ranks are excluded. Within equal ranks, the owned TFE operation preserves
TensorFlow's normal broadcasting and incompatible-shape behavior.

### Coverage declaration (analyzer routing)

Declared packages/modules/symbols (`rules/coverage.py`):

- packages: `tensorflow`
- modules: `tensorflow`, `tensorflow.linalg`, `tensorflow.nn`, `tensorflow.math`
- symbols: `tensorflow.matmul`, `tensorflow.linalg.matmul`, `tensorflow.nn.relu`,
  `tensorflow.nn.sigmoid`, `tensorflow.nn.tanh`, `tensorflow.abs`,
  `tensorflow.negative`, `tensorflow.square`, `tensorflow.exp`,
  `tensorflow.math.log`, `tensorflow.math.sqrt`, `tensorflow.add`,
  `tensorflow.math.add`, `tensorflow.multiply`, `tensorflow.math.multiply`,
  `tensorflow.subtract`, `tensorflow.math.subtract`, `tensorflow.divide`,
  `tensorflow.math.divide`, `tensorflow.maximum`, `tensorflow.minimum`,
  `tensorflow.reduce_mean`, `tensorflow.math.reduce_mean`, `tensorflow.reduce_sum`,
  `tensorflow.math.reduce_sum`, `tensorflow.nn.softmax`,
  `tensorflow.argmax`, `tensorflow.nn.bias_add`

### Boundary annotation types

Import-free markers (`rextio_tensorflow.types` — never import TensorFlow):

| Annotation | Plugin type key | Rust native type |
| --- | --- | --- |
| `TensorF32Cpu2D` | `rextio-tensorflow/tensor-f32-cpu-2d` | `rextio_tensorflow_runtime::RxtTfTensor` |
| `TensorF32Cpu1D` | `rextio-tensorflow/tensor-f32-cpu-1d` | `rextio_tensorflow_runtime::RxtTfTensor` |
| `TensorI64Cpu1D` | `rextio-tensorflow/tensor-i64-cpu-1d` | `rextio_tensorflow_runtime::RxtTfTensor` |

Runtime values remain ordinary `tf.Tensor` / EagerTensor objects. Intermediates
between helpers stay `TFE_TensorHandle`-native (`RxtTfTensor` RAII). Python
`for` / `if` that Rextio core can prove from scalar values remain ordinary core
Rust control flow. Tensor-data-dependent branches are not part of this plugin
surface.

`TensorI64Cpu1D` is the default `tf.argmax` result type and is also a real
input boundary type: native functions annotated with it accept only exact
CPU int64 rank-1 EagerTensors. They reject float inputs, other ranks,
`tf.Variable`, alternate runtimes, and non-CPU devices before executing.

### Canonical lowered helpers

Lowering emits calls into the exact generated module
`rextio_tensorflow_runtime` (single helper block; no Cargo TF crates):

| Op | Emitted Rust (shape) |
| --- | --- |
| matmul | `rextio_tensorflow_runtime::matmul(&a, &b)?` |
| relu | `rextio_tensorflow_runtime::relu(&x)?` |
| sigmoid | `rextio_tensorflow_runtime::sigmoid(&x)?` |
| tanh | `rextio_tensorflow_runtime::tanh(&x)?` |
| abs / negative / square / exp / log / sqrt | `rextio_tensorflow_runtime::{abs,negative,square,exp,log,sqrt}(&x)?` |
| add / `+` | `rextio_tensorflow_runtime::add(&a, &b)?` |
| bias add (NHWC) | `rextio_tensorflow_runtime::bias_add(&value, &bias)?` |
| multiply / `*` | `rextio_tensorflow_runtime::mul(&a, &b)?` |
| subtract / `-` | `rextio_tensorflow_runtime::sub(&a, &b)?` |
| divide / `/` | `rextio_tensorflow_runtime::div(&a, &b)?` |
| maximum / minimum | `rextio_tensorflow_runtime::{maximum,minimum}(&a, &b)?` |
| reduce_mean axis=0/1 | `reduce_mean_axis{0,1}[_keepdims](&x)?` |
| reduce_sum axis=0/1 | `reduce_sum_axis{0,1}[_keepdims](&x)?` |
| softmax final axis | rank 1: `softmax_axis0(&x)?`; rank 2: `softmax_axis1(&x)?` |
| argmax axis=0/1 (int64) | `rextio_tensorflow_runtime::argmax_axis{0,1}(&x)?` |
| boundary extract | `extract_f32_cpu_{1,2}d` / `extract_i64_cpu_1d` |
| boundary materialize | `materialize_tensor` (via `EagerTensorFromHandle`, ownership transfer) |

---

## Static preconditions (must hold at claim/lower time)

All of the following are required for a site to be **Claimed** and lowered:

1. **Annotations** — claimed TensorFlow operation operands use the plugin
   float32 CPU types above (not bare `tf.Tensor` or unannotated/`None` types).
   `TensorI64Cpu1D` is limited to the classification result and exact rank-1
   function input boundary; it is not accepted as a TensorFlow operation input.
2. **Functional style only** — covered calls with a **receiver** are
   `NotCovered` (no method-style receivers on matmul / unary operations / add /
   reduce_mean / reduce_sum). Lowering also rejects claimed/rendered receivers
   with `ValueError`.
3. **Positional operands only** for matmul / activations / math unary / add
   (keywords → `Rejected`).
4. **Matmul** — exactly two rank-2 tensors; no transpose keywords.
5. **Unary operations** — activations and the exact listed math-unary targets
   accept exactly one rank-1 or rank-2 tensor and no keywords. Alternate
   TensorFlow spellings such as `tf.math.abs` are not inferred aliases.
6. **Elementwise binary ops** — exactly two tensors in a supported rank pair
   (table above). Calls accept only the explicitly listed canonical targets
   and two positional operands; scalars and inferred aliases are excluded.
7. **reduce_mean / reduce_sum** — one rank-2 tensor plus static literal integer
   axis 0 or 1, either by keyword or as the second positional operand with
   exactly aligned `operand_literals`. `axis` metadata must have type `int`.
   `keepdims` is omitted or a **named** literal bool with type `bool`;
   positional keepdims, duplication, dynamic values, and extra keywords are
   rejected. The positional axis is validated at lower time but never emitted
   as a TFE input.
8. **Classification** — Softmax accepts rank-1 with its final axis omitted or
   literal axis 0, and rank-2 with explicit final axis 1. Raw TFE Softmax is
   last-axis-only and transpose is excluded. ArgMax accepts explicit literal
   axis 0 or 1 and retains default int64 output. Neither accepts `keepdims`;
   extra/output-type keywords are rejected.
9. **BiasAdd** — value is rank-2 float32 CPU, bias is rank-1 float32 CPU, both
   are positional and ordered value-then-bias. `data_format` is omitted or
   named literal `NHWC`; NCHW, dynamic/positional format, `name`, keyword
   tensor operands, and other rank orders are rejected.
10. **No dynamic axis/dtype/rank proof** — only the fixed Alpha vocabulary.
11. **Inference-oriented slice** — not training/`GradientTape`, graph/Session,
   `tf.function`/AutoGraph, or non-`CPU:0` execution.

Static claims do **not** prove concrete shapes (e.g. matmul inner dimensions).
Concrete incompatibilities fail later inside the owned TFE operation, including
Maximum/Minimum shapes that cannot broadcast.

---

## Unsupported / fallback forms

Anything outside the tables above is either:

| Outcome | Meaning | Typical cases |
| --- | --- | --- |
| **`NotCovered`** | Plugin declines; site may stay on ordinary Python fallback | Unknown symbols (`tf.cos`, …); method receivers on covered targets; untyped (`None`) operands |
| **`Rejected`** | Recognized shape but not lowerable; diagnostic + Python fallback | Wrong ranks; keywords on unary/binary calls; missing/dynamic/forged metadata; positional keepdims; non-plugin tensor types; unsupported BiasAdd format/forms |

### Explicit exclusions (not Alpha-supported)

- GPU, `CPU:1` or later, and every device other than `CPU:0`
- Training, `GradientTape`, optimizers, or a general TensorFlow no-grad contract
- `tf.Variable` or any non-exact EagerTensor at the Python boundary (E2E rejects it)
- Graph / Session, `tf.function`, AutoGraph, Keras, or SavedModel
- Tensor-data-dependent Python `if` / `for`, `tf.cond`, or `tf.while_loop`
- Non-float32 operation inputs; int64 is supported only for the exact rank-1
  ArgMax classification result and its annotated Python boundary
- Rank-1 matmul; rank-3+ / batched matmul
- Dynamic or non-0/1 reduction/classification axes; positional `keepdims`;
  rank-1 Softmax axis 1, rank-2 Softmax axis 0/default;
  `tf.argmax(output_type=...)`
- Matmul transpose / other keywords
- In-place ops; scalar operands; inferred aliases such as `tf.math.truediv`;
  `tf.math.maximum` / `tf.math.minimum`, raw-op forms, and mixed-rank
  broadcasting for `tf.maximum` / `tf.minimum`
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
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import tensorflow as tf

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorI64Cpu1D:
    h = tf.matmul(x, weight)           # rank-2 → rank-2
    h = tf.nn.relu(h)                  # rank-2 → rank-2
    h = tf.nn.sigmoid(h)               # optional; rank-2 → rank-2
    h = h + bias                       # or tf.add(h, bias); rank-2
    probabilities = tf.nn.softmax(h, axis=1)  # literal axis=1 → rank-2
    return tf.argmax(probabilities, axis=1)   # default int64 → rank-1
```

Also accepted (when types match the tables):

- `tf.linalg.matmul(a, b)` (alias of matmul rule)
- Explicit `tf.math.{add,multiply,subtract,divide}` aliases and `+ * - /`
  across the bounded rank matrix
- top-level `tf.maximum` / `tf.minimum` for two same-rank rank-1 or rank-2
  tensors with TensorFlow-compatible broadcast shapes
- rank-1 `tf.nn.relu` / `sigmoid` / `tanh`
- rank-1/rank-2 `tf.abs`, `tf.negative`, `tf.square`, `tf.exp`,
  `tf.math.log`, and `tf.math.sqrt`
- rank-1 `tf.nn.softmax(x)` and `tf.nn.softmax(x, axis=0)`
- `tf.nn.bias_add(matrix, bias)` and the explicit
  `data_format="NHWC"` form
- `tf.reduce_mean(x, 0, keepdims=True)` and
  `tf.reduce_sum(x, axis=1, keepdims=False)`
- `tf.argmax(x, 0)`; rank-2 Softmax remains explicit axis 1 only

Core-lowerable scalar Python control flow around claimed ops is supported. The
real-Cargo E2E uses `range(depth)` and an integer condition to choose relu or
sigmoid; tensor-dependent control flow remains unsupported.

### Rejected or not covered (stay on fallback or fail claim)

| Example | Outcome (claim layer) |
| --- | --- |
| `tf.matmul(rank1, rank2)` | `Rejected` (wrong ranks) |
| `tf.matmul(a, b, transpose_b=True)` | `Rejected` (keywords) |
| `tf.reduce_mean(x)` without an explicit axis | `Rejected` |
| `tf.reduce_mean(x, 0, True)` positional keepdims | `Rejected` |
| dynamic/non-aligned/forged axis metadata | `Rejected` |
| `tf.reduce_sum(x, axis=2)` or duplicate axis | `Rejected` |
| rank-2 `tf.nn.softmax(x)` / `tf.nn.softmax(x, axis=0)` | `Rejected` |
| rank-1 `tf.nn.softmax(x, axis=1)` | `Rejected` |
| `tf.argmax(x, axis=1, output_type=tf.int32)` | `Rejected` |
| `tf.nn.bias_add(x, bias, data_format="NCHW")` | `Rejected` |
| `tf.maximum(rank2, rank1)` / `tf.minimum(rank1, rank2)` | `Rejected` (mixed ranks / broadcasting excluded) |
| `tf.math.maximum(x, y)` / raw-op forms | `NotCovered` (only exact top-level targets are declared) |
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
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D, TensorI64Cpu1D
import tensorflow as tf

def inference(
    x: TensorF32Cpu2D,
    weight: TensorF32Cpu2D,
    bias: TensorF32Cpu1D,
) -> TensorI64Cpu1D:
    h = tf.matmul(x, weight)
    h = tf.nn.relu(h)
    h = h + bias
    return tf.argmax(tf.nn.softmax(h, axis=1), axis=1)
```

### Boundary and ABI contract (summary)

- Python boundary types: `TensorF32Cpu2D` / `TensorF32Cpu1D` / `TensorI64Cpu1D`
  (import-free markers); the classification head materializes exactly int64 rank-1 output.
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

## Install

```bash
# Published package (CPython 3.11 only; installs tensorflow==2.21.0 exactly):
python -m pip install --no-cache-dir "rextio-tensorflow==0.1.0"

# Source contributors:
python -m pip install -e ".[dev]"
```

The published package was verified in a fresh CPython 3.11 environment with
the exact TensorFlow 2.21.0 dependency and plugin API 1.3 entry-point metadata.
The exact CPython, TensorFlow, ABI, and platform requirements above still
apply.

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

Focused unit tests cover analyzer-resolved import aliases, claim accept/reject,
positional-literal alignment, lower emission into
`rextio_tensorflow_runtime`, plugin API 1.3 loader contract, empty crate deps,
runtime-helper hardening (`RTLD_NOLOAD`, private bridge symbols, no
`unwrap`/`panic!` in helpers), and **platform ABI profile source contracts**
(certified macOS arm64, experimental Linux x86_64/aarch64, unsupported/
Windows/32-bit fail-closed). The E2E uses the invoking CPython 3.11
environment, requires exact TensorFlow 2.21.0, and fails if the configured
interpreter or platform contract differs. Merged PR #1 produced hosted
candidate-wheel real-Cargo evidence on macOS ARM64 and Linux x86_64. The
declared certification class remains **macOS ARM64 only**; Linux stays
experimental pending a separate support-promotion decision. The original
vertical slice remains rank-2 matmul → rank-2 activations → scalar Rust
control flow → broadcast add → classification. The 0.1.2 follow-up adds a
real-Cargo slice spanning rank-1 relu/sigmoid/tanh → functional multiply →
rank-1 Softmax default/axis 0 → Abs/Neg/Square/Exp/Log/Sqrt → NHWC BiasAdd →
subtraction → reverse-broadcast RealDiv → axis-0/axis-1 keepdims reductions →
same-rank broadcast Maximum/Minimum → ArgMax axis 0, with CPU,
NaN/Inf/domain/signed-zero, shape-error,
no-host-resolve, provenance, and lifetime checks. The Linux
probe is opt-in and does not claim certification when it has not been run.

---

## Package metadata

| Field | Value |
| --- | --- |
| Name | `rextio-tensorflow` |
| Version | `0.1.2` (unreleased) |
| Entry point | `rextio.plugins` → `rextio_tensorflow.plugin:plugin` |
| Classifier | `Development Status :: 3 - Alpha` |
| Release date | Not yet released |
| Distribution state | **Unreleased — no 0.1.2 tag or PyPI publication**; [`rextio-tensorflow==0.1.0`](https://pypi.org/project/rextio-tensorflow/0.1.0/) remains live |
| License | MIT |

The isolated PEP 517 build backend is pinned exactly to `setuptools==82.0.1`
and `wheel==0.47.0`; CI package/test tools are likewise exact-pinned under
`ci/`. Transitive TensorFlow dependencies remain resolved by its exact 2.21.0
wheel metadata.

This is the unreleased 0.1.2 source on its integration branch. The tagged
public Alpha 0.1.0 remains the latest GitHub/PyPI release; its final CI run
`29597803215` completed 13/13 jobs successfully, followed by the verified
no-cache CPython 3.11 installation described above.

The long description attached to the already-uploaded PyPI 0.1.0 artifacts was
frozen from the pre-live candidate README and cannot be changed in place. It
may therefore retain release-pending wording; this GitHub README records the
verified post-release state. A future package version will carry the updated
long description.

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
