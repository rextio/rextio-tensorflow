# Changelog

All notable changes to `rextio-tensorflow` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [Unreleased]

### Added

- Add `tf.nn.tanh(x)` for one positional float32 CPU rank-2 tensor with no
  keywords. The owned TFE `Tanh` operation reuses the existing same-wheel,
  RAII, eager-context, and float32 rank-2 result validation path.
- Add `tf.reduce_sum(x, axis=1[, keepdims=False])` for float32 CPU rank-2
  tensors. The owned TFE `Sum` path reuses the rank-1 axis handle, int32
  `Tidx`, RAII ownership, and rank-1 result validation of reduce-mean; dynamic,
  positional, duplicate, wrong, and extra metadata remains fail-closed.

- Narrow CPU inference classification-head lane: rank-2 float32
  `tf.nn.softmax(axis=1)` followed by `tf.argmax(axis=1)` with the default
  int64 rank-1 result. The generated runtime owns `Softmax`/`ArgMax` TFE ops,
  uses a scalar int32 axis handle, validates/materializes the int64 boundary,
  and preserves the existing RAII, provenance, exact-version, and fail-closed
  contracts.
- Claim/lower/fallback/lifetime coverage and real-Cargo numerical, shape, and
  dtype E2E proof for the classification head on the existing macOS arm64 and
  Linux x86_64 CI profiles.
- Exact CPU int64 rank-1 parameter extraction for `TensorI64Cpu1D`, including
  real-Cargo compile/runtime, rejection, materialization, and lifetime proof.

### Changed

- Prepared 0.1.1 compatibility with `rextio` 0.1.5 / plugin API 1.4 while
  retaining this provider's declared API **1.3** and its existing package,
  CPython, TensorFlow, and private-ABI pins.
- Replaced provider-side exact host API equality guards with a compatible
  minimum guard: host API 1.x minor 3 or newer is accepted, while older,
  major-mismatched, or malformed hosts fail closed if dependencies are
  bypassed. Core's loader remains the primary compatibility authority.
- Reject standalone Rust lowering explicitly. This provider remains a PyO3
  host-extension-only plugin and does not declare `artifact_capability()`.
- Added focused Core 0.1.3/0.1.4/0.1.5 CI compatibility coverage without
  changing the native macOS/Linux E2E gates.

## [0.1.0] — 2026-07-18

Public native-AOT Alpha release, tagged and live on PyPI as
[`rextio-tensorflow==0.1.0`](https://pypi.org/project/rextio-tensorflow/0.1.0/).
No performance benchmark gate or speedup claim.

### Added

- Public Alpha package for Rextio plugin API **1.3** with standard Alpha
  metadata and no private-upload classifier.
- Frozen baseline: CPython `>=3.11,<3.12`, `tensorflow==2.21.0`,
  `rextio>=0.1.3,<0.2`, CPU float32 rank-1/2, inference only.
- **Platform ABI profiles** in the generated runtime helper:
  - **Certified**: macOS arm64 (`macos-arm64`) — real-Cargo E2E path
  - **Experimental**: Linux **GNU/glibc** x86_64 and aarch64
    (`target_env=gnu` only; manylinux TF 2.21) — wheel artifact ABI verified;
    no certified Linux E2E claim; explicit `#[link(name = "dl")]` on Linux GNU
  - **Native-build fail-closed**: Windows (deferred), Linux musl, and every
    other triple via clear `compile_error!` (not a runtime dlfcn promise)
- Explicit per-profile loader flags, wheel image basenames, and
  `platform.machine()` checks (Darwin vs Linux glibc `RTLD_NOLOAD` / `RTLD_LOCAL`
  numeric values).
- Import-minimal plugin facade (`rextio_tensorflow.plugin`) and annotation
  vocabulary (`rextio_tensorflow.types`).
- Canonical generated Rust module `rextio_tensorflow_runtime`: dlopen/dlsym of
  the active TensorFlow wheel TFE C API (~20–30 symbols) plus private
  EagerTensor ABI bridge symbols `EagerTensor_Handle` /
  `EagerTensorFromHandle` with exact version and symbol validation.
- RAII owned status/op/tensor handles; borrowed-vs-owned handle semantics;
  reuse of the Python eager context; no Session, no duplicate Context, no
  DLPack, no host resolve on the inference path.
- Exact active-wheel `RTLD_NOLOAD` handles with per-image symbol resolution,
  `dladdr` provenance checks, strong Python context/capsule ownership, `Rc`
  handle sharing, and fail-closed synchronous-context validation.
- Alpha claim/lower surface: `tf.matmul` / `tf.linalg.matmul`, `tf.nn.relu`,
  `tf.add` / binop `+`, `tf.reduce_mean(..., axis=1)`, optional
  `tf.nn.sigmoid`.
- Focused plugin contract/claim/lower/import-minimal tests, platform-profile
  source contracts, one real-Cargo E2E (macOS arm64 certified), and an opt-in Linux experimental probe
  (`REXTIO_TF_LINUX_PROBE=1`).
- Machine-readable Linux/macOS × x86/x64/ARM32/ARM64 truth matrix. Missing
  pinned runtimes are tested as explicit native-build fail-closed outcomes,
  not reported as native support.
- Read-only, immutable-Action-pinned GitHub CI split into quality,
  platform-contract, real native-E2E, and package jobs, plus a manual
  availability-gated Linux AArch64 workflow.
- Exact PEP 517 backend pins and candidate-wheel installation before the
  real native route/lifetime E2E; editable-source execution is not accepted as
  public CI certification evidence.
- Portable real-Cargo E2E interpreter selection: the invoking CPython 3.11 +
  TensorFlow 2.21.0 environment replaces the historical fixed temporary path.
- Docs: README, this CHANGELOG, `docs/implementation-plan-0.1.0.md`.

### Notes

- Final GitHub Actions
  [run `29597803215`](https://github.com/rextio/rextio-tensorflow/actions/runs/29597803215)
  completed **13/13 jobs successfully**.
- A live no-cache CPython 3.11 install from PyPI resolved
  `tensorflow==2.21.0` and exposed plugin API **1.3** entry-point metadata.
- The already-uploaded 0.1.0 PyPI long description was frozen from the
  pre-live candidate README and cannot be changed in place; the current GitHub
  README records the verified release state.
- Does **not** depend on the abandoned high-level TensorFlow Rust crate;
  `tensorflow-sys` is reference-only, not a Cargo dependency.
- Runtime incompatibility raises a fail-closed native exception in Alpha;
  transparent call-time fallback requires a future core runtime-availability
  hook and is not claimed by this release.
- Linux experimental support is based on offline inspection of official
  `tensorflow==2.21.0` manylinux wheels; it is **not** a performance promise
  and is **not** a certified support claim. The merged release PR passed
  candidate-wheel real-Cargo E2E on Linux x86_64; Linux AArch64 remains
  artifact-verified/manual, and both Linux profiles remain experimental.
- The exact private eager bridge and eager-context ABI remains prominently
  disclosed; runtime mismatch, `RTLD_NOLOAD` failure, or per-image provenance
  failure remains fail-closed with no transparent Python replay.
