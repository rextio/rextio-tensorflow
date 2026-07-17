# Changelog

All notable changes to `rextio-tensorflow` are documented here following Keep a
Changelog and Semantic Versioning conventions.

## [0.1.0] — Alpha (unreleased / private)

Private local Alpha PoC. Package remains unpublished (`Private :: Do Not
Upload`). No performance benchmark gate.

### Added

- Private package scaffold for Rextio plugin API **1.3** with
  `Private :: Do Not Upload`.
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
  source contracts, one real-Cargo E2E against `/tmp/rextio-tensorflow-stage0/venv`
  (macOS arm64 certified), and an opt-in Linux experimental probe
  (`REXTIO_TF_LINUX_PROBE=1`).
- Docs: README, this CHANGELOG, `docs/implementation-plan-0.1.0.md`.

### Notes

- Does **not** depend on the abandoned high-level TensorFlow Rust crate;
  `tensorflow-sys` is reference-only, not a Cargo dependency.
- Runtime incompatibility raises a fail-closed native exception in Alpha;
  transparent call-time fallback requires a future core runtime-availability
  hook and is not claimed by this release.
- Linux experimental support is based on offline inspection of official
  `tensorflow==2.21.0` manylinux wheels; it is **not** a performance promise
  and is **not** certified real-Cargo E2E evidence.
