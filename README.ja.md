# rextio-tensorflow

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio プロジェクトアイコン"></p>
<p align="center"><strong>Rextio 所有の薄い TFE C API runtime で限定 TensorFlow 推論を lowering します。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-tensorflow/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-tensorflow?label=PyPI" alt="PyPI の rextio-tensorflow"></a> <a href="https://github.com/rextio/rextio-tensorflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT ライセンス"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <a href="./README.zh-hans.md">简体中文</a> · <a href="./README.zh-hant.md">繁體中文</a> · <strong>日本語</strong></p>

`rextio-tensorflow` は [Rextio](https://github.com/rextio/rextio) の公開 native-AOT Alpha PoC です。TensorFlow 2.21.0 の狭い推論面を、load 済み wheel の eager TFE C API を呼ぶ所有 Rust helper に lowering します。TensorFlow の pure Rust 再実装ではなく、TensorFlow Rust crate 依存もありません。

> [!IMPORTANT]
> exact-version、inference-only Alpha です。対象外 site は Python fallback に残り、native site は ABI/symbol/version/dtype/rank/device/eager-context 不一致で fail closed します。透過 runtime retry、training、speedup 主張はありません。

## 限定経路を見る

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

Rextio は rank/type 経路を claim し、intermediate を所有 `TFE_TensorHandle` のまま保ち、境界で通常の `tf.Tensor` を materialize します。具体的な matrix/broadcast は TensorFlow runtime が検証します。

## 仕組み

```text
typed Python → Rextio API 1.6 claim/revalidation → owned helper → RTLD_NOLOAD/provenance checks → TF 2.21.0 synchronous eager context
```

- active wheel/context を再利用し、第二の TensorFlow context は作りません。
- 公開 TFE C operation/status/tensor handle を RAII で所有します。
- 関数境界は exact 2.21.0 wheel の private EagerTensor bridge symbol 3 個を使い、この private ABI が主な Alpha risk です。
- marker は CPU/CUDA、dtype、rank を記述し、runtime 値は exact EagerTensor です。
- standalone Rust lowering は拒否し、PyO3 host extension のみです。

## インストールと最初の利用

```bash
python3.11 -m pip install 'rextio>=0.1.6,<0.2' 'rextio-tensorflow==0.1.3'
```

resolver は exact `tensorflow==2.21.0` を導入します。同じ CPython 3.11 環境が TensorFlow import、Rextio、生成 extension load を行う必要があります。

## 互換性契約

| Component | 契約 |
| --- | --- |
| Package | `rextio-tensorflow==0.1.3`（公開 Alpha、2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.6,<0.2` / plugin API `1.6` |
| TensorFlow | Python/C とも exact `2.21.0`; `TF_Version() == "2.21.0"` |
| CPU | exact `CPU:0`; float32 rank 1/2、ArgMax/input boundary int64 rank 1 |
| 生成 runtime | `tensorflow`/`tensorflow-sys` Cargo dependency なし |
| 認証 evidence | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| Mode | synchronous eager inference; training なし |

macOS arm64 は **Certified Alpha**。Linux x86_64 GNU/glibc は hosted real-Cargo E2E がある未認証 **Experimental**、Linux AArch64 は **Experimental/availability-gated**。macOS x86_64 は exact wheel がなく、Windows、musl、i686、ARMv7、他 triple は native build で fail closed します。

## 対応 CPU 面

| 系統 | 対応形 |
| --- | --- |
| Matmul | `tf.matmul`/`tf.linalg.matmul`; rank 2×2; transpose keyword なし |
| Activation/unary | `tf.nn.relu/sigmoid/tanh`; `tf.abs/negative/square/exp`, `tf.math.log/sqrt`; rank 1/2、追加 keyword なし |
| Elementwise | `tf.add/multiply/subtract/divide`, `tf.math.*`, `+ * - /`; 1/1, 2/2, 2/1, 1/2; scalar/options なし |
| Maximum/minimum | top-level `tf.maximum/minimum`; 同 rank 1/1 または 2/2 |
| Mean/sum | `tf.reduce_mean/sum`/`tf.math.*`; rank 2; literal axis 0/1; `keepdims` 省略または named literal bool |
| Softmax/ArgMax | softmax: rank 1 axis 省略/0 または rank 2 axis 1。ArgMax: rank-2 float32、axis 0/1、default output type → int64 rank 1 |
| Transpose | `tf.transpose`; rank 2; default permutation のみ、`perm`/`conjugate`/`name` なし |
| Bias add | `tf.nn.bias_add`; rank-2 + rank-1; data format 省略または literal `NHWC` |

Claim は rank のみを証明し具体的次元は TFE が検査します。Marker は `TensorF32Cpu2D`, `TensorF32Cpu1D`, `TensorI64Cpu1D`。Tensor-dependent control flow、Variable、async eager、別 runtime は対象外です。

## Fallback と fail-closed

対象外 call は分析時に Python fallback、誤った rank/option は `RXTP-TENSORFLOW-*` で拒否します。Lowering は rule ID/rank/target/operand/literal を再検証し drift に `ValueError`。native に入ると Python replay はなく、import 済み exact TF 2.21.0 wheel、synchronous eager、exact EagerTensor、dtype/rank/device、context capsule、private symbol、`dladdr` provenance が全て必要です。

## CUDA：build-only、non-promoting evidence

Linux `x86_64-unknown-linux-gnu`、CPython 3.11、Rust 1.93.1、TF 2.21.0、唯一の exact `GPU:0`、許可 SM、GPU resident float32 rank-1/2 と次のみです。

```text
tf.matmul → tf.nn.bias_add(NHWC) → tf.nn.relu → tf.reduce_mean(axis=1)
```

`rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64` authorization が必要です。Hosted CI は synthetic probe で link するだけです。WSL2/RTX 3060 (`sm_86`) evidence は self-attested/verifier-valid ですが、verifier は schema/payload integrity のみを検証します。`support_claim=false`, `certification_ready=false`, `kernel_activity_verified=false`, `runtime_transfer_profiled=false` です。transfer、混在 device、GPU:1、multi-GPU、Variable/GradientTape/forward accumulator、async eager、`tf.function`、XLA、training、Windows/macOS CUDA、性能/support 主張はありません。[完全な契約](docs/cuda-build-only-0.1.2.md)を参照してください。

## 詳細

- [CUDA 契約](docs/cuda-build-only-0.1.2.md)
- [platform truth matrix](ci/platform-contract.json)
- [変更履歴](CHANGELOG.md)

## ライセンス

[MIT](LICENSE)
