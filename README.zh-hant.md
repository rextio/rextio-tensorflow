# rextio-tensorflow

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 專案圖示"></p>
<p align="center"><strong>透過 Rextio 自有的薄 TFE C API runtime lowering 有界 TensorFlow 推論。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-tensorflow/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-tensorflow?label=PyPI" alt="PyPI 上的 rextio-tensorflow"></a> <a href="https://github.com/rextio/rextio-tensorflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 授權條款"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <a href="./README.zh-hans.md">简体中文</a> · <strong>繁體中文</strong> · <a href="./README.ja.md">日本語</a></p>

`rextio-tensorflow` 是 [Rextio](https://github.com/rextio/rextio) 的公開 native-AOT Alpha PoC。它把 TensorFlow 2.21.0 的狹窄推論表面 lowering 為自有 Rust helper，呼叫已載入 wheel 的 eager TFE C API；不是 pure Rust 重寫，也沒有 TensorFlow Rust crate 相依。

> [!IMPORTANT]
> 這是精確版本、inference-only Alpha。不支援的 site 留在 Python fallback；native site 遇到 ABI/symbol/version/dtype/rank/device/eager-context 不符時 fail closed。沒有透明 runtime 重試、training 或 speedup 聲明。

## 查看限定路徑

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

Rextio claim 精確 rank/type 路徑，intermediate 保持為自有 `TFE_TensorHandle`，邊界 materialize 一般 `tf.Tensor`。具體 matrix/broadcast 相容性由 TensorFlow runtime 驗證。

## 運作方式

```text
typed Python → Rextio API 1.6 claim/revalidation → owned helper → RTLD_NOLOAD/provenance checks → TF 2.21.0 synchronous eager context
```

- 重用 active wheel/context，不建立第二個 TensorFlow context。
- 公開 TFE C operation/status/tensor handle 由 RAII 管理。
- 函式邊界使用精確 2.21.0 wheel 的三個 private EagerTensor bridge symbol；private ABI 是主要 Alpha 風險。
- marker 描述 CPU/CUDA、dtype、rank；runtime 值仍是 exact EagerTensor。
- 拒絕 standalone Rust lowering，只整合 PyO3 host extension。

## 安裝與首次使用

```bash
python3.11 -m pip install 'rextio>=0.1.6,<0.2' 'rextio-tensorflow==0.1.3'
```

resolver 安裝精確 `tensorflow==2.21.0`。同一 CPython 3.11 環境必須完成 TensorFlow import、Rextio 與生成 extension 載入。

## 相容性契約

| 元件 | 契約 |
| --- | --- |
| 套件 | `rextio-tensorflow==0.1.3`（公開 Alpha，2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.6,<0.2` / plugin API `1.6` |
| TensorFlow | Python/C 均精確 `2.21.0`；`TF_Version() == "2.21.0"` |
| CPU | exact `CPU:0`；float32 rank 1/2，ArgMax/input boundary int64 rank 1 |
| 生成 runtime | 無 `tensorflow`/`tensorflow-sys` Cargo dependency |
| 認證 evidence | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 模式 | synchronous eager inference；無 training |

macOS arm64 為 **Certified Alpha**。Linux x86_64 GNU/glibc 有 hosted real-Cargo E2E，但仍是未認證 **Experimental**；Linux AArch64 為 **Experimental/availability-gated**。macOS x86_64 無 exact wheel；Windows、musl、i686、ARMv7 與其他 triple 在 native build 時 fail closed。

## 支援的 CPU 表面

| 家族 | 接受形式 |
| --- | --- |
| Matmul | `tf.matmul`/`tf.linalg.matmul`；rank 2×2；無 transpose keyword |
| Activation/unary | `tf.nn.relu/sigmoid/tanh`；`tf.abs/negative/square/exp`、`tf.math.log/sqrt`；rank 1/2，無額外 keyword |
| Elementwise | `tf.add/multiply/subtract/divide`、`tf.math.*`、`+ * - /`；1/1、2/2、2/1、1/2；無 scalar/options |
| Maximum/minimum | 頂層 `tf.maximum/minimum`；同 rank 1/1 或 2/2 |
| Mean/sum | `tf.reduce_mean/sum`/`tf.math.*`；rank 2；literal axis 0/1；`keepdims` 省略或 named literal bool |
| Softmax/ArgMax | softmax：rank 1 axis 省略/0 或 rank 2 axis 1。ArgMax：rank-2 float32、axis 0/1、default output type → int64 rank 1 |
| Transpose | `tf.transpose`；rank 2；僅預設 permutation，無 `perm`/`conjugate`/`name` |
| Bias add | `tf.nn.bias_add`；rank-2 + rank-1；data format 省略或 literal `NHWC` |

Claim 只證明 rank，具體維度由 TFE 檢查。Marker 為 `TensorF32Cpu2D`、`TensorF32Cpu1D`、`TensorI64Cpu1D`。Tensor-dependent control flow、Variable、async eager、其他 runtime 不在範圍。

## Fallback 與 fail-closed

不支援 call 在分析時留在 Python fallback，錯誤 rank/option 用 `RXTP-TENSORFLOW-*` 拒絕。Lowering 重驗 rule ID/rank/target/operand/literal，漂移時丟 `ValueError`。進入 native 後不會 Python replay；已 import 的 exact TF 2.21.0 wheel、synchronous eager、exact EagerTensor、dtype/rank/device、context capsule、private symbol 與 `dladdr` provenance 必須全部相符，否則 native exception。

## CUDA：build-only、non-promoting 證據

僅限 Linux `x86_64-unknown-linux-gnu`、CPython 3.11、Rust 1.93.1、TF 2.21.0、唯一 exact `GPU:0`、允許 SM、已駐留 GPU 的 float32 rank-1/2，以及：

```text
tf.matmul → tf.nn.bias_add(NHWC) → tf.nn.relu → tf.reduce_mean(axis=1)
```

需要 `rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64` authorization。Hosted CI 用 synthetic probe 只 link，不 import TensorFlow、load extension 或執行 CUDA。WSL2/RTX 3060 (`sm_86`) evidence 是 self-attested/verifier-valid，verifier 只證明 schema/payload integrity。欄位為 `support_claim=false`、`certification_ready=false`、`kernel_activity_verified=false`、`runtime_transfer_profiled=false`。不含 transfer、混合 device、GPU:1、multi-GPU、Variable/GradientTape/forward accumulator、async eager、`tf.function`、XLA、training、Windows/macOS CUDA 或效能/支援聲明。見[完整契約](docs/cuda-build-only-0.1.2.md)。

## 更多文件

- [CUDA 契約](docs/cuda-build-only-0.1.2.md)
- [平台 truth matrix](ci/platform-contract.json)
- [變更記錄](CHANGELOG.md)

## 授權

[MIT](LICENSE)
