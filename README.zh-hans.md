# rextio-tensorflow

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 项目图标"></p>
<p align="center"><strong>通过 Rextio 自有的薄 TFE C API runtime lowering 有界 TensorFlow 推理。</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-tensorflow/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-tensorflow?label=PyPI" alt="PyPI 上的 rextio-tensorflow"></a> <a href="https://github.com/rextio/rextio-tensorflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 许可证"></a></p>
<p align="center"><a href="./README.md">English</a> · <a href="./README.ko.md">한국어</a> · <strong>简体中文</strong> · <a href="./README.zh-hant.md">繁體中文</a> · <a href="./README.ja.md">日本語</a></p>

`rextio-tensorflow` 是 [Rextio](https://github.com/rextio/rextio) 的公开 native-AOT Alpha PoC。它把 TensorFlow 2.21.0 的狭窄推理表面 lowering 为自有 Rust helper，调用已加载 wheel 的 eager TFE C API；并非用 pure Rust 重写 TensorFlow，也没有 TensorFlow Rust crate 依赖。

> [!IMPORTANT]
> 这是精确版本、inference-only Alpha。不支持的 site 留在 Python fallback；native site 遇到 ABI/symbol/version/dtype/rank/device/eager-context 不匹配时 fail closed。没有透明 runtime 重试、training 或 speedup 声明。

## 查看限定路径

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

Rextio claim 精确 rank/type 路径，intermediate 保持为自有 `TFE_TensorHandle`，边界处 materialize 普通 `tf.Tensor`。具体 matrix/broadcast 兼容性由 TensorFlow runtime 验证。

## 工作原理

```text
typed Python → Rextio API 1.6 claim/revalidation → owned helper → RTLD_NOLOAD/provenance checks → TF 2.21.0 synchronous eager context
```

- 复用 active wheel/context，不创建第二个 TensorFlow context。
- 公共 TFE C operation/status/tensor handle 由 RAII 管理。
- 函数边界使用精确 2.21.0 wheel 的三个 private EagerTensor bridge symbol；private ABI 是主要 Alpha 风险。
- marker 描述 CPU/CUDA、dtype、rank；runtime 值仍是 exact EagerTensor。
- 拒绝 standalone Rust lowering，仅集成 PyO3 host extension。

## 安装与首次使用

```bash
python3.11 -m pip install 'rextio>=0.1.6,<0.2' 'rextio-tensorflow==0.1.3'
```

resolver 安装精确 `tensorflow==2.21.0`。同一 CPython 3.11 环境必须完成 TensorFlow import、Rextio 和生成 extension 的加载。

## 兼容性契约

| 组件 | 契约 |
| --- | --- |
| 包 | `rextio-tensorflow==0.1.3`（公开 Alpha，2026-07-27） |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.6,<0.2` / plugin API `1.6` |
| TensorFlow | Python/C 均精确 `2.21.0`；`TF_Version() == "2.21.0"` |
| CPU | exact `CPU:0`；float32 rank 1/2，ArgMax/input boundary int64 rank 1 |
| 生成 runtime | 无 `tensorflow`/`tensorflow-sys` Cargo dependency |
| 认证 evidence | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 模式 | synchronous eager inference；无 training |

macOS arm64 为 **Certified Alpha**。Linux x86_64 GNU/glibc 有 hosted real-Cargo E2E，但仍是未认证 **Experimental**；Linux AArch64 为 **Experimental/availability-gated**。macOS x86_64 无 exact wheel；Windows、musl、i686、ARMv7 和其他 triple 在 native build 时 fail closed。

## 支持的 CPU 表面

| 家族 | 接受形式 |
| --- | --- |
| Matmul | `tf.matmul`/`tf.linalg.matmul`；rank 2×2；无 transpose keyword |
| Activation/unary | `tf.nn.relu/sigmoid/tanh`；`tf.abs/negative/square/exp`、`tf.math.log/sqrt`；rank 1/2，无额外 keyword |
| Elementwise | `tf.add/multiply/subtract/divide`、`tf.math.*`、`+ * - /`；1/1、2/2、2/1、1/2；无 scalar/options |
| Maximum/minimum | 顶层 `tf.maximum/minimum`；同 rank 1/1 或 2/2 |
| Mean/sum | `tf.reduce_mean/sum`/`tf.math.*`；rank 2；literal axis 0/1；`keepdims` 省略或 named literal bool |
| Softmax/ArgMax | softmax：rank 1 axis 省略/0，或 rank 2 axis 1。ArgMax：rank-2 float32、axis 0/1、default output type → int64 rank 1 |
| Transpose | `tf.transpose`；rank 2；仅默认 permutation，无 `perm`/`conjugate`/`name` |
| Bias add | `tf.nn.bias_add`；rank-2 + rank-1；data format 省略或 literal `NHWC` |

Claim 只证明 rank，具体维度由 TFE 检查。Marker 为 `TensorF32Cpu2D`、`TensorF32Cpu1D`、`TensorI64Cpu1D`。Tensor-dependent control flow、Variable、async eager、其他 runtime 不在范围内。

## Fallback 与 fail-closed

不支持的 call 在分析时留在 Python fallback，错误 rank/option 用 `RXTP-TENSORFLOW-*` 拒绝。Lowering 重新验证 rule ID/rank/target/operand/literal，漂移时抛 `ValueError`。进入 native 后不会 Python replay；已 import 的 exact TF 2.21.0 wheel、synchronous eager、exact EagerTensor、dtype/rank/device、context capsule、private symbol 与 `dladdr` provenance 必须全部匹配，否则 native exception。

## CUDA：build-only、non-promoting 证据

仅限 Linux `x86_64-unknown-linux-gnu`、CPython 3.11、Rust 1.93.1、TF 2.21.0、唯一 exact `GPU:0`、允许 SM、已驻留 GPU 的 float32 rank-1/2，以及：

```text
tf.matmul → tf.nn.bias_add(NHWC) → tf.nn.relu → tf.reduce_mean(axis=1)
```

需要 `rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64` authorization。Hosted CI 用 synthetic probe 只 link，不 import TensorFlow、load extension 或执行 CUDA。WSL2/RTX 3060 (`sm_86`) evidence 是 self-attested/verifier-valid，verifier 仅证明 schema/payload integrity。其字段为 `support_claim=false`、`certification_ready=false`、`kernel_activity_verified=false`、`runtime_transfer_profiled=false`。不包含 transfer、混合 device、GPU:1、multi-GPU、Variable/GradientTape/forward accumulator、async eager、`tf.function`、XLA、training、Windows/macOS CUDA 或性能/支持声明。见[完整契约](docs/cuda-build-only-0.1.2.md)。

## 更多文档

- [CUDA 契约](docs/cuda-build-only-0.1.2.md)
- [平台 truth matrix](ci/platform-contract.json)
- [变更记录](CHANGELOG.md)

## 许可证

[MIT](LICENSE)
