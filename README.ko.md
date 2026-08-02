# rextio-tensorflow

<p align="center"><img src="./assets/readme/rextio-icon.png" width="112" alt="Rextio 프로젝트 아이콘"></p>
<p align="center"><strong>Rextio가 소유한 얇은 TFE C API runtime을 통해 제한된 TensorFlow 추론을 lowering합니다.</strong></p>
<p align="center"><a href="https://pypi.org/project/rextio-tensorflow/0.1.3/"><img src="https://img.shields.io/pypi/v/rextio-tensorflow?label=PyPI" alt="PyPI의 rextio-tensorflow"></a> <a href="https://github.com/rextio/rextio-tensorflow/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT 라이선스"></a></p>
<p align="center"><a href="./README.md">English</a> · <strong>한국어</strong> · <a href="./README.zh-hans.md">简体中文</a> · <a href="./README.zh-hant.md">繁體中文</a> · <a href="./README.ja.md">日本語</a></p>

`rextio-tensorflow`는 [Rextio](https://github.com/rextio/rextio)의 공개 native-AOT Alpha PoC입니다. TensorFlow 2.21.0의 좁은 추론 표면을, 이미 로드된 wheel의 eager TFE C API를 호출하는 소유 Rust helper로 내립니다. TensorFlow를 pure Rust로 다시 구현하지 않으며 TensorFlow Rust crate dependency도 없습니다.

> [!IMPORTANT]
> 정확한 버전에 고정된 추론 전용 Alpha입니다. 지원하지 않는 site는 Python fallback에 남고, native site는 ABI/symbol/version/dtype/rank/device/eager-context 불일치 시 fail closed합니다. 투명한 runtime 재시도, training, speedup 주장은 없습니다.

## 제한된 경로 확인

```python
import tensorflow as tf
from rextio_tensorflow.types import TensorF32Cpu1D, TensorF32Cpu2D

def inference(x: TensorF32Cpu2D, weight: TensorF32Cpu2D, bias: TensorF32Cpu1D) -> TensorF32Cpu1D:
    hidden = tf.matmul(x, weight)
    biased = tf.nn.bias_add(hidden, bias)
    activated = tf.nn.relu(biased)
    return tf.reduce_mean(activated, axis=1)
```

Rextio는 rank/type 경로를 claim하고 intermediate를 소유 `TFE_TensorHandle`로 유지하며 경계에서 일반 `tf.Tensor`를 만듭니다. 구체적인 matrix/broadcast 호환성은 TensorFlow runtime이 검증합니다.

## 동작 방식

```text
typed Python → Rextio API 1.6 claim/revalidation → owned runtime helper → RTLD_NOLOAD/provenance checks → TF 2.21.0 synchronous eager context
```

- active wheel과 context를 재사용하며 두 번째 TensorFlow context를 만들지 않습니다.
- 공개 TFE C 연산/status/tensor handle은 RAII로 소유합니다.
- 함수 경계는 정확한 2.21.0 wheel의 private EagerTensor bridge symbol 3개를 사용합니다. 이 private ABI가 주요 Alpha 위험입니다.
- marker는 CPU/CUDA, dtype, rank를 나타내고 runtime 값은 exact EagerTensor입니다.
- standalone Rust lowering은 거부하며 PyO3 host-extension에서만 동작합니다.

## 설치와 첫 사용

```bash
python3.11 -m pip install 'rextio>=0.1.6,<0.2' 'rextio-tensorflow==0.1.3'
```

resolver는 정확한 `tensorflow==2.21.0`을 설치합니다. 같은 CPython 3.11 환경이 TensorFlow import, Rextio, 생성 extension load를 모두 담당해야 합니다.

## 호환성 계약

| 구성 요소 | 계약 |
| --- | --- |
| 패키지 | `rextio-tensorflow==0.1.3` (공개 Alpha, 2026-07-27) |
| CPython | `>=3.11,<3.12` |
| Rextio / API | `>=0.1.6,<0.2` / plugin API `1.6` |
| TensorFlow | Python/C 모두 정확히 `2.21.0`; `TF_Version() == "2.21.0"` |
| CPU | exact `CPU:0`; float32 rank 1/2, ArgMax/input boundary int64 rank 1 |
| 생성 runtime | `tensorflow`/`tensorflow-sys` Cargo dependency 없음 |
| 인증 evidence | `rustc 1.93.1`, `cargo 1.93.1`, `aarch64-apple-darwin` |
| 모드 | synchronous eager inference; training 없음 |

macOS arm64는 **Certified Alpha**입니다. Linux x86_64 GNU/glibc는 hosted real-Cargo E2E가 있는 **Experimental**이고 비인증입니다. Linux AArch64 GNU/glibc는 **Experimental/availability-gated**입니다. macOS x86_64는 exact wheel 부재로 unsupported이며 Windows, musl, i686, ARMv7, 기타 triple은 native build에서 fail closed합니다.

## 지원되는 CPU 표면

| 계열 | 허용 형식 |
| --- | --- |
| Matmul | `tf.matmul`/`tf.linalg.matmul`; rank 2×2; transpose keyword 없음 |
| Activation/unary | `tf.nn.relu/sigmoid/tanh`; `tf.abs/negative/square/exp`, `tf.math.log/sqrt`; rank 1/2, 추가 keyword 없음 |
| Elementwise | `tf.add/multiply/subtract/divide`, `tf.math.*`, `+ * - /`; 1/1, 2/2, 2/1, 1/2; scalar/option 없음 |
| Maximum/minimum | top-level `tf.maximum/minimum`; 같은 rank 1/1 또는 2/2 |
| Mean/sum | `tf.reduce_mean/sum`/`tf.math.*`; rank 2; literal axis 0/1; `keepdims` 생략 또는 named literal bool |
| Softmax/ArgMax | softmax: rank 1 axis 생략/0 또는 rank 2 axis 1. ArgMax: rank-2 float32, axis 0/1, default output type → int64 rank 1 |
| Transpose | `tf.transpose`; rank 2; default permutation만, `perm`/`conjugate`/`name` 없음 |
| Bias add | `tf.nn.bias_add`; rank-2 + rank-1; data format 생략 또는 literal `NHWC` |

Claim은 rank만 증명하며 구체적인 차원은 TFE가 검사합니다. Marker는 `TensorF32Cpu2D`, `TensorF32Cpu1D`, `TensorI64Cpu1D`입니다. Tensor-dependent control flow, Variable, async eager, 다른 runtime은 제외됩니다.

## Fallback과 fail-closed

지원하지 않는 call은 분석 시 Python fallback에 남고 잘못된 rank/option은 `RXTP-TENSORFLOW-*`로 거부됩니다. Lowering은 rule ID/rank/target/operand/literal을 다시 검증하고 drift 시 `ValueError`입니다. Native 실행 뒤에는 Python replay가 없습니다. 이미 import된 exact TF 2.21.0 wheel, synchronous eager, exact EagerTensor, dtype/rank/device, context capsule, private symbol과 `dladdr` provenance가 모두 맞아야 하며 불일치는 native exception입니다.

## CUDA: build-only, non-promoting 증거

Linux `x86_64-unknown-linux-gnu`, CPython 3.11, Rust 1.93.1, TF 2.21.0, GPU 하나의 exact `GPU:0`, 허용 SM, 이미 GPU에 있는 float32 rank-1/2와 다음 경로만 허용합니다.

```text
tf.matmul → tf.nn.bias_add(NHWC) → tf.nn.relu → tf.reduce_mean(axis=1)
```

`rextio-device-cuda/cuda-tensorflow-tfe-linux-x86_64` authorization이 필요합니다. Hosted CI는 synthetic probe로 link만 하며 TensorFlow import/extension load/CUDA 실행을 하지 않습니다. WSL2/RTX 3060 (`sm_86`) evidence는 self-attested/verifier-valid이지만 verifier는 schema/payload integrity만 검증합니다. `support_claim=false`, `certification_ready=false`, `kernel_activity_verified=false`, `runtime_transfer_profiled=false`입니다. Transfer, 혼합 device, GPU:1, multi-GPU, Variable/GradientTape/forward accumulator, async eager, `tf.function`, XLA, training, Windows/macOS CUDA, 성능/지원 주장은 없습니다. [전체 계약](docs/cuda-build-only-0.1.2.md)을 읽으세요.

## 추가 문서

- [CUDA 계약](docs/cuda-build-only-0.1.2.md)
- [플랫폼 truth matrix](ci/platform-contract.json)
- [변경 기록](CHANGELOG.md)

## 라이선스

[MIT](LICENSE)
