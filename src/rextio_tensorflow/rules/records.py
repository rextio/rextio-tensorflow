"""Machine-readable rule records for rextio-tensorflow Alpha."""

from rextio.plugins.api import RuleRecord, RuleScope

RULE_RECORDS: tuple[RuleRecord, ...] = (
    RuleRecord(
        id="rextio-tensorflow/matmul-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.matmul / tf.linalg.matmul on float32 CPU rank-2 tensors "
                "(two positional operands)"
            ),
        ),
        constraint=(
            "Exactly two positional float32 CPU rank-2 tensor operands and no "
            "keywords. Static claims prove rank only; concrete matrix dimension "
            "compatibility is checked by TFE MatMul at runtime. Result is "
            "float32 CPU rank-2. Executed via TFE MatMul "
            "on the active tensorflow==2.21.0 wheel. Transpose keywords and "
            "other dtypes/devices/ranks stay outside the Alpha surface."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-001",
        guidance=(
            "Annotate both operands as rextio_tensorflow.types.TensorF32Cpu2D "
            "and call tf.matmul(a, b) or tf.linalg.matmul(a, b) with two "
            "positional CPU float32 rank-2 tensors."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/relu-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.relu on float32 CPU rank-2 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-2 tensor operand and no keywords. "
            "Result preserves the operand type."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-002",
        guidance=(
            "Call tf.nn.relu(x) with a TensorF32Cpu2D positional argument; "
            "do not use in-place or keyword forms."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/add-call-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.add / tf.math.add on float32 CPU rank-2 (+ optional rank-1 "
                "bias) or same-rank tensors"
            ),
        ),
        constraint=(
            "Two positional float32 CPU tensor operands for tf.add / "
            "tf.math.add: rank-2 + rank-2, rank-1 + rank-1, or rank-2 + rank-1 "
            "trailing bias (either order). Claims prove ranks only; concrete "
            "broadcast dimension compatibility is checked by AddV2 at runtime. "
            "Result rank is the broadcast maximum."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-003",
        guidance=(
            "Use tf.add(x, bias) with TensorF32Cpu2D / TensorF32Cpu1D annotations "
            "on the Alpha-supported broadcast shapes."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/add-binop-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "binary + on float32 CPU rank-2 (+ optional rank-1 bias) or "
                "same-rank tensors"
            ),
        ),
        constraint=(
            "Binop '+' over two float32 CPU plugin tensors: rank-2 + rank-2, "
            "rank-1 + rank-1, or rank-2 + rank-1 trailing bias (either order)."
            " Concrete broadcast dimensions are checked by AddV2 at runtime."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-006",
        guidance=(
            "Write x + bias with TensorF32Cpu2D / TensorF32Cpu1D annotations on "
            "the Alpha-supported broadcast shapes."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/reduce-mean-axis1-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.reduce_mean(x, axis=1) on float32 CPU rank-2 tensors "
                "(literal axis keyword)"
            ),
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal keyword axis=1; "
            "keepdims omitted or False. Positional axis is not claimed in "
            "Alpha. Result is float32 CPU "
            "rank-1."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-004",
        guidance=(
            "Write tf.reduce_mean(x, axis=1) with a TensorF32Cpu2D operand and "
            "a static axis=1 literal."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/sigmoid-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.sigmoid on float32 CPU rank-2 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-2 tensor operand and no keywords. "
            "Result preserves the operand type."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-005",
        guidance=(
            "Call tf.nn.sigmoid(x) with a TensorF32Cpu2D positional argument."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/softmax-axis1-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.softmax(x, axis=1) on a float32 CPU rank-2 tensor",
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal keyword axis=1 and no "
            "other keywords. Softmax runs on the rank-2 final axis and returns "
            "a float32 CPU rank-2 tensor. Dynamic or alternate axes remain "
            "outside the Alpha surface."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-007",
        guidance=(
            "Write tf.nn.softmax(x, axis=1) with a TensorF32Cpu2D operand and "
            "a static axis=1 literal."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/argmax-axis1-i64-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.argmax(x, axis=1) with default int64 output on float32 CPU rank-2",
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal keyword axis=1 and no "
            "other keywords. The owned TFE ArgMax wrapper receives a scalar "
            "int32 axis handle and materializes exactly an int64 CPU rank-1 "
            "EagerTensor. output_type overrides and dynamic axes are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-008",
        guidance=(
            "Write tf.argmax(x, axis=1) with a TensorF32Cpu2D operand and omit "
            "output_type to retain TensorFlow's default int64 result."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/unsupported-tensor-surface",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="type",
            pattern=(
                "tensor dtype/device/rank or call shape outside the Alpha "
                "float32 CPU rank-1/2 inference slice"
            ),
        ),
        constraint=(
            "Covered TensorFlow call sites whose operand types fall outside the "
            "registered float32 CPU rank-1/2 vocabulary, or whose shape is "
            "recognized but not lowerable, are rejected with guidance and stay "
            "on the Python fallback."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-TENSORFLOW-010",
        guidance=(
            "Keep the Alpha slice on float32 CPU rank-1/2 matmul, relu, add, "
            "reduce_mean(axis=1), sigmoid, softmax(axis=1), and default-int64 "
            "argmax(axis=1); other dtypes, devices, ranks, and dynamic literals "
            "remain on the fallback."
        ),
        stability="experimental",
        verified=False,
    ),
)


def tensorflow_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "tensorflow_rule_records"]
