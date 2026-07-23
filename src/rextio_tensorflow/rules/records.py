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
        id="rextio-tensorflow/relu-f32-cpu-1d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.relu on float32 CPU rank-1 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-1 tensor operand and no keywords. "
            "The owned same-wheel TFE Relu result preserves float32, CPU:0, and rank 1."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-018",
        guidance="Call tf.nn.relu(x) with a TensorF32Cpu1D positional argument.",
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
                "binary + on float32 CPU rank-2 (+ optional rank-1 bias) or same-rank tensors"
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
        id="rextio-tensorflow/mul-binop-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "binary * on float32 CPU rank-2 (+ optional rank-1 bias) or same-rank tensors"
            ),
        ),
        constraint=(
            "Binop '*' over two float32 CPU plugin tensors only: rank-2 * rank-2, "
            "rank-1 * rank-1, or rank-2 * rank-1 trailing broadcast (either order). "
            "Concrete broadcast dimensions are checked by TFE Mul at runtime. "
            "Scalar operands are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-012",
        guidance=(
            "Write x * y with TensorF32Cpu2D / TensorF32Cpu1D annotations on "
            "the Alpha-supported broadcast shapes."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/mul-call-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.multiply / tf.math.multiply on float32 CPU rank-1/rank-2 "
                "same-rank or trailing rank-2/rank-1 broadcast tensors"
            ),
        ),
        constraint=(
            "Exactly two positional plugin tensor operands and no keywords. "
            "The accepted rank matrix matches binary '*'; concrete broadcasting "
            "is checked by the same owned TFE Mul operation."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-013",
        guidance="Call tf.multiply(x, y) with two supported annotated tensors.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/sub-call-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.subtract / tf.math.subtract on float32 CPU rank-1/rank-2 "
                "same-rank or trailing rank-2/rank-1 broadcast tensors"
            ),
        ),
        constraint=(
            "Exactly two positional plugin tensor operands and no keywords. "
            "Concrete broadcasting and incompatible-shape errors are delegated "
            "to the owned same-wheel TFE Sub operation."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-014",
        guidance="Call tf.subtract(x, y) with two supported annotated tensors.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/sub-binop-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "binary - on float32 CPU rank-1/rank-2 same-rank or trailing "
                "rank-2/rank-1 broadcast tensors"
            ),
        ),
        constraint=(
            "Exactly two plugin tensor operands in the bounded rank matrix. "
            "TFE Sub preserves operand order and owns shape-error semantics."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-015",
        guidance="Write x - y with two supported annotated tensors.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/div-call-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.divide / tf.math.divide on float32 CPU rank-1/rank-2 "
                "same-rank or trailing rank-2/rank-1 broadcast tensors"
            ),
        ),
        constraint=(
            "Exactly two positional plugin tensor operands and no keywords. "
            "The owned same-wheel TFE RealDiv operation supplies TensorFlow's "
            "floating division, special-value, broadcasting, and error semantics."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-016",
        guidance="Call tf.divide(x, y) with two supported annotated tensors.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/div-binop-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="binop",
            pattern=(
                "binary / on float32 CPU rank-1/rank-2 same-rank or trailing "
                "rank-2/rank-1 broadcast tensors"
            ),
        ),
        constraint=(
            "Exactly two plugin tensor operands in the bounded rank matrix. "
            "TFE RealDiv preserves operand order and TensorFlow float32 semantics."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-017",
        guidance="Write x / y with two supported annotated tensors.",
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
                "(literal keyword or positional axis)"
            ),
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal axis=1, passed either "
            "positionally with aligned operand literal metadata or by keyword; "
            "named keepdims is omitted or False. Result is float32 CPU rank-1."
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
        id="rextio-tensorflow/reduce-mean-literal-axis-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.reduce_mean on float32 CPU rank-2 tensors with literal "
                "axis=0|1 and named literal keepdims=True|False"
            ),
        ),
        constraint=(
            "Axis is a statically proven int literal 0 or 1, passed by keyword "
            "or as one aligned positional literal. keepdims is omitted or a "
            "named bool literal; positional keepdims is excluded. TFE Mean "
            "receives only the tensor plus an owned int32 axis handle. The "
            "result is rank-2 when keepdims=True and rank-1 otherwise."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-022",
        guidance=(
            "Use tf.reduce_mean(x, axis=0|1, keepdims=True|False), keeping "
            "keepdims named when present."
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
        guidance=("Call tf.nn.sigmoid(x) with a TensorF32Cpu2D positional argument."),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/sigmoid-f32-cpu-1d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.sigmoid on float32 CPU rank-1 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-1 tensor operand and no keywords. "
            "The owned same-wheel TFE Sigmoid result preserves float32, CPU:0, and rank 1."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-019",
        guidance="Call tf.nn.sigmoid(x) with a TensorF32Cpu1D positional argument.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/reduce-sum-axis1-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.reduce_sum(x, axis=1) on float32 CPU rank-2 tensors "
                "(literal keyword or positional axis)"
            ),
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal axis=1, passed either "
            "positionally with aligned metadata or by keyword; keepdims is "
            "omitted or False. The owned TFE Sum wrapper returns a float32 "
            "CPU rank-1 EagerTensor."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-011",
        guidance=(
            "Write tf.reduce_sum(x, axis=1) with a TensorF32Cpu2D operand and "
            "a static axis=1 literal."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/reduce-sum-literal-axis-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=(
                "tf.reduce_sum on float32 CPU rank-2 tensors with literal "
                "axis=0|1 and named literal keepdims=True|False"
            ),
        ),
        constraint=(
            "Axis is a statically proven int literal 0 or 1, passed by keyword "
            "or as one aligned positional literal. keepdims is omitted or a "
            "named bool literal; positional keepdims is excluded. TFE Sum "
            "receives only the tensor plus an owned int32 axis handle. The "
            "result is rank-2 when keepdims=True and rank-1 otherwise."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-023",
        guidance=(
            "Use tf.reduce_sum(x, axis=0|1, keepdims=True|False), keeping "
            "keepdims named when present."
        ),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/tanh-f32-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.tanh on float32 CPU rank-2 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-2 tensor operand and no keywords. "
            "The active TensorFlow 2.21.0 wheel executes the owned TFE Tanh op; "
            "the result preserves the operand type."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-009",
        guidance=("Call tf.nn.tanh(x) with a TensorF32Cpu2D positional argument."),
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/tanh-f32-cpu-1d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.tanh on float32 CPU rank-1 tensors",
        ),
        constraint=(
            "One positional float32 CPU rank-1 tensor operand and no keywords. "
            "The owned same-wheel TFE Tanh result preserves float32, CPU:0, and rank 1."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-020",
        guidance="Call tf.nn.tanh(x) with a TensorF32Cpu1D positional argument.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/abs-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.abs on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Abs result preserves rank, "
            "dtype, and CPU:0 residency."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-026",
        guidance="Call tf.abs(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/negative-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.negative on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Neg result preserves rank, "
            "dtype, and CPU:0 residency."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-027",
        guidance="Call tf.negative(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/square-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.square on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Square result preserves rank, "
            "dtype, and CPU:0 residency."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-028",
        guidance="Call tf.square(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/exp-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.exp on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Exp result preserves rank, "
            "dtype, and CPU:0 residency."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-029",
        guidance="Call tf.exp(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/log-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.math.log on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Log result preserves rank, "
            "dtype, and CPU:0 residency, including TensorFlow domain behavior."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-030",
        guidance="Call tf.math.log(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/sqrt-f32-cpu",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.math.sqrt on float32 CPU rank-1/rank-2 tensors",
        ),
        constraint=(
            "Exactly one positional float32 CPU rank-1 or rank-2 tensor and no "
            "keywords. The owned same-wheel TFE Sqrt result preserves rank, "
            "dtype, CPU:0 residency, domain behavior, and signed zero."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-031",
        guidance="Call tf.math.sqrt(x) with TensorF32Cpu1D or TensorF32Cpu2D.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/softmax-axis0-f32-cpu-1d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=("tf.nn.softmax(x) or tf.nn.softmax(x, axis=0) on a float32 CPU rank-1 tensor"),
        ),
        constraint=(
            "One float32 CPU rank-1 tensor with axis omitted or explicit literal "
            "axis=0, passed by keyword or as one aligned positional literal, "
            "and no other keywords. Raw TFE Softmax operates on the final axis "
            "and returns float32 CPU rank-1."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-025",
        guidance=(
            "Call tf.nn.softmax(x) or tf.nn.softmax(x, axis=0) with a TensorF32Cpu1D operand."
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
            "One float32 CPU rank-2 tensor plus explicit literal axis=1, passed "
            "by keyword or as one aligned positional literal, and no other "
            "keywords. Raw TFE Softmax is final-axis-only and returns float32 "
            "CPU rank-2. Axis=0 remains fallback because transpose is excluded."
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
            "One float32 CPU rank-2 tensor plus literal axis=1, passed by keyword "
            "or as one aligned positional literal, and no other keywords. The "
            "owned TFE ArgMax wrapper receives a scalar int32 axis handle and "
            "materializes exactly an int64 CPU rank-1 EagerTensor."
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
        id="rextio-tensorflow/argmax-axis0-i64-cpu-2d",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern=("tf.argmax(x, axis=0) with default int64 output on float32 CPU rank-2"),
        ),
        constraint=(
            "One float32 CPU rank-2 tensor plus literal axis=0, passed by keyword "
            "or as one aligned positional literal, and no other keywords. The "
            "owned TFE ArgMax wrapper receives scalar int32 axis 0 and returns "
            "an int64 CPU rank-1 EagerTensor. output_type overrides are excluded."
        ),
        outcome="native",
        diagnostic_code="RXTP-TENSORFLOW-024",
        guidance="Use tf.argmax(x, axis=0) and retain the default int64 output.",
        stability="experimental",
        verified=True,
    ),
    RuleRecord(
        id="rextio-tensorflow/bias-add-unproven-fallback",
        provider="rextio-tensorflow",
        scope=RuleScope(
            kind="call",
            pattern="tf.nn.bias_add on TensorFlow tensors",
        ),
        constraint=(
            "Native claim is withheld until the exact TFE BiasAdd data_format "
            "attribute, required public-symbol provenance, broadcasting, dtype, "
            "device, and error semantics are certified on every native profile."
        ),
        outcome="fallback",
        diagnostic_code="RXTP-TENSORFLOW-021",
        guidance=(
            "Keep tf.nn.bias_add on Python, or use an already-supported explicit "
            "add spelling when its semantics are appropriate."
        ),
        stability="experimental",
        verified=False,
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
            "Keep the Alpha slice on float32 CPU rank-1/2 matmul, activations, "
            "exact math unary operations, add/multiply/subtract/divide, reductions, "
            "rank-1 softmax(axis=0/default), rank-2 softmax(axis=1), and "
            "default-int64 argmax; other dtypes, devices, ranks, aliases, and "
            "dynamic literals remain on the fallback."
        ),
        stability="experimental",
        verified=False,
    ),
)


def tensorflow_rule_records() -> tuple[RuleRecord, ...]:
    """Return stable ordered rule records."""
    return RULE_RECORDS


__all__ = ["RULE_RECORDS", "tensorflow_rule_records"]
