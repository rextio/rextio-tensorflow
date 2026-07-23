"""Declared TensorFlow package and symbol coverage for Alpha."""

from rextio.plugins.api import CoverageDecl

COVERAGE = CoverageDecl(
    packages=("tensorflow",),
    modules=(
        "tensorflow",
        "tensorflow.linalg",
        "tensorflow.nn",
        "tensorflow.math",
    ),
    symbols=(
        "tensorflow.matmul",
        "tensorflow.linalg.matmul",
        "tensorflow.nn.relu",
        "tensorflow.nn.sigmoid",
        "tensorflow.nn.tanh",
        "tensorflow.add",
        "tensorflow.math.add",
        "tensorflow.reduce_mean",
        "tensorflow.math.reduce_mean",
        "tensorflow.reduce_sum",
        "tensorflow.math.reduce_sum",
        "tensorflow.nn.softmax",
        "tensorflow.argmax",
    ),
)

__all__ = ["COVERAGE"]
