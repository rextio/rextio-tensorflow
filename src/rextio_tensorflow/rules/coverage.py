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
        "tensorflow.add",
        "tensorflow.math.add",
        "tensorflow.reduce_mean",
        "tensorflow.math.reduce_mean",
    ),
)

__all__ = ["COVERAGE"]
