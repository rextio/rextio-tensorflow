"""Import-free public annotation vocabulary for rextio-tensorflow.

These marker classes intentionally import neither TensorFlow nor Rextio.
Runtime values remain ordinary ``tf.Tensor`` / EagerTensor objects; the
integrated Rextio analyzer resolves the exact dotted annotation spellings to
plugin types without executing the annotations.
"""

from __future__ import annotations

from typing import Any


class TensorF32Cpu2D:
    """A float32 CPU rank-2 tensor (plugin type ``rextio-tensorflow/tensor-f32-cpu-2d``)."""


class TensorF32Cpu1D:
    """A float32 CPU rank-1 tensor (plugin type ``rextio-tensorflow/tensor-f32-cpu-1d``)."""


class TensorI64Cpu1D:
    """An int64 CPU rank-1 tensor produced by the classification head."""


class TensorF32Cuda0_2D:
    """A float32 rank-2 EagerTensor already resident on TensorFlow ``GPU:0``."""


class TensorF32Cuda0_1D:
    """A float32 rank-1 EagerTensor already resident on TensorFlow ``GPU:0``."""


def __getattr__(name: str) -> Any:
    """Reject misspelled annotation names with the normal module error."""
    raise AttributeError(name)


__all__ = [
    "TensorF32Cpu1D",
    "TensorF32Cpu2D",
    "TensorF32Cuda0_1D",
    "TensorF32Cuda0_2D",
    "TensorI64Cpu1D",
]
