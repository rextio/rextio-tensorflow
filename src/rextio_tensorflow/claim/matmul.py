"""Fail-closed claims for ``tf.matmul`` / ``tf.linalg.matmul``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_MATMUL,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

MATMUL_RULE = "rextio-tensorflow/matmul-f32-cpu-2d"
MATMUL_TARGETS = frozenset(
    {
        "tensorflow.matmul",
        "tensorflow.linalg.matmul",
        "tf.matmul",
        "tf.linalg.matmul",
    }
)


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim float32 CPU rank-2 matmul, else None."""
    if site.kind != "call" or site.target not in MATMUL_TARGETS:
        return None
    if site.receiver is not None:
        return NotCovered()
    operands = tuple(site.operand_types)
    if site.keywords or len(operands) != 2:
        return reject(
            site,
            DIAGNOSTIC_MATMUL,
            "only tf.matmul(a, b) with two positional tensors is supported",
            "Pass both matrices positionally; omit transpose/keyword forms.",
        )
    if any(operand is None for operand in operands):
        return NotCovered()
    if any(not is_tensor_type(operand) for operand in operands):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand types are outside the float32 CPU rank-1/2 tensor surface",
            "Annotate operands with rextio_tensorflow.types.TensorF32Cpu2D.",
        )
    if operands != (TENSOR_F32_CPU_2D, TENSOR_F32_CPU_2D):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            f"matmul requires two float32 CPU rank-2 tensors; got {operands!r}",
            "Use TensorF32Cpu2D for both matmul operands.",
        )
    return Claimed(rule_id=MATMUL_RULE, result_type=TENSOR_F32_CPU_2D)


__all__ = ["MATMUL_RULE", "MATMUL_TARGETS", "try_claim"]
