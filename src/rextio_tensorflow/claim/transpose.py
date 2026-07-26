"""Fail-closed claims for default rank-2 float32 CPU ``tf.transpose``."""

from __future__ import annotations

from rextio.plugins.api import Claimed, ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.diagnostics import (
    DIAGNOSTIC_TRANSPOSE,
    DIAGNOSTIC_UNSUPPORTED,
    TENSOR_F32_CPU_2D,
    is_tensor_type,
    reject,
)

TRANSPOSE_RULE = "rextio-tensorflow/transpose-f32-cpu-2d"
TRANSPOSE_TARGETS = frozenset(
    {
        "tensorflow.transpose",
        "tf.transpose",
    }
)


def _literal_metadata_is_aligned(site: ClaimSite) -> bool:
    return not site.operand_literals or (
        len(site.operand_literals) == 1
        and not site.operand_literals[0].is_literal
    )


def try_claim(site: ClaimSite) -> ClaimResult | None:
    """Claim default ``tf.transpose(x)`` on float32 CPU rank-2 tensors only."""
    if site.kind != "call":
        return None
    if site.target not in TRANSPOSE_TARGETS:
        return None
    if site.receiver is not None:
        return NotCovered()
    if (
        len(site.operand_types) != 1
        or site.keywords
        or not _literal_metadata_is_aligned(site)
    ):
        return reject(
            site,
            DIAGNOSTIC_TRANSPOSE,
            (
                "bounded transpose requires exactly one positional float32 CPU "
                "rank-2 tensor and no keywords (default perm only)"
            ),
            (
                "Call tf.transpose(x) with a TensorF32Cpu2D operand; omit perm, "
                "conjugate, and name."
            ),
        )
    operand = site.operand_types[0]
    if operand is None:
        return NotCovered()
    if not is_tensor_type(operand):
        return reject(
            site,
            DIAGNOSTIC_UNSUPPORTED,
            "operand type is outside the float32 CPU tensor surface",
            "Annotate the operand as rextio_tensorflow.types.TensorF32Cpu2D.",
        )
    if operand != TENSOR_F32_CPU_2D:
        return reject(
            site,
            DIAGNOSTIC_TRANSPOSE,
            f"bounded transpose requires float32 CPU rank-2; got {operand!r}",
            "Use TensorF32Cpu2D for the default transpose operand.",
        )
    return Claimed(rule_id=TRANSPOSE_RULE, result_type=TENSOR_F32_CPU_2D)


__all__ = ["TRANSPOSE_RULE", "TRANSPOSE_TARGETS", "try_claim"]
