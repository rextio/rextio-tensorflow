"""Claim router: dispatch analysis sites to Alpha claim modules."""

from __future__ import annotations

from rextio.config.schema import RextioConfig
from rextio.plugins.api import ClaimResult, ClaimSite, NotCovered

from rextio_tensorflow.claim import (
    activations,
    add,
    classification,
    cuda,
    matmul,
    reductions,
    unary,
)

__all__ = ["claim"]


def claim(site: ClaimSite, config: RextioConfig) -> ClaimResult:
    """Decide, at analysis time, whether this plugin lowers the site.

    Deterministic by contract: pure function of site kind, target, operand
    types, receiver metadata, and static keyword literals.
    """
    del config
    for lane in (cuda, matmul, activations, unary, add, reductions, classification):
        result = lane.try_claim(site)
        if result is not None:
            return result
    return NotCovered()
