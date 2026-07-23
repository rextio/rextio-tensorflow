"""Lower router: dispatch claimed sites to Alpha lower modules."""

from __future__ import annotations

from rextio.plugins.api import ClaimSite, LoweredExpr, LoweringContext

from rextio_tensorflow.lower import activations, add, classification, matmul, reductions

__all__ = ["lower"]


def lower(claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
    """Emit the Rust expression for a previously claimed site.

    Independently revalidates authoritative claim metadata and fails closed
    with ``ValueError`` (not ``assert``) so guards survive ``python -O``.
    """
    for lane in (matmul, activations, add, reductions, classification):
        result = lane.try_lower(claimed, ctx)
        if result is not None:
            return result
    raise ValueError(
        f"rextio-tensorflow cannot lower unclaimed site: {claimed.kind} {claimed.target!r}"
    )
