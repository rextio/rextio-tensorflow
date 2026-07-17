"""The rextio-tensorflow plugin object and entry-point factory.

Implements plugin API 1.3: describe/covers, annotation vocabulary, claim/lower,
and an empty crate dependency list (native code dlopens the active TensorFlow
wheel; no Cargo TensorFlow crate). This module never imports tensorflow;
user-facing types are also import-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rextio_tensorflow.__about__ import __version__

if TYPE_CHECKING:
    from rextio.config.schema import RextioConfig
    from rextio.plugins.api import (
        ClaimResult,
        ClaimSite,
        CoverageDecl,
        CrateDependency,
        LoweredExpr,
        LoweringContext,
        PluginType,
        RuleRecord,
    )
    from rextio.plugins.models import RextioPlugin

PLUGIN_ID = "rextio-tensorflow"
REQUIRED_PLUGIN_API = "1.3"

__all__ = ["PLUGIN_ID", "REQUIRED_PLUGIN_API", "RextioTensorflowPlugin", "plugin"]


def _require_api_13() -> None:
    from rextio.plugins.api import PLUGIN_API_VERSION

    if PLUGIN_API_VERSION != REQUIRED_PLUGIN_API:
        raise RuntimeError(
            "rextio-tensorflow requires Rextio plugin API 1.3 "
            f"(rextio>=0.1.3,<0.2); this environment advertises "
            f"PLUGIN_API_VERSION={PLUGIN_API_VERSION!r}"
        )


class RextioTensorflowPlugin:
    """Plugin API 1.3 provider for the Alpha float32 CPU TensorFlow slice."""

    plugin_id = PLUGIN_ID
    api_version = REQUIRED_PLUGIN_API

    def to_rextio_plugin(self) -> RextioPlugin:
        """Return the v1 metadata Rextio core registers this plugin under."""
        _require_api_13()
        from rextio.plugins.models import RextioPlugin

        from rextio_tensorflow.rules import COVERAGE

        return RextioPlugin(
            id=PLUGIN_ID,
            name=f"TensorFlow TFE (rextio-tensorflow {__version__})",
            source_language="python",
            target_language="rust",
            packages=COVERAGE.packages,
        )

    def covers(self) -> CoverageDecl:
        """Return the packages, modules, and symbols this plugin covers."""
        _require_api_13()
        from rextio_tensorflow.rules import COVERAGE

        return COVERAGE

    def describe(self, config: RextioConfig) -> tuple[RuleRecord, ...]:
        """Return the rule records for the resolved project configuration."""
        _require_api_13()
        from rextio_tensorflow.rules import tensorflow_rule_records

        del config
        return tensorflow_rule_records()

    def type_vocabulary(self) -> tuple[PluginType, ...]:
        """Return the annotation vocabulary this plugin adds to the analyzer."""
        _require_api_13()
        from rextio_tensorflow.plugin_types import plugin_types

        return plugin_types()

    def claim(self, site: ClaimSite, config: RextioConfig) -> ClaimResult:
        """Decide, at analysis time, whether this plugin lowers the site."""
        _require_api_13()
        from rextio_tensorflow.claim import claim as claim_site

        return claim_site(site, config)

    def lower(self, claimed: ClaimSite, ctx: LoweringContext) -> LoweredExpr:
        """Emit the Rust expression for a previously claimed site."""
        _require_api_13()
        from rextio_tensorflow.lower import lower as lower_site

        return lower_site(claimed, ctx)

    def crate_dependencies(self) -> tuple[CrateDependency, ...]:
        """Return no Cargo crates: runtime binds the active TF wheel via dlsym."""
        _require_api_13()
        return ()


def plugin() -> RextioTensorflowPlugin:
    """Entry-point factory for the ``rextio.plugins`` group."""
    return RextioTensorflowPlugin()
