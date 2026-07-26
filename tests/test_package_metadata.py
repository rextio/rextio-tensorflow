"""Package metadata contracts for the 0.1.3 Unreleased Alpha candidate."""

from __future__ import annotations

import tomllib
from pathlib import Path

from rextio_tensorflow import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]
CHANGELOG = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
CI_WORKFLOW = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
    encoding="utf-8"
)


def test_release_metadata_is_public_alpha() -> None:
    """The package remains Alpha without any private-upload classifier."""
    classifiers = PROJECT["classifiers"]
    assert "Development Status :: 3 - Alpha" in classifiers
    assert not any(classifier.startswith("Private ::") for classifier in classifiers)
    assert PROJECT["description"].startswith("Public Alpha ")


def test_release_version_and_exact_tensorflow_pin() -> None:
    """The candidate preserves its exact private-ABI runtime boundary."""
    assert __version__ == "0.1.3"
    assert "tensorflow==2.21.0" in PROJECT["dependencies"]
    assert "rextio>=0.1.6,<0.2" in PROJECT["dependencies"]


def test_changelog_marks_013_unreleased_and_preserves_012_history() -> None:
    assert "## [0.1.3] — Unreleased" in CHANGELOG
    assert "## [0.1.2] — 2026-07-26" in CHANGELOG
    assert "context-bound prepared-constant cache" in CHANGELOG
    assert "tf.transpose" in CHANGELOG
    assert "FunctionDef" in CHANGELOG
    assert "not** delivered in 0.1.3" in CHANGELOG or "not delivered in 0.1.3" in CHANGELOG


def test_ci_triggers_include_013_branch() -> None:
    assert '- "0.1.3"' in CI_WORKFLOW or "- '0.1.3'" in CI_WORKFLOW
    assert 'version("rextio-tensorflow") == "0.1.3"' in CI_WORKFLOW
