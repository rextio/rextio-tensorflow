"""Release metadata contracts for the public 0.1.0 Alpha."""

from __future__ import annotations

import tomllib
from pathlib import Path

from rextio_tensorflow import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "project"
]


def test_release_metadata_is_public_alpha() -> None:
    """The package remains Alpha without any private-upload classifier."""
    classifiers = PROJECT["classifiers"]
    assert "Development Status :: 3 - Alpha" in classifiers
    assert not any(classifier.startswith("Private ::") for classifier in classifiers)
    assert PROJECT["description"].startswith("Public Alpha ")


def test_release_version_and_exact_tensorflow_pin() -> None:
    """The release preserves its exact private-ABI runtime boundary."""
    assert __version__ == "0.1.0"
    assert "tensorflow==2.21.0" in PROJECT["dependencies"]
    assert "rextio>=0.1.3,<0.2" in PROJECT["dependencies"]
