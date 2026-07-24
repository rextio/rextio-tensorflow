"""GPU-free contracts for the opt-in real-NVIDIA CUDA E3 harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "certify_cuda_candidate.py"


def _module():
    spec = importlib.util.spec_from_file_location("certify_cuda_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_requires_output_and_exposes_real_gpu_contract() -> None:
    module = _module()
    parser = module.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])

    help_text = parser.format_help()
    assert "real-NVIDIA" in help_text
    assert "--sm" in help_text
    assert "--expected-tensorflow-commit" in help_text


def test_environment_validation_rejects_non_clean_or_wrong_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    clean = module.CheckoutIdentity(
        root=tmp_path,
        head="16e368a000000000000000000000000000000000",
        dirty=False,
    )
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: "1")
    with pytest.raises(RuntimeError, match="descended"):
        module.validate_checkout(
            clean,
            expected="16e368a000000000000000000000000000000000",
            required_ancestor="16e368a000000000000000000000000000000000",
        )
    with pytest.raises(RuntimeError, match="clean"):
        module.validate_checkout(
            module.CheckoutIdentity(tmp_path, clean.head, True),
            expected=clean.head,
            required_ancestor=clean.head,
        )


def test_source_contract_rejects_transfer_tokens_and_wrong_chain() -> None:
    module = _module()
    valid = "\n".join(module.E3_RUST_CALLS)
    module.assert_frozen_source_contract(valid)

    with pytest.raises(RuntimeError, match="transfer"):
        module.assert_frozen_source_contract(valid + "\nTFE_TensorHandleResolve")
    with pytest.raises(RuntimeError, match="chain"):
        module.assert_frozen_source_contract("\n".join(reversed(module.E3_RUST_CALLS)))


def test_atomic_write_never_leaves_partial_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    output = tmp_path / "evidence.json"
    module.atomic_write_json(output, {"state": "complete"})
    assert output.read_text(encoding="utf-8") == '{"state":"complete"}\n'
    assert not list(tmp_path.glob(".evidence.json.*"))

    monkeypatch.setattr(module.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("no")))
    with pytest.raises(OSError, match="no"):
        module.atomic_write_json(output, {"state": "partial"})
    assert output.read_text(encoding="utf-8") == '{"state":"complete"}\n'
    assert not list(tmp_path.glob(".evidence.json.*"))
