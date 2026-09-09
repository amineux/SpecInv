"""KerOp ↔ SpecInv spectrum artifact: format, export, and fail-closed import."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from specinv.operators import power_law_operator
from specinv.spectrum_contract import (
    ARTIFACT_SCHEMA,
    JOINT_B,
    JOINT_N_MODES,
    JointPathError,
    default_fixture_path,
    estimate_ill_posedness,
    export_kerop_spectrum,
    fingerprint,
    import_kerop,
    load_spectrum_artifact,
    operator_from_artifact,
    require_kerop,
    write_spectrum_artifact,
)


def test_estimate_ill_posedness_recovers_power_law() -> None:
    sigma = power_law_operator(128, 1.5).singular_values
    assert estimate_ill_posedness(sigma) == pytest.approx(1.5, rel=1e-6)


def test_export_requires_kerop_and_matches_section_5_1_weights() -> None:
    kerop = require_kerop()
    artifact = export_kerop_spectrum(n_modes=64)
    assert artifact.schema == ARTIFACT_SCHEMA
    assert artifact.source == "kerop.SpectralOperatorModel"
    assert artifact.kerop_version == kerop.__version__
    assert artifact.n == 64
    assert artifact.feature_count >= 1
    ranks = np.arange(1, 65, dtype=np.float64)
    expected = ranks ** (-1.0 / JOINT_B)
    assert np.allclose(artifact.eigenvalues, expected, rtol=1e-12, atol=1e-14)
    # Same numbers as SpecInv's old generator — provenance is KerOp, not a fallback.
    paper = power_law_operator(64, 1.5).singular_values
    assert np.allclose(artifact.eigenvalues, paper, rtol=1e-12, atol=1e-14)
    operator = operator_from_artifact(artifact)
    assert operator.ill_posedness == pytest.approx(1.5)
    assert operator.n_modes == 64


def test_npz_and_json_roundtrip(tmp_path: Path) -> None:
    artifact = export_kerop_spectrum(n_modes=32)
    dest = tmp_path / "spectrum.npz"
    write_spectrum_artifact(dest, artifact)
    loaded_npz = load_spectrum_artifact(dest)
    loaded_json = load_spectrum_artifact(dest.with_suffix(".json"))
    assert fingerprint(loaded_npz) == fingerprint(artifact)
    assert fingerprint(loaded_json) == fingerprint(artifact)
    assert loaded_npz.operator_id == artifact.operator_id
    assert loaded_npz.seed == artifact.seed
    assert loaded_npz.feature_count == artifact.feature_count
    header = json.loads(dest.with_suffix(".json").read_text())
    assert header["schema"] == ARTIFACT_SCHEMA
    assert header["n"] == 32
    assert "eigenvalues" in header


def test_load_missing_file_is_joint_path_error(tmp_path: Path) -> None:
    with pytest.raises(JointPathError, match="not found"):
        load_spectrum_artifact(tmp_path / "nope.npz")


def test_import_kerop_fails_closed_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    import specinv.spectrum_contract as contract

    monkeypatch.setattr(contract, "candidate_kerop_roots", lambda explicit=None: [])

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "kerop" or name.startswith("kerop."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(JointPathError, match="not importable"):
        import_kerop()


def test_committed_fixture_is_a_contract_export() -> None:
    path = default_fixture_path()
    if not path.is_file():
        pytest.skip("fixture not generated yet")
    artifact = load_spectrum_artifact(path)
    assert artifact.n == JOINT_N_MODES
    assert artifact.source == "kerop.SpectralOperatorModel"
    live = export_kerop_spectrum(n_modes=artifact.n)
    assert np.allclose(artifact.eigenvalues, live.eigenvalues, rtol=1e-12, atol=1e-14)
    assert artifact.operator_id == live.operator_id
