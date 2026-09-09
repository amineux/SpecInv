"""KerOp filter_contract/v1: load only what KerOp writes; fail-closed otherwise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from specinv.operators import power_law_operator
from specinv.spectrum_contract import (
    CONTRACT_SCHEMA,
    JointPathError,
    default_fixture_stem,
    estimate_ill_posedness,
    find_kerop_contract_files,
    import_kerop,
    load_filter_contract,
    operator_from_contract,
    resolve_contract_stem,
    select_operator,
    walltime_task_name,
)


def _write_contract(
    directory: Path,
    *,
    n_modes: int = 32,
    operator_id: str = "kerop.spectral",
    schema: str = CONTRACT_SCHEMA,
    b: float = 0.5,
    extra_operators: list[dict] | None = None,
) -> Path:
    evals = np.arange(1, n_modes + 1, dtype=np.float64) ** -1.5
    header = {
        "schema": schema,
        "contract_version": 1,
        "producer": "kerop",
        "producer_version": "test",
        "operators": [
            {
                "operator_id": operator_id,
                "seed": 20260301,
                "n": 150,
                "n_features": 12,
                "feature_count": 12,
                "b": b,
                "r": 0.5,
                "kind": "kernel_integral_operator",
                "arrays": {
                    "eigenvalues": "spectral/eigenvalues",
                    "rf_spectrum": "spectral/rf_spectrum",
                },
                "reproduce": {
                    "task_kwargs": {"n_modes": 512},
                    "train_sizes": [150, 300, 600],
                },
                "recorded_bar": {"median_speedup_at_matched_risk": 27.7},
            }
        ],
    }
    if extra_operators:
        header["operators"].extend(extra_operators)
    stem = directory / "filter_contract_v1"
    stem.with_suffix(".json").write_text(json.dumps(header))
    arrays = {
        "spectral/eigenvalues": evals,
        "spectral/rf_spectrum": np.ones(12, dtype=np.float64),
    }
    np.savez(stem.with_suffix(".npz"), **arrays)
    return stem


def test_estimate_ill_posedness_recovers_power_law() -> None:
    sigma = power_law_operator(128, 1.5).singular_values
    assert estimate_ill_posedness(sigma) == pytest.approx(1.5, rel=1e-6)


def test_load_filter_contract_and_operator(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, n_modes=64, b=0.5)
    contract = load_filter_contract(stem)
    assert contract.schema == CONTRACT_SCHEMA
    chosen = select_operator(contract, require_min_modes=64)
    assert chosen.operator_id == "kerop.spectral"
    assert chosen.n == 150
    assert chosen.n_modes == 64
    assert chosen.ill_posedness == pytest.approx(2.0)
    assert chosen.smoothness == pytest.approx(1.5)
    operator = operator_from_contract(chosen, n_modes=32)
    assert operator.n_modes == 32
    assert operator.ill_posedness == pytest.approx(2.0)


def test_resolve_stem_from_json_or_dir(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path)
    assert resolve_contract_stem(stem.with_suffix(".json")) == stem
    assert resolve_contract_stem(tmp_path) == stem


def test_load_missing_files_is_joint_path_error(tmp_path: Path) -> None:
    with pytest.raises(JointPathError, match="missing"):
        resolve_contract_stem(tmp_path / "nope")


def test_wrong_schema_is_rejected(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, schema="kerop.specinv.spectrum/v1")
    with pytest.raises(JointPathError, match="schema id"):
        load_filter_contract(stem)


def test_dirichlet1d_is_not_the_graded_inverse(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, n_modes=3072)
    dirichlet_evals = (np.arange(1, 13, dtype=np.float64) * np.pi) ** -2.0
    header = json.loads(stem.with_suffix(".json").read_text())
    header["operators"].append(
        {
            "operator_id": "kerop.dirichlet1d",
            "seed": 20260301,
            "n": 150,
            "n_features": 110,
            "feature_count": 110,
            "kind": "dirichlet1d_solution_operator",
            "arrays": {
                "eigenvalues": "dirichlet1d/eigenvalues",
                "rf_spectrum": "dirichlet1d/rf_spectrum",
            },
        }
    )
    stem.with_suffix(".json").write_text(json.dumps(header))
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as arrays:
        data = {k: arrays[k] for k in arrays.files}
    data["dirichlet1d/eigenvalues"] = dirichlet_evals
    data["dirichlet1d/rf_spectrum"] = np.ones(8)
    np.savez(stem.with_suffix(".npz"), **data)

    contract = load_filter_contract(stem)
    with pytest.raises(JointPathError, match="not FEM"):
        select_operator(contract, operator_id="kerop.dirichlet1d")
    aliased = contract.by_id("kerop.poisson")
    assert aliased.operator_id == "kerop.dirichlet1d"
    assert aliased.note is not None and "not a FEM" in aliased.note
    assert walltime_task_name("kerop.dirichlet1d") == "poisson"


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


def test_prefers_sibling_kerop_fixture(tmp_path: Path) -> None:
    kerop_root = tmp_path / "KerOp"
    dest = kerop_root / "fixtures"
    dest.mkdir(parents=True)
    stem = _write_contract(dest, n_modes=64, schema=CONTRACT_SCHEMA)
    found = find_kerop_contract_files(kerop_root)
    assert found == stem
    assert load_filter_contract(found).schema == CONTRACT_SCHEMA


def test_committed_fixture_is_kerops_export() -> None:
    stem = default_fixture_stem()
    if not stem.with_suffix(".json").is_file():
        pytest.skip("fixture not copied yet")
    contract = load_filter_contract(stem)
    assert contract.schema == CONTRACT_SCHEMA
    assert contract.producer == "kerop"
    spectral = select_operator(contract)
    assert spectral.operator_id == "kerop.spectral"
    assert spectral.n_modes >= 2048
    assert spectral.n == 150
    dirichlet = contract.by_id("kerop.dirichlet1d")
    assert dirichlet.n_modes == 12
    assert dirichlet.operator_id == "kerop.dirichlet1d"
    assert dirichlet.note is not None
    assert "not a FEM" in dirichlet.note
