"""The joint script is fail-closed: KerOp filter_contract/v1 or nothing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from specinv.experiments import joint
from specinv.experiments._common import build_suite
from specinv.spectrum_contract import CONTRACT_SCHEMA, JointPathError


def _write_contract(
    directory: Path,
    *,
    n_modes: int = 64,
    schema: str = CONTRACT_SCHEMA,
    operator_id: str = "kerop.spectral",
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
                "b": 2.0 / 3.0,
                "r": 0.5,
                "kind": "kernel_integral_operator",
                "arrays": {
                    "eigenvalues": "spectral/eigenvalues",
                    "rf_spectrum": "spectral/rf_spectrum",
                },
                "reproduce": {
                    "task_kwargs": {"n_modes": 512, "n_test": 1000},
                    "train_sizes": [150, 300, 600],
                },
                "recorded_bar": {"median_speedup_at_matched_risk": 27.7},
            }
        ],
    }
    stem = directory / "filter_contract_v1"
    stem.with_suffix(".json").write_text(json.dumps(header))
    np.savez(
        stem.with_suffix(".npz"),
        **{
            "spectral/eigenvalues": evals,
            "spectral/rf_spectrum": np.ones(12, dtype=np.float64),
        },
    )
    return stem


def test_missing_contract_is_fail_closed(tmp_path: Path) -> None:
    code = joint.main(
        [
            "--results-dir",
            str(tmp_path / "out"),
            "--contract",
            str(tmp_path / "missing"),
            "--quick",
            "--no-figures",
        ]
    )
    assert code == 2
    assert not (tmp_path / "out" / "joint.json").exists()


def test_wrong_schema_is_not_a_joint_result(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, schema="kerop.specinv.spectrum/v1")
    code = joint.main(
        [
            "--results-dir",
            str(tmp_path / "out"),
            "--contract",
            str(stem),
            "--quick",
            "--no-figures",
        ]
    )
    assert code == 2
    assert not (tmp_path / "out" / "joint.json").exists()


def test_skip_kerop_bar_is_a_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, n_modes=2048)

    def fake_specinv(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "status": "pass",
            "pass": True,
            "graded": False,
            "quick": True,
            "n_pass": 3,
            "n_total": 3,
            "required_pass": 3,
            "required_total": 3,
            "operator_id": "kerop.spectral",
            "n_modes": 2048,
            "smoothness": 1.5,
            "ill_posedness": 1.5,
            "suite_dir": str(tmp_path),
            "criteria": [],
        }

    monkeypatch.setattr(joint, "run_specinv_bar", fake_specinv)
    code = joint.main(
        [
            "--results-dir",
            str(tmp_path / "out"),
            "--contract",
            str(stem),
            "--quick",
            "--no-figures",
            "--skip-kerop-bar",
        ]
    )
    assert code == 1
    payload = json.loads((tmp_path / "out" / "joint.json").read_text())
    assert payload["joint_status"] == "fail"
    assert payload["joint_pass"] is False
    assert payload["schema"] == CONTRACT_SCHEMA
    assert "live matched-risk number" in payload["kerop"]["reason"]


def test_build_suite_from_contract_does_not_use_a_bare_power_law(tmp_path: Path) -> None:
    stem = _write_contract(tmp_path, n_modes=64)
    contract_evals = np.arange(1, 65, dtype=np.float64) ** -1.5
    suite = build_suite(n_modes=64, spectrum_path=stem)
    assert suite.operator.n_modes == 64
    assert np.allclose(suite.operator.singular_values, contract_evals)


def test_resolve_refuses_to_invent_a_1d_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(joint, "find_kerop_contract_files", lambda root=None: None)
    with pytest.raises(JointPathError, match="No fallback"):
        joint.resolve_contract(
            contract_arg=None,
            kerop_root=tmp_path / "no-kerop",
            results_dir=tmp_path,
            write_if_missing=False,
        )
