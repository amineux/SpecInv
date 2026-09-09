"""The joint script is fail-closed: no KerOp → no silent 1-D fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from specinv.experiments import joint
from specinv.experiments._common import build_suite
from specinv.spectrum_contract import (
    JointPathError,
    export_kerop_spectrum,
    write_spectrum_artifact,
)


def test_joint_exits_2_when_kerop_cannot_be_imported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(root: object = None) -> None:
        raise JointPathError(
            "kerop is not importable. This is a FAIL, not a joint result — "
            "the old 1D suite will not be substituted."
        )

    monkeypatch.setattr(joint, "require_kerop", boom)
    code = joint.main(["--results-dir", str(tmp_path), "--quick", "--no-figures"])
    assert code == 2
    assert not (tmp_path / "joint.json").exists()


def test_skip_kerop_bar_is_a_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = export_kerop_spectrum(n_modes=32)
    dest = tmp_path / "spectrum.npz"
    write_spectrum_artifact(dest, artifact)

    monkeypatch.setattr(joint, "require_kerop", lambda root=None: object())
    monkeypatch.setattr(
        joint,
        "resolve_spectrum",
        lambda args: (artifact, dest, "path"),
    )
    monkeypatch.setattr(
        joint,
        "apply_filter_to_path_target",
        lambda art, args: {
            "operator_id": art.operator_id,
            "fingerprint": "x",
            "n_modes": 32,
            "n_test": 8,
            "delta": 0.05,
            "training": {},
            "scnet_mean_relative": 0.10,
            "oracle_tikhonov_mean_relative": 0.12,
            "beats_oracle_tikhonov_on_path_target": True,
            "filter_min": 0.0,
            "filter_max": 1.0,
            "filter_is_bounded": True,
        },
    )
    monkeypatch.setattr(
        joint,
        "_quick_specinv_bars",
        lambda art, args, path_target: {
            "graded_11_of_11": False,
            "passed_smoke": True,
            "rate_slope_three_point": 0.4,
            "beats_oracle_tikhonov": True,
            "zero_shot_drift_2x": 0.01,
            "path_target": path_target,
        },
    )
    code = joint.main(
        [
            "--results-dir",
            str(tmp_path / "out"),
            "--quick",
            "--no-figures",
            "--skip-kerop-bar",
        ]
    )
    assert code == 1
    payload = (tmp_path / "out" / "joint.json").read_text()
    assert "drops the matched-excess-risk requirement" in payload
    assert "all_passed" in payload


def test_build_suite_from_artifact_does_not_use_a_bare_power_law(tmp_path: Path) -> None:
    artifact = export_kerop_spectrum(n_modes=64)
    dest = tmp_path / "spectrum.npz"
    write_spectrum_artifact(dest, artifact)
    suite = build_suite(n_modes=64, spectrum_path=dest)
    assert suite.operator.n_modes == 64
    assert (suite.operator.singular_values == artifact.eigenvalues).all()


def test_joint_refuses_non_kerop_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = export_kerop_spectrum(n_modes=16)
    # Frozen dataclass: rebuild with a forged source via load after rewrite.
    dest = tmp_path / "forged.json"
    payload: dict[str, Any] = artifact.metadata()
    payload["source"] = "handwritten_1d_suite"
    payload["eigenvalues"] = artifact.eigenvalues.tolist()
    dest.write_text(__import__("json").dumps(payload))

    monkeypatch.setattr(joint, "require_kerop", lambda root=None: object())
    code = joint.main(
        [
            "--results-dir",
            str(tmp_path / "out"),
            "--spectrum",
            str(dest),
            "--quick",
            "--no-figures",
        ]
    )
    assert code == 2
