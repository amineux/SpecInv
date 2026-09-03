"""End-to-end checks that the reproduction scripts run and emit valid artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from specinv.experiments import convergence, filters, run_all, zero_shot
from specinv.experiments._common import (
    PAPER_DELTAS,
    build_suite,
    environment_info,
    evaluate_methods,
)
from specinv.experiments.filters import half_power_mode, tail_mass, transition_width


def test_paper_deltas_match_section_5_2() -> None:
    assert PAPER_DELTAS == (1e-1, 5e-2, 1e-2, 5e-3, 1e-3)


def test_environment_info_is_serialisable() -> None:
    json.dumps(environment_info())


def test_evaluate_methods_runs_every_baseline_without_a_model() -> None:
    suite = build_suite(n_modes=256)
    batch = suite.sample(16, 0.05, np.random.default_rng(0), n_modes=256)
    outputs = evaluate_methods(None, suite, batch)
    assert "scnet" not in outputs.reconstructions
    for name in ("oracle_tikhonov", "oracle_tsvd", "prior_wiener"):
        assert outputs.reconstructions[name].shape == batch.true_coefficients.shape
    summaries = outputs.summarise(batch.true_coefficients)
    assert all(0.0 < s.mean_relative < 5.0 for s in summaries.values())


def test_transition_width_distinguishes_sharp_from_gradual() -> None:
    modes = np.arange(1, 65, dtype=float)
    step = (modes <= 8).astype(float)
    gradual = 1.0 / (1.0 + (modes / 8.0) ** 2)
    assert transition_width(step) < transition_width(gradual)


def test_half_power_mode_locates_the_cutoff() -> None:
    modes = np.arange(1, 65, dtype=float)
    assert half_power_mode((modes <= 10).astype(float)) == pytest.approx(10.0, abs=1.0)


def test_tail_mass_penalises_a_heavy_tail() -> None:
    modes = np.arange(1, 129, dtype=float)
    sharp = (modes <= 8).astype(float)
    heavy = 1.0 / (1.0 + (modes / 8.0) ** 2)
    assert tail_mass(sharp, 8.0) < tail_mass(heavy, 8.0)


@pytest.mark.slow
def test_convergence_script_writes_valid_output(tmp_path: Path) -> None:
    code = convergence.main(
        [
            "--results-dir", str(tmp_path),
            "--epochs", "12",
            "--n-train", "192",
            "--n-test", "96",
            "--no-figures",
            "--skip-extended",
            "--skip-white",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "convergence.json").read_text())
    sweep = payload["sweeps"]["paper_grid"]
    assert len(sweep["deltas"]) == len(PAPER_DELTAS)
    assert "scnet" in sweep["rate_fits"]
    assert payload["theory"]["deterministic_rate"] == pytest.approx(0.5)
    assert (tmp_path / "convergence.csv").exists()


@pytest.mark.slow
def test_zero_shot_script_writes_valid_output(tmp_path: Path) -> None:
    code = zero_shot.main(
        [
            "--results-dir", str(tmp_path),
            "--epochs", "12",
            "--n-train", "192",
            "--n-test", "96",
            "--no-figures",
            "--skip-cnn",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "zero_shot.json").read_text())
    assert payload["resolutions"] == [256, 512, 1024, 2048]
    curves = payload["primary"]["curves"]["scnet"]
    assert set(curves) == {"256", "512", "1024", "2048"}
    # With the trained aperture the operator acts on coefficients, so the error must be
    # essentially independent of the evaluation grid regardless of training quality.
    values = [curves[k] for k in ("256", "512", "1024", "2048")]
    assert max(values) / min(values) < 1.02


@pytest.mark.slow
def test_filters_script_verifies_its_own_claims(tmp_path: Path) -> None:
    code = filters.main(
        [
            "--results-dir", str(tmp_path),
            "--epochs", "150",
            "--n-train", "512",
            "--n-test", "192",
            "--no-figures",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "filters.json").read_text())
    checks = payload["claim_checks"]
    assert checks["filter_is_bounded_in_unit_interval"]
    assert checks["preserves_leading_modes"]
    assert checks["sharper_than_tikhonov"]
    assert (tmp_path / "filter_profiles.csv").exists()


def test_criteria_builder_reads_the_experiment_payloads() -> None:
    """Guards the criteria wiring without paying for a full run."""
    convergence_payload = {
        "sweeps": {
            "paper_grid": {
                "rate_fits": {"scnet": {"slope": 0.49}, "oracle_tikhonov": {"slope": 0.47}},
                "errors": {
                    "scnet": [0.25, 0.18, 0.08, 0.057, 0.026],
                    "oracle_tikhonov": [0.258, 0.188, 0.087, 0.063, 0.029],
                },
            },
            "extended_grid": {"rate_fits": {"scnet": {"slope": 0.497}}},
        },
        "theory": {"deterministic_rate": 0.5},
    }
    zero_shot_payload = {
        "resolutions": [256, 512, 1024, 2048],
        "summary": {
            "scnet_error_at_train_resolution": 0.2501,
            "scnet_error_at_finest_resolution": 0.2503,
            "relative_drift_train_to_finest": 0.0008,
            "scnet_full_aperture_drift": 0.01,
        },
        "matched_to_paper": {"curves": {"scnet": {"2048": 0.2300}}},
    }
    filters_payload = {
        "claim_checks": {
            "filter_is_bounded_in_unit_interval": True,
            "preserves_leading_modes": True,
            "suppresses_high_modes": True,
            "sharper_than_tikhonov": True,
            "lighter_tail_than_tikhonov": True,
        }
    }
    criteria = run_all.build_criteria(
        convergence_payload, zero_shot_payload, filters_payload
    )
    assert all(c.passed for c in criteria)
    names = {c.name for c in criteria}
    assert "beats_oracle_tikhonov" in names
    assert "zero_shot_stability" in names
    json.dumps([c.as_dict() for c in criteria])


def test_criteria_fail_when_scnet_loses_to_tikhonov() -> None:
    convergence_payload = {
        "sweeps": {
            "paper_grid": {
                "rate_fits": {"scnet": {"slope": 0.49}},
                "errors": {
                    "scnet": [0.30, 0.20],
                    "oracle_tikhonov": [0.258, 0.188],
                },
            }
        },
        "theory": {"deterministic_rate": 0.5},
    }
    zero_shot_payload = {
        "resolutions": [256],
        "summary": {
            "scnet_error_at_train_resolution": 0.25,
            "scnet_error_at_finest_resolution": 0.25,
            "relative_drift_train_to_finest": 0.0,
            "scnet_full_aperture_drift": 0.0,
        },
        "matched_to_paper": None,
    }
    filters_payload = {
        "claim_checks": dict.fromkeys(
            [
                "filter_is_bounded_in_unit_interval",
                "preserves_leading_modes",
                "suppresses_high_modes",
                "sharper_than_tikhonov",
                "lighter_tail_than_tikhonov",
            ],
            True,
        )
    }
    criteria = run_all.build_criteria(
        convergence_payload, zero_shot_payload, filters_payload
    )
    failed = {c.name for c in criteria if not c.passed}
    assert "beats_oracle_tikhonov" in failed