"""Metric and rate-fitting behaviour, including recovery of a known slope."""

from __future__ import annotations

import numpy as np
import pytest

from specinv import fit_rate, relative_errors, summarise_errors
from specinv.theory import (
    deterministic_rate,
    optimal_truncation_index,
    predicted_error_terms,
    statistical_rate,
)


def test_relative_error_of_a_perfect_reconstruction_is_zero() -> None:
    truth = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(relative_errors(truth, truth), 0.0)


def test_relative_error_is_scale_invariant() -> None:
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((10, 8))
    prediction = truth + 0.1 * rng.standard_normal((10, 8))
    base = relative_errors(prediction, truth)
    scaled = relative_errors(1e6 * prediction, 1e6 * truth)
    assert np.allclose(base, scaled)


def test_relative_error_known_value() -> None:
    truth = np.array([[3.0, 4.0]])          # norm 5
    prediction = np.array([[3.0, 3.0]])     # residual norm 1
    assert relative_errors(prediction, truth)[0] == pytest.approx(0.2)


def test_summary_pools_consistently() -> None:
    rng = np.random.default_rng(1)
    truth = rng.standard_normal((200, 16))
    prediction = truth + 0.05 * rng.standard_normal((200, 16))
    summary = summarise_errors(prediction, truth)
    per_sample = relative_errors(prediction, truth)
    assert summary.mean_relative == pytest.approx(float(np.mean(per_sample)))
    assert summary.median_relative == pytest.approx(float(np.median(per_sample)))
    assert summary.n_samples == 200
    assert summary.stderr_relative == pytest.approx(
        summary.std_relative / np.sqrt(200)
    )
    assert 0.0 < summary.aggregate_relative < 1.0


def test_fit_rate_recovers_an_exact_power_law() -> None:
    deltas = np.geomspace(1e-1, 1e-5, 9)
    errors = 0.77 * deltas**0.5
    fit = fit_rate(deltas, errors)
    assert fit.slope == pytest.approx(0.5, abs=1e-10)
    assert fit.prefactor == pytest.approx(0.77, rel=1e-8)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.stderr == pytest.approx(0.0, abs=1e-8)


def test_fit_rate_confidence_interval_brackets_a_noisy_slope() -> None:
    rng = np.random.default_rng(2)
    deltas = np.geomspace(1e-1, 1e-6, 12)
    errors = 0.8 * deltas**0.5 * np.exp(0.02 * rng.standard_normal(12))
    fit = fit_rate(deltas, errors)
    assert fit.ci95[0] <= 0.5 <= fit.ci95[1]


def test_fit_rate_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        fit_rate(np.array([0.1, 0.01]), np.array([0.3, 0.1]))
    with pytest.raises(ValueError):
        fit_rate(np.array([0.1, 0.0, 0.001]), np.array([0.3, 0.1, 0.03]))


def test_theory_rates() -> None:
    assert deterministic_rate(1.5, 1.5) == pytest.approx(0.5)
    assert statistical_rate(1.5, 1.5) == pytest.approx(3 / 7)
    assert deterministic_rate(3.0, 1.0) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        deterministic_rate(0.0, 1.0)


def test_optimal_truncation_index_exponent() -> None:
    """N ~ delta^{-1/(s+p)} = delta^{-1/3} for s=p=1.5."""
    n = optimal_truncation_index(np.array([1e-3, 1e-6]), 1.5, 1.5)
    assert n[0] == pytest.approx(10.0, rel=1e-8)
    assert n[1] == pytest.approx(100.0, rel=1e-8)


def test_predicted_error_terms_balance_at_the_theoretical_index() -> None:
    """Theorem 4.6 balances E_1 = delta*N^p against E_3 = N^{-s}."""
    for delta in [1e-2, 1e-4, 1e-6]:
        index = float(optimal_truncation_index(delta, 1.5, 1.5))
        terms = predicted_error_terms(delta, int(round(index)), 1.5, 1.5)
        # Rounding the index to an integer perturbs the balance, so allow slack; the
        # exponent identity itself is checked exactly below.
        assert terms.stability == pytest.approx(terms.truncation, rel=0.35)


def test_predicted_error_at_the_optimum_scales_as_delta_to_the_rate() -> None:
    """Substituting N ~ delta^{-1/(s+p)} must give exactly delta^{s/(s+p)}."""
    rate = deterministic_rate(1.5, 1.5)
    for delta in [1e-3, 1e-5, 1e-7]:
        index = float(optimal_truncation_index(delta, 1.5, 1.5))
        stability = delta * index**1.5
        truncation = index**-1.5
        assert stability == pytest.approx(delta**rate, rel=1e-9)
        assert truncation == pytest.approx(delta**rate, rel=1e-9)
