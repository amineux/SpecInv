"""Classical filters must satisfy their defining identities and oracle optimality."""

from __future__ import annotations

import numpy as np
import pytest

from specinv import (
    InverseProblemSuite,
    SobolevPrior,
    landweber_damping,
    oracle_spectral_bound,
    oracle_tikhonov,
    oracle_tsvd,
    power_law_operator,
    relative_errors,
    tikhonov_damping,
    tsvd_damping,
    wiener_damping,
)
from specinv.filters import discrepancy_principle_tikhonov


def test_tikhonov_damping_matches_paper_filter_function() -> None:
    """lambda_n = sigma_n * g_alpha(sigma_n) with g_alpha = sigma/(sigma^2+alpha)."""
    sv = power_law_operator(32, 1.5).singular_values
    alpha = 1e-3
    g = sv / (sv**2 + alpha)
    assert np.allclose(tikhonov_damping(sv, alpha), sv * g)


def test_tikhonov_limits() -> None:
    sv = power_law_operator(16, 1.5).singular_values
    assert np.allclose(tikhonov_damping(sv, 0.0), 1.0)
    assert np.all(tikhonov_damping(sv, 1e12) < 1e-6)


def test_tsvd_is_a_step_function() -> None:
    sv = power_law_operator(64, 1.5).singular_values
    damping = tsvd_damping(sv, sv[9] ** 2)
    assert np.all(damping[:10] == 1.0)
    assert np.all(damping[10:] == 0.0)


def test_landweber_increases_with_iterations() -> None:
    sv = power_law_operator(32, 1.5).singular_values
    step = 0.5 / sv.max() ** 2
    d10 = landweber_damping(sv, 10, step)
    d100 = landweber_damping(sv, 100, step)
    assert np.all(d100 >= d10 - 1e-12)
    assert np.all((d10 >= 0.0) & (d10 <= 1.0))
    with pytest.raises(ValueError):
        landweber_damping(sv, 10, step=10.0)


def test_wiener_damping_is_monotone_in_noise() -> None:
    sv = power_law_operator(32, 1.5).singular_values
    signal = np.arange(1, 33, dtype=float) ** -2.0
    quiet = wiener_damping(sv, signal, np.full(32, 1e-6))
    loud = wiener_damping(sv, signal, np.full(32, 1e-2))
    assert np.all(quiet >= loud - 1e-12)
    assert np.all((loud >= 0.0) & (loud <= 1.0))


def _batch(delta: float = 0.05, n: int = 64):
    suite = InverseProblemSuite(power_law_operator(128, 1.5), SobolevPrior(1.5))
    batch = suite.sample(n, delta, np.random.default_rng(0), n_modes=128)
    return suite, batch, suite.operator.restrict(128).singular_values


def test_oracle_tikhonov_is_optimal_within_its_family() -> None:
    """No fixed alpha may beat the per-sample oracle on any sample."""
    _, batch, sv = _batch()
    oracle = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
    oracle_error = relative_errors(oracle.reconstruction, batch.true_coefficients)
    for alpha in [1e-8, 1e-5, 1e-3, 1e-1]:
        fixed = tikhonov_damping(sv, alpha) * (batch.noisy_data / sv)
        assert np.all(
            relative_errors(fixed, batch.true_coefficients) >= oracle_error - 1e-9
        )


def test_oracle_tsvd_is_optimal_over_cutoffs() -> None:
    _, batch, sv = _batch()
    oracle = oracle_tsvd(batch.noisy_data, sv, batch.true_coefficients)
    oracle_error = relative_errors(oracle.reconstruction, batch.true_coefficients)
    for cutoff in [1, 2, 4, 8, 16, 64]:
        damping = np.zeros_like(sv)
        damping[:cutoff] = 1.0
        fixed = damping * (batch.noisy_data / sv)
        assert np.all(
            relative_errors(fixed, batch.true_coefficients) >= oracle_error - 1e-9
        )


def test_oracle_spectral_bound_dominates_every_other_filter() -> None:
    """It is the pointwise minimiser, so nothing in [0,1] may beat it."""
    _, batch, sv = _batch()
    bound = oracle_spectral_bound(batch.noisy_data, sv, batch.true_coefficients)
    floor = relative_errors(bound.reconstruction, batch.true_coefficients)
    for other in (
        oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients),
        oracle_tsvd(batch.noisy_data, sv, batch.true_coefficients),
    ):
        assert np.all(
            relative_errors(other.reconstruction, batch.true_coefficients)
            >= floor - 1e-9
        )
    assert np.all((bound.damping >= 0.0) & (bound.damping <= 1.0))


def test_discrepancy_principle_matches_its_residual_target() -> None:
    _, batch, sv = _batch(delta=0.05)
    selection = discrepancy_principle_tikhonov(
        batch.noisy_data, sv, batch.noise_norm, tau=1.1
    )
    residual = np.linalg.norm((1.0 - selection.damping) * batch.noisy_data, axis=-1)
    assert np.allclose(residual, 1.1 * batch.noise_norm, rtol=1e-4)


def test_discrepancy_principle_is_worse_than_the_oracle() -> None:
    """A non-oracle rule cannot beat the oracle it approximates."""
    _, batch, sv = _batch(delta=0.05)
    practical = discrepancy_principle_tikhonov(batch.noisy_data, sv, batch.noise_norm)
    oracle = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
    assert np.all(
        relative_errors(practical.reconstruction, batch.true_coefficients)
        >= relative_errors(oracle.reconstruction, batch.true_coefficients) - 1e-9
    )


def test_discrepancy_principle_is_a_usable_regulariser() -> None:
    """Sanity: it must be far better than no regularisation at all."""
    _, batch, sv = _batch(delta=0.05, n=128)
    practical = discrepancy_principle_tikhonov(batch.noisy_data, sv, batch.noise_norm)
    naive = batch.noisy_data / sv
    assert np.mean(
        relative_errors(practical.reconstruction, batch.true_coefficients)
    ) < 0.1 * np.mean(relative_errors(naive, batch.true_coefficients))


def test_noise_norm_exceeds_the_per_mode_level_under_critical_colouring() -> None:
    """delta (per-mode) and ||eps|| (L2) are different quantities; keep them straight."""
    _, batch, _ = _batch(delta=0.05, n=256)
    clean_norm = np.linalg.norm(batch.clean_data, axis=-1)
    assert np.mean(batch.noise_norm / clean_norm) > 1.5 * 0.05


def test_regularisation_beats_the_naive_inverse() -> None:
    _, batch, sv = _batch(delta=0.05)
    naive = batch.noisy_data / sv
    oracle = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
    assert np.mean(
        relative_errors(oracle.reconstruction, batch.true_coefficients)
    ) < np.mean(relative_errors(naive, batch.true_coefficients))


def test_invalid_parameters_raise() -> None:
    sv = power_law_operator(8, 1.5).singular_values
    with pytest.raises(ValueError):
        tikhonov_damping(sv, -1.0)
    with pytest.raises(ValueError):
        tsvd_damping(sv, -1.0)
    with pytest.raises(ValueError):
        discrepancy_principle_tikhonov(np.zeros((2, 8)), sv, noise_norm=0.0)
    with pytest.raises(ValueError):
        discrepancy_principle_tikhonov(np.zeros((2, 8)), sv, noise_norm=0.1, tau=0.5)
