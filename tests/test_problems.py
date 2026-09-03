"""The suite must realise the noise levels and decay laws it claims."""

from __future__ import annotations

import numpy as np
import pytest

from specinv import InverseProblemSuite, NoiseModel, SobolevPrior, power_law_operator


def make_suite(noise_model: NoiseModel = NoiseModel.CRITICAL) -> InverseProblemSuite:
    return InverseProblemSuite(
        power_law_operator(512, 1.5), SobolevPrior(1.5), noise_model
    )


def test_prior_decay_exponent() -> None:
    """Sec. 5.1: |f_n| ~ n^{-(s+1/2)}."""
    prior = SobolevPrior(smoothness=1.5)
    assert prior.decay_exponent == 2.0
    scale = prior.coefficient_scale(64)
    assert scale[0] == pytest.approx(1.0)
    assert scale[15] == pytest.approx(16.0**-2.0)


def test_prior_sample_matches_scale() -> None:
    prior = SobolevPrior(smoothness=1.5)
    samples = prior.sample(20000, 8, np.random.default_rng(0))
    empirical = samples.std(axis=0)
    assert np.allclose(empirical, prior.coefficient_scale(8), rtol=0.05)


def test_white_energy_noise_hits_delta_exactly() -> None:
    suite = make_suite(NoiseModel.WHITE_ENERGY)
    batch = suite.sample(64, 0.05, np.random.default_rng(1))
    assert np.allclose(batch.realised_noise_level, 0.05, rtol=1e-10)


@pytest.mark.parametrize(
    "noise_model", [NoiseModel.CRITICAL, NoiseModel.WHITE_POINTWISE]
)
def test_noise_scales_linearly_with_delta(noise_model: NoiseModel) -> None:
    suite = make_suite(noise_model)
    levels = []
    for delta in [0.01, 0.1]:
        batch = suite.sample(512, delta, np.random.default_rng(2))
        levels.append(float(np.mean(batch.realised_noise_level)))
    assert levels[1] / levels[0] == pytest.approx(10.0, rel=0.02)


def test_critical_noise_profile_is_n_to_the_minus_half() -> None:
    suite = make_suite(NoiseModel.CRITICAL)
    batch = suite.sample(4000, 0.1, np.random.default_rng(3))
    empirical = (batch.noisy_data - batch.clean_data).std(axis=0)
    n = np.arange(1, batch.n_modes + 1, dtype=float)
    ratio = empirical / (n**-0.5)
    # The profile is n^{-1/2}, so the ratio must be flat across the spectrum.
    assert np.std(ratio) / np.mean(ratio) < 0.05


def test_critical_noise_energy_grows_only_logarithmically() -> None:
    """The critical colouring sits on the borderline of L^2, hence log growth."""
    suite = InverseProblemSuite(power_law_operator(4096, 1.5), SobolevPrior(1.5))
    energies = {}
    for n_modes in [256, 4096]:
        batch = suite.sample(200, 0.1, np.random.default_rng(4), n_modes=n_modes)
        residual = batch.noisy_data - batch.clean_data
        energies[n_modes] = float(np.mean(np.linalg.norm(residual, axis=-1) ** 2))
    growth = energies[4096] / energies[256]
    harmonic = np.log(4096) / np.log(256)
    assert growth == pytest.approx(harmonic, rel=0.15)


def test_observable_rates() -> None:
    assert make_suite(NoiseModel.CRITICAL).theoretical_rate() == pytest.approx(0.5)
    assert make_suite(NoiseModel.WHITE_ENERGY).theoretical_rate() == pytest.approx(
        1.5 / 3.5
    )
    assert make_suite().deterministic_rate() == pytest.approx(0.5)


def test_signal_noise_crossover_and_band() -> None:
    """The crossover sets where a noise-floor estimate is valid."""
    suite = make_suite()
    # p + a - q = 1.5 + 2.0 - 0.5 = 3, so n* = delta^{-1/3}.
    assert suite.signal_noise_crossover(1e-3) == pytest.approx(10.0, rel=1e-6)
    assert suite.recommended_noise_band(1e-3, 256) == (30, 60)
    with pytest.raises(ValueError):
        suite.recommended_noise_band(1e-9, 256)


def test_sampling_is_deterministic_given_the_seed() -> None:
    suite = make_suite()
    a = suite.sample(8, 0.05, np.random.default_rng(7))
    b = suite.sample(8, 0.05, np.random.default_rng(7))
    assert np.array_equal(a.noisy_data, b.noisy_data)
    assert np.array_equal(a.true_coefficients, b.true_coefficients)


def test_resolution_change_extends_rather_than_replaces() -> None:
    """Refining the grid must keep the same statistics on the shared modes."""
    suite = InverseProblemSuite(power_law_operator(2048, 1.5), SobolevPrior(1.5))
    coarse = suite.sample(3000, 0.05, np.random.default_rng(11), n_modes=256)
    fine = suite.sample(3000, 0.05, np.random.default_rng(11), n_modes=2048)
    assert np.allclose(
        coarse.true_coefficients, fine.true_coefficients[:, :256], atol=0
    )
    coarse_std = coarse.noise_scale[:, :64].mean(axis=0)
    fine_std = fine.noise_scale[:, :64].mean(axis=0)
    assert np.allclose(coarse_std, fine_std, rtol=0.05)


def test_zero_delta_is_noise_free() -> None:
    suite = make_suite()
    batch = suite.sample(4, 0.0, np.random.default_rng(0))
    assert np.array_equal(batch.noisy_data, batch.clean_data)


def test_invalid_inputs() -> None:
    suite = make_suite()
    with pytest.raises(ValueError):
        suite.sample(4, -0.1, np.random.default_rng(0))
    with pytest.raises(ValueError):
        SobolevPrior(smoothness=-1.0)
    with pytest.raises(ValueError):
        SobolevPrior(amplitude="unknown")  # type: ignore[arg-type]