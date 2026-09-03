"""SC-Net's structural guarantees: boundedness, interpretability, mesh independence.

These are the properties the paper's theory assumes (Assumption 1, Theorem 4.1) and the
properties that distinguish the model from a spatial black box, so they are asserted
directly rather than inferred from end-to-end error numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specinv import (
    InverseProblemSuite,
    SCNet,
    SCNetConfig,
    SobolevPrior,
    power_law_operator,
)


def make_suite(n_modes: int = 2048) -> InverseProblemSuite:
    return InverseProblemSuite(power_law_operator(n_modes, 1.5), SobolevPrior(1.5))


def calibrated_net(config: SCNetConfig | None = None, n_modes: int = 256) -> SCNet:
    suite = make_suite()
    batch = suite.sample(64, 0.05, np.random.default_rng(0), n_modes=n_modes)
    net = SCNet(config or SCNetConfig())
    sv = torch.as_tensor(
        suite.operator.restrict(n_modes).singular_values, dtype=torch.float32
    )
    net.calibrate(torch.as_tensor(batch.noisy_data, dtype=torch.float32), sv)
    return net


def test_filter_respects_the_boundedness_constraint() -> None:
    """Sec. 3.1.2 requires Psi in [0, C_Psi]; the sigmoid output must enforce it."""
    suite = make_suite()
    for c_psi in [1.0, 2.5]:
        net = calibrated_net(SCNetConfig(c_psi=c_psi))
        for delta in [0.0, 1e-3, 0.1, 1.0]:
            batch = suite.sample(16, delta, np.random.default_rng(1), n_modes=256)
            sv = suite.operator.restrict(256).singular_values
            psi = net.filter_profile(batch.noisy_data, sv)
            assert np.all(psi >= 0.0)
            assert np.all(psi <= c_psi)


def test_reconstruction_equals_damping_times_naive_inverse() -> None:
    """Eq. (6) must hold exactly: the model is a spectral filter, nothing more."""
    suite = make_suite()
    net = calibrated_net()
    batch = suite.sample(8, 0.05, np.random.default_rng(2), n_modes=256)
    sv = suite.operator.restrict(256).singular_values
    psi = net.filter_profile(batch.noisy_data, sv)
    expected = psi * (batch.noisy_data / sv)
    assert np.allclose(net.reconstruct(batch.noisy_data, sv), expected, rtol=1e-5)


def test_filter_is_identical_across_resolutions() -> None:
    """Discretisation invariance at the level of the learned object itself.

    Given the same measurement coefficients, the damping at mode n must not depend on how
    many modes the grid happens to carry.  This is stronger than comparing errors and is
    what §3.3 actually claims.
    """
    suite = make_suite()
    net = calibrated_net()
    net.set_aperture(None)
    coarse = suite.sample(32, 0.05, np.random.default_rng(3), n_modes=256)
    fine = suite.sample(32, 0.05, np.random.default_rng(3), n_modes=2048)
    assert np.allclose(coarse.noisy_data, fine.noisy_data[:, :256])

    psi_coarse = net.filter_profile(
        coarse.noisy_data, suite.operator.restrict(256).singular_values
    )
    psi_fine = net.filter_profile(
        fine.noisy_data, suite.operator.restrict(2048).singular_values
    )
    assert np.allclose(psi_coarse, psi_fine[:, :256], atol=1e-6)


def test_aperture_zeroes_modes_beyond_the_truncation_index() -> None:
    suite = make_suite()
    net = calibrated_net()
    assert net.effective_aperture == 256
    batch = suite.sample(4, 0.05, np.random.default_rng(4), n_modes=1024)
    sv = suite.operator.restrict(1024).singular_values
    psi = net.filter_profile(batch.noisy_data, sv)
    assert np.all(psi[:, 256:] == 0.0)
    assert np.any(psi[:, :256] > 0.0)

    net.set_aperture(None)
    assert net.effective_aperture is None
    assert np.any(net.filter_profile(batch.noisy_data, sv)[:, 256:] != 0.0)


def test_feature_clamping_keeps_the_network_inside_its_calibrated_domain() -> None:
    """The guarantee clamping provides, stated exactly.

    Clamping does not by itself force the damping to zero on unseen modes -- that is what
    the aperture is for.  What it does guarantee is that the MLP is never *evaluated*
    outside the box of standardised feature values recorded at calibration, so the filter
    on a finer grid is an interpolation of learned behaviour rather than an extrapolation.
    """
    suite = make_suite()
    net = calibrated_net(SCNetConfig(clamp_features=True))
    batch = suite.sample(8, 0.05, np.random.default_rng(5), n_modes=2048)
    sv = torch.as_tensor(
        suite.operator.restrict(2048).singular_values, dtype=torch.float32
    )
    raw = net.raw_features(torch.as_tensor(batch.noisy_data, dtype=torch.float32), sv)
    standardised = net._standardise(raw)
    assert torch.all(standardised >= net.feature_lo - 1e-5)
    assert torch.all(standardised <= net.feature_hi + 1e-5)

    # Without clamping the same features leave the calibrated box.
    loose = calibrated_net(SCNetConfig(clamp_features=False))
    unclamped = loose._standardise(
        loose.raw_features(torch.as_tensor(batch.noisy_data, dtype=torch.float32), sv)
    )
    assert torch.any(unclamped < loose.feature_lo - 1e-3)


def test_log_sigma_feature_saturates_beyond_the_training_spectrum() -> None:
    """The mode-index feature is the one that would otherwise extrapolate without bound."""
    suite = make_suite()
    net = calibrated_net(SCNetConfig(clamp_features=True))
    batch = suite.sample(4, 0.05, np.random.default_rng(6), n_modes=2048)
    sv = torch.as_tensor(
        suite.operator.restrict(2048).singular_values, dtype=torch.float32
    )
    index = net.config.feature_names.index("log_sigma")
    standardised = net._standardise(
        net.raw_features(torch.as_tensor(batch.noisy_data, dtype=torch.float32), sv)
    )
    tail = standardised[..., 300:, index]
    assert torch.allclose(tail, tail.flatten()[0].expand_as(tail), atol=1e-5)


def test_features_are_resolution_independent() -> None:
    suite = make_suite()
    net = calibrated_net()
    coarse = suite.sample(8, 0.05, np.random.default_rng(6), n_modes=256)
    fine = suite.sample(8, 0.05, np.random.default_rng(6), n_modes=2048)
    f_coarse = net.raw_features(
        torch.as_tensor(coarse.noisy_data, dtype=torch.float32),
        torch.as_tensor(
            suite.operator.restrict(256).singular_values, dtype=torch.float32
        ),
    )
    f_fine = net.raw_features(
        torch.as_tensor(fine.noisy_data, dtype=torch.float32),
        torch.as_tensor(
            suite.operator.restrict(2048).singular_values, dtype=torch.float32
        ),
    )
    assert torch.allclose(f_coarse, f_fine[:, :256], atol=1e-5)


def test_noise_floor_feature_tracks_the_noise_level() -> None:
    """The model is never told delta, so this feature must carry it."""
    suite = make_suite()
    net = calibrated_net()
    index = net.config.feature_names.index("log_noise")
    values = []
    for delta in [1e-3, 1e-2, 1e-1]:
        batch = suite.sample(64, delta, np.random.default_rng(7), n_modes=256)
        features = net.raw_features(
            torch.as_tensor(batch.noisy_data, dtype=torch.float32),
            torch.as_tensor(
                suite.operator.restrict(256).singular_values, dtype=torch.float32
            ),
        )
        values.append(float(features[..., 0, index].mean()))
    assert values[0] < values[1] < values[2]
    # A decade in delta must move the feature by about a decade.
    assert values[1] - values[0] == pytest.approx(1.0, abs=0.15)
    assert values[2] - values[1] == pytest.approx(1.0, abs=0.15)


def test_paper_feature_set_uses_raw_inputs() -> None:
    net = calibrated_net(SCNetConfig(feature_set="paper"))
    assert net.config.feature_names == ("coeff", "sigma")
    assert net.config.n_features == 2


def test_disabling_the_noise_estimate_drops_those_features() -> None:
    config = SCNetConfig(use_noise_estimate=False)
    assert "log_noise" not in config.feature_names
    assert "log_snr" not in config.feature_names
    assert config.n_features == 4


def test_lipschitz_bound_is_finite_and_positive() -> None:
    """Assumption 1 needs a finite L_Psi; expose it so the theory is checkable."""
    net = calibrated_net()
    bound = net.lipschitz_bound()
    assert np.isfinite(bound)
    assert bound > 0.0


def test_gradients_flow_to_every_parameter() -> None:
    suite = make_suite()
    net = calibrated_net()
    batch = suite.sample(8, 0.05, np.random.default_rng(8), n_modes=256)
    sv = torch.as_tensor(
        suite.operator.restrict(256).singular_values, dtype=torch.float32
    )
    prediction = net(torch.as_tensor(batch.noisy_data, dtype=torch.float32), sv)
    prediction.pow(2).sum().backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_mode_count_mismatch_raises() -> None:
    net = calibrated_net()
    with pytest.raises(ValueError):
        net.filter_coefficients(torch.zeros(2, 128), torch.ones(64))


def test_invalid_configs_raise() -> None:
    with pytest.raises(ValueError):
        SCNetConfig(c_psi=0.5)
    with pytest.raises(ValueError):
        SCNetConfig(feature_set="nonsense")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SCNetConfig(noise_band=(64, 32))
    with pytest.raises(ValueError):
        SCNetConfig(aperture_modes=0)