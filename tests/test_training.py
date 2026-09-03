"""Training must actually learn a regulariser, and do so reproducibly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from specinv import (
    InverseProblemSuite,
    SCNet,
    SCNetConfig,
    SobolevPrior,
    SpatialCNN,
    SpatialCNNConfig,
    TrainConfig,
    evaluate_on_resolutions,
    oracle_tikhonov,
    power_law_operator,
    relative_errors,
    sobolev_loss,
    summarise_errors,
    train_scnet,
    train_spatial_cnn,
)


def make_suite(n_modes: int = 1024) -> InverseProblemSuite:
    return InverseProblemSuite(power_law_operator(n_modes, 1.5), SobolevPrior(1.5))


def short_config(**kwargs: object) -> TrainConfig:
    defaults: dict[str, object] = {
        "n_train": 256,
        "n_val": 128,
        "n_modes": 256,
        "epochs": 40,
        "batch_size": 64,
        "seed": 0,
        "gamma": 0.0,
    }
    defaults.update(kwargs)
    return TrainConfig(**defaults)  # type: ignore[arg-type]


def test_sobolev_loss_is_zero_for_a_perfect_fit() -> None:
    truth = torch.randn(4, 16)
    weights = torch.ones(16)
    assert sobolev_loss(truth, truth, weights).item() == pytest.approx(0.0)


def test_sobolev_loss_weights_high_modes_more() -> None:
    truth = torch.zeros(1, 4)
    weights = torch.tensor([1.0, 2.0, 4.0, 8.0])
    low = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    high = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    a = sobolev_loss(low, truth, weights, relative=False).item()
    b = sobolev_loss(high, truth, weights, relative=False).item()
    assert b == pytest.approx(8.0 * a)


def test_relative_loss_is_invariant_to_sample_scale() -> None:
    truth = torch.randn(3, 8)
    prediction = truth + 0.1 * torch.randn(3, 8)
    weights = torch.ones(8)
    a = sobolev_loss(prediction, truth, weights, relative=True)
    b = sobolev_loss(1e4 * prediction, 1e4 * truth, weights, relative=True)
    assert a.item() == pytest.approx(b.item(), rel=1e-4)


@pytest.mark.slow
def test_training_reduces_the_validation_error() -> None:
    suite = make_suite()
    net = SCNet(SCNetConfig(hidden_sizes=(32, 32)))
    history = train_scnet(net, suite, short_config(epochs=150, n_train=512))
    assert len(history.val_error) == 150
    assert history.best_val_error < history.val_error[0]
    # Validation mixes delta log-uniformly over [1e-3, 1e-1]; a trained filter lands near
    # 0.11 there, an untrained one above 0.5.
    assert history.best_val_error < 0.2


@pytest.mark.slow
def test_trained_model_beats_the_naive_inverse() -> None:
    suite = make_suite()
    net = SCNet(SCNetConfig(hidden_sizes=(32, 32)))
    train_scnet(net, suite, short_config())
    batch = suite.sample(128, 0.05, np.random.default_rng(5), n_modes=256)
    sv = suite.operator.restrict(256).singular_values
    learned = summarise_errors(
        net.reconstruct(batch.noisy_data, sv), batch.true_coefficients
    ).mean_relative
    naive = summarise_errors(
        batch.noisy_data / sv, batch.true_coefficients
    ).mean_relative
    assert learned < naive
    assert learned < 0.4


@pytest.mark.slow
def test_training_is_reproducible_given_the_seed() -> None:
    suite = make_suite()
    errors = []
    for _ in range(2):
        net = SCNet(SCNetConfig(hidden_sizes=(16, 16)))
        train_scnet(net, suite, short_config(epochs=15))
        batch = suite.sample(32, 0.05, np.random.default_rng(0), n_modes=256)
        sv = suite.operator.restrict(256).singular_values
        errors.append(net.reconstruct(batch.noisy_data, sv))
    assert np.allclose(errors[0], errors[1])


@pytest.mark.slow
def test_evaluate_on_resolutions_returns_per_sample_errors() -> None:
    suite = make_suite(2048)
    net = SCNet(SCNetConfig(hidden_sizes=(16, 16)))
    train_scnet(net, suite, short_config(epochs=15))
    out = evaluate_on_resolutions(net, suite, [256, 512], delta=0.05, n_samples=16)
    assert set(out) == {256, 512}
    for value in out.values():
        assert value.shape == (16,)
        assert np.all(value > 0.0)


@pytest.mark.slow
def test_spatial_cnn_trains_and_runs_at_other_resolutions() -> None:
    """The control must be a working model, otherwise the comparison is unfair."""
    from specinv import SineBasis

    suite = make_suite(512)
    cnn = SpatialCNN(SpatialCNNConfig(channels=12, depth=2, dilations=(1, 2)))
    history = train_spatial_cnn(cnn, suite, short_config(epochs=25))
    assert history.best_val_error < 1.0

    batch = suite.sample(8, 0.05, np.random.default_rng(1), n_modes=256)
    prediction = cnn.reconstruct_coefficients(batch.noisy_data, SineBasis(256))
    assert prediction.shape == batch.true_coefficients.shape
    assert np.all(np.isfinite(prediction))

    finer = suite.sample(8, 0.05, np.random.default_rng(1), n_modes=512)
    prediction = cnn.reconstruct_coefficients(finer.noisy_data, SineBasis(512))
    assert prediction.shape == finer.true_coefficients.shape


@pytest.mark.slow
def test_scnet_approaches_the_oracle_tikhonov_baseline() -> None:
    """A cheap guard that training is not silently broken.

    The headline claim (strictly beating Oracle Tikhonov) needs the full training budget;
    here we only require getting within 15% of it after 60 epochs.
    """
    suite = make_suite()
    net = SCNet(SCNetConfig())
    train_scnet(net, suite, short_config(epochs=60, n_train=512))
    batch = suite.sample(256, 0.05, np.random.default_rng(9), n_modes=256)
    sv = suite.operator.restrict(256).singular_values
    learned = np.mean(
        relative_errors(net.reconstruct(batch.noisy_data, sv), batch.true_coefficients)
    )
    oracle = np.mean(
        relative_errors(
            oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients).reconstruction,
            batch.true_coefficients,
        )
    )
    assert learned < 1.15 * oracle


def test_invalid_train_configs() -> None:
    with pytest.raises(ValueError):
        TrainConfig(delta_range=(0.1, 0.01))
    with pytest.raises(ValueError):
        TrainConfig(n_train=0)
    with pytest.raises(ValueError):
        TrainConfig(gamma=-1.0)