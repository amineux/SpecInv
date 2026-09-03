"""The diagonal operator must agree with a genuine Fredholm integral operator."""

from __future__ import annotations

import numpy as np
import pytest

from specinv import (
    DiagonalSpectralOperator,
    SineBasis,
    dense_kernel_matrix,
    power_law_operator,
)


def test_power_law_spectrum() -> None:
    operator = power_law_operator(16, ill_posedness=1.5)
    expected = np.arange(1, 17, dtype=float) ** -1.5
    assert np.allclose(operator.singular_values, expected)
    assert operator.ill_posedness == 1.5
    assert operator.n_modes == 16


def test_apply_then_naive_inverse_is_identity() -> None:
    rng = np.random.default_rng(0)
    operator = power_law_operator(64, 1.5)
    coefficients = rng.standard_normal((4, 64))
    recovered = operator.naive_inverse(operator.apply(coefficients))
    assert np.allclose(recovered, coefficients)


def test_diagonal_operator_matches_integral_kernel() -> None:
    """The diagonal representation is a real integral operator, not a modelling shortcut.

    Builds the kernel k(x,x') = sum_n sigma_n v_n(x) v_n(x') and checks that quadrature
    against it reproduces the coefficient-space forward map.
    """
    basis = SineBasis(256)
    operator = power_law_operator(256, 1.5)
    kernel = dense_kernel_matrix(basis, operator)

    rng = np.random.default_rng(2)
    coefficients = rng.standard_normal(256) * np.arange(1, 257, dtype=float) ** -2.0
    f_grid = basis.synthesize(coefficients)

    spatial = kernel @ f_grid
    spectral = basis.synthesize(operator.apply(coefficients))
    assert np.allclose(spatial, spectral, atol=1e-10)


def test_kernel_is_symmetric_and_square_integrable() -> None:
    basis = SineBasis(128)
    operator = power_law_operator(128, 1.5)
    kernel = dense_kernel_matrix(basis, operator)
    assert np.allclose(kernel, kernel.T, atol=1e-12)
    assert np.isfinite(np.sum(kernel**2))


def test_restrict() -> None:
    operator = power_law_operator(128, 1.5)
    restricted = operator.restrict(32)
    assert restricted.n_modes == 32
    assert np.allclose(restricted.singular_values, operator.singular_values[:32])
    with pytest.raises(ValueError):
        operator.restrict(0)
    with pytest.raises(ValueError):
        operator.restrict(129)


def test_optimal_truncation_index_follows_theory() -> None:
    """N ~ delta^{-1/(s+p)}: halving delta must widen the retained band."""
    operator = power_law_operator(4096, 1.5)
    previous = 0
    for delta in [1e-1, 1e-2, 1e-3, 1e-4]:
        index = operator.optimal_truncation_index(delta, smoothness=1.5)
        assert index > previous
        previous = index
    # Check the exponent to within one mode over two decades.
    n1 = operator.optimal_truncation_index(1e-2, 1.5)
    n2 = operator.optimal_truncation_index(1e-4, 1.5)
    assert n2 / n1 == pytest.approx(100 ** (1 / 3), rel=0.15)


def test_rejects_invalid_spectra() -> None:
    with pytest.raises(ValueError):
        DiagonalSpectralOperator(np.array([1.0, 0.0]), 1.5)
    with pytest.raises(ValueError):
        DiagonalSpectralOperator(np.array([0.1, 1.0]), 1.5)  # increasing
    with pytest.raises(ValueError):
        power_law_operator(8, ill_posedness=-1.0)


def test_mode_count_mismatch_raises() -> None:
    operator = power_law_operator(8, 1.5)
    with pytest.raises(ValueError):
        operator.apply(np.zeros(9))