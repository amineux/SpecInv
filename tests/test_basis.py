"""The sine basis must be an exact isometry, since every reported norm depends on it."""

from __future__ import annotations

import numpy as np
import pytest

from specinv import SineBasis


@pytest.mark.parametrize("n_grid", [7, 32, 256, 1024])
def test_analyze_synthesize_roundtrip(n_grid: int) -> None:
    rng = np.random.default_rng(0)
    values = rng.standard_normal((5, n_grid))
    basis = SineBasis(n_grid)
    assert np.allclose(basis.synthesize(basis.analyze(values)), values, atol=1e-12)


@pytest.mark.parametrize("n_grid", [16, 128, 512])
def test_parseval(n_grid: int) -> None:
    """Coefficient Euclidean norm equals the quadrature L2 norm of the grid function."""
    rng = np.random.default_rng(1)
    values = rng.standard_normal(n_grid)
    basis = SineBasis(n_grid)
    discrete_l2 = np.sqrt(basis.spacing * np.sum(values**2))
    assert basis.l2_norm(basis.analyze(values)) == pytest.approx(discrete_l2, rel=1e-12)


def test_basis_is_orthonormal() -> None:
    basis = SineBasis(64)
    modes = basis.evaluate_modes(basis.grid)
    gram = basis.spacing * (modes @ modes.T)
    assert np.allclose(gram, np.eye(64), atol=1e-12)


def test_analyze_recovers_known_coefficients() -> None:
    """A pure mode must produce a single non-zero coefficient of the right size."""
    basis = SineBasis(128)
    for n, amplitude in [(1, 0.7), (5, -2.0), (37, 0.25)]:
        values = amplitude * np.sqrt(2.0) * np.sin(n * np.pi * basis.grid)
        coefficients = basis.analyze(values)
        expected = np.zeros(128)
        expected[n - 1] = amplitude
        assert np.allclose(coefficients, expected, atol=1e-12)


def test_laplacian_eigenvalues_give_the_h1_seminorm() -> None:
    """The Sobolev loss uses sum_n lambda_n |f_n|^2 = ||grad f||^2 with lambda_n=(n pi)^2.

    Checked against direct quadrature of ``(f')^2`` for a smooth band-limited function.
    """
    basis = SineBasis(4096)
    coefficients = np.zeros(4096)
    for n, amplitude in [(1, 0.8), (4, -0.5), (9, 0.3)]:
        coefficients[n - 1] = amplitude

    spectral = float(np.sum(basis.laplacian_eigenvalues() * coefficients**2))

    x = basis.grid
    derivative = sum(
        amplitude * np.sqrt(2.0) * n * np.pi * np.cos(n * np.pi * x)
        for n, amplitude in [(1, 0.8), (4, -0.5), (9, 0.3)]
    )
    quadrature = float(basis.spacing * np.sum(derivative**2))
    assert spectral == pytest.approx(quadrature, rel=1e-3)


def test_laplacian_eigenvalues_are_analytic() -> None:
    basis = SineBasis(8)
    expected = (np.arange(1, 9) * np.pi) ** 2
    assert np.allclose(basis.laplacian_eigenvalues(), expected)


def test_coefficients_are_resolution_independent() -> None:
    """The same continuum function has the same low-order coefficients on any grid.

    This is the property the zero-shot experiment rests on: refining the grid must extend
    the coefficient vector rather than change it.
    """

    def f(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * 0.4 + np.sin(3 * np.pi * x) * 0.9 - np.sin(7 * np.pi * x)

    reference = SineBasis(256).analyze(f(SineBasis(256).grid))[:16]
    for n_grid in [512, 1024, 2048]:
        basis = SineBasis(n_grid)
        assert np.allclose(basis.analyze(f(basis.grid))[:16], reference, atol=1e-12)


def test_sobolev_norm_increases_with_order() -> None:
    rng = np.random.default_rng(3)
    basis = SineBasis(64)
    coefficients = rng.standard_normal(64)
    norms = [basis.sobolev_norm(coefficients, order=r) for r in (0.0, 0.5, 1.0)]
    assert norms[0] < norms[1] < norms[2]


def test_invalid_shapes_raise() -> None:
    basis = SineBasis(32)
    with pytest.raises(ValueError):
        basis.analyze(np.zeros(31))
    with pytest.raises(ValueError):
        basis.synthesize(np.zeros(33))
    with pytest.raises(ValueError):
        SineBasis(0)