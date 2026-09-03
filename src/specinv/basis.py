"""Orthonormal sine (Dirichlet) basis on the unit interval.

The suite of §5.1 of arXiv:2603.20602 uses a 1D Fredholm integral operator that is
diagonalised in a Fourier basis.  We use the Dirichlet sine system on :math:`\\Omega=(0,1)`,

.. math::
    v_n(x) = \\sqrt{2}\\,\\sin(n\\pi x), \\qquad n = 1, 2, \\dots

which is orthonormal in :math:`L^2(0,1)` and simultaneously diagonalises the Dirichlet
Laplacian, :math:`-\\Delta v_n = (n\\pi)^2 v_n`.  That second property is what makes the
Sobolev-weighted loss of Eq. (7) computable in the spectral domain (§3.2, "Remark on
Computation").

Discretisation
--------------
A grid of ``n_grid`` interior points :math:`x_j = j h`, :math:`h = 1/(n_{\\mathrm{grid}}+1)`,
:math:`j = 1,\\dots,n_{\\mathrm{grid}}`, supports exactly ``n_grid`` sine modes.  The
discrete sine transform of type I is *exactly* orthogonal on this grid, so with the
quadrature weight :math:`\\sqrt{h}` folded in, the map

    grid samples :math:`f(x_j)` :math:`\\longleftrightarrow` coefficients
    :math:`\\langle f, v_n\\rangle_{L^2}`

is a linear isometry up to the quadrature error.  Consequently Parseval holds *discretely*:
the Euclidean norm of the coefficient vector equals the (trapezoid-free) :math:`L^2` norm of
the grid function.  Every norm reported by :mod:`specinv` is therefore an :math:`L^2` norm
that does not depend on the resolution -- the property the zero-shot experiment relies on.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dst, idst

__all__ = ["SineBasis"]

FloatArray = NDArray[np.float64]


class SineBasis:
    """Orthonormal Dirichlet sine basis on ``(0, 1)`` discretised on ``n_grid`` points.

    Parameters
    ----------
    n_grid:
        Number of interior grid points, which equals the number of representable modes.

    Notes
    -----
    ``analyze`` and ``synthesize`` are mutual inverses to machine precision, and
    ``analyze`` is an isometry from the :math:`L^2`-weighted grid space onto
    :math:`\\mathbb{R}^{n_\\mathrm{grid}}`.
    """

    def __init__(self, n_grid: int) -> None:
        if n_grid < 1:
            raise ValueError(f"n_grid must be >= 1, got {n_grid}")
        self._n_grid = int(n_grid)
        self._h = 1.0 / (self._n_grid + 1.0)

    @property
    def n_grid(self) -> int:
        """Number of interior grid points."""
        return self._n_grid

    @property
    def n_modes(self) -> int:
        """Number of representable sine modes (equal to :attr:`n_grid`)."""
        return self._n_grid

    @property
    def spacing(self) -> float:
        """Grid spacing :math:`h = 1/(n_\\mathrm{grid}+1)`."""
        return self._h

    @property
    def grid(self) -> FloatArray:
        """Interior grid points :math:`x_j = j h`, ``shape (n_grid,)``."""
        return np.arange(1, self._n_grid + 1, dtype=np.float64) * self._h

    @property
    def mode_indices(self) -> FloatArray:
        """Mode indices :math:`n = 1, \\dots, n_\\mathrm{modes}`, as floats."""
        return np.arange(1, self._n_modes_int + 1, dtype=np.float64)

    @property
    def _n_modes_int(self) -> int:
        return self._n_grid

    def laplacian_eigenvalues(self) -> FloatArray:
        """Dirichlet Laplacian eigenvalues :math:`\\lambda_n = (n\\pi)^2`."""
        return (self.mode_indices * np.pi) ** 2

    def analyze(self, values: FloatArray) -> FloatArray:
        """Map grid samples ``f(x_j)`` to coefficients :math:`\\langle f, v_n\\rangle`.

        Parameters
        ----------
        values:
            Array with trailing axis of length :attr:`n_grid`.

        Returns
        -------
        Coefficients with the same shape.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self._n_grid:
            raise ValueError(
                f"expected trailing axis {self._n_grid}, got {values.shape[-1]}"
            )
        return np.asarray(dst(values, type=1, norm="ortho", axis=-1)) * np.sqrt(self._h)

    def synthesize(self, coefficients: FloatArray) -> FloatArray:
        """Map coefficients back to grid samples ``f(x_j)``. Inverse of :meth:`analyze`."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.shape[-1] != self._n_modes_int:
            raise ValueError(
                f"expected trailing axis {self._n_modes_int}, got {coefficients.shape[-1]}"
            )
        return np.asarray(
            idst(coefficients / np.sqrt(self._h), type=1, norm="ortho", axis=-1)
        )

    def evaluate_modes(self, x: FloatArray) -> FloatArray:
        """Evaluate the first :attr:`n_modes` basis functions at arbitrary points.

        Returns an array of shape ``(n_modes, len(x))`` holding
        :math:`v_n(x) = \\sqrt2 \\sin(n\\pi x)`.  Used to resample a coefficient vector onto
        a grid different from the one it was produced on.
        """
        x = np.asarray(x, dtype=np.float64)
        return np.sqrt(2.0) * np.sin(np.outer(self.mode_indices, x) * np.pi)

    def l2_norm(self, coefficients: FloatArray) -> FloatArray:
        """:math:`L^2(\\Omega)` norm from coefficients (Parseval)."""
        return np.sqrt(np.sum(np.asarray(coefficients, dtype=np.float64) ** 2, axis=-1))

    def sobolev_norm(self, coefficients: FloatArray, order: float = 1.0) -> FloatArray:
        """:math:`H^{\\text{order}}` norm computed spectrally.

        Uses :math:`\\|f\\|_{H^r}^2 = \\sum_n (1 + \\lambda_n^{r}) |f_n|^2` with
        :math:`\\lambda_n = (n\\pi)^2`.
        """
        coefficients = np.asarray(coefficients, dtype=np.float64)
        weights = 1.0 + self.laplacian_eigenvalues() ** order
        return np.sqrt(np.sum(weights * coefficients**2, axis=-1))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"SineBasis(n_grid={self._n_grid})"