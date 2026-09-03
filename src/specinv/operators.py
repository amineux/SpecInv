"""Forward operators that are diagonal in the sine basis.

§5.1 of arXiv:2603.20602 specifies a 1D Fredholm integral equation of the first kind whose
operator is diagonalised in the Fourier basis with singular values

.. math:: \\sigma_n \\sim n^{-p}, \\qquad p = 1.5 .

Working with the diagonal representation directly is not a shortcut: the operator
:math:`\\mathcal{K} = (-\\Delta)^{-p/2}` on :math:`(0,1)` with Dirichlet boundary conditions
*is* a Fredholm integral operator with a symmetric, square-integrable kernel, and its
singular system is exactly :math:`\\{(n\\pi)^{-p}, v_n, v_n\\}`.  :func:`dense_kernel_matrix`
materialises the kernel so that the equivalence can be checked numerically (see
``tests/test_operators.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .basis import SineBasis

__all__ = [
    "DiagonalSpectralOperator",
    "power_law_operator",
    "dense_kernel_matrix",
]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DiagonalSpectralOperator:
    """A compact linear operator diagonal in the sine basis.

    Attributes
    ----------
    singular_values:
        Strictly positive, non-increasing array :math:`(\\sigma_n)_{n=1}^{N}`.
    ill_posedness:
        The decay exponent :math:`p` with :math:`\\sigma_n \\asymp n^{-p}`, kept alongside the
        values because the theory of §4 is stated in terms of :math:`p`.
    """

    singular_values: FloatArray
    ill_posedness: float

    def __post_init__(self) -> None:
        sv = np.asarray(self.singular_values, dtype=np.float64)
        if sv.ndim != 1:
            raise ValueError("singular_values must be one-dimensional")
        if sv.size == 0:
            raise ValueError("singular_values must be non-empty")
        if np.any(sv <= 0.0):
            raise ValueError("singular values must be strictly positive")
        if np.any(np.diff(sv) > 1e-12):
            raise ValueError("singular values must be non-increasing")
        object.__setattr__(self, "singular_values", sv)

    @property
    def n_modes(self) -> int:
        """Number of retained singular triplets."""
        return int(self.singular_values.size)

    def apply(self, coefficients: FloatArray) -> FloatArray:
        """Forward map :math:`\\mathcal{K}f`, acting on coefficient vectors."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        self._check(coefficients)
        return coefficients * self.singular_values

    def naive_inverse(self, data_coefficients: FloatArray) -> FloatArray:
        """Unregularised pseudo-inverse :math:`y_n/\\sigma_n` of Eq. (3)."""
        data_coefficients = np.asarray(data_coefficients, dtype=np.float64)
        self._check(data_coefficients)
        return data_coefficients / self.singular_values

    def restrict(self, n_modes: int) -> DiagonalSpectralOperator:
        """Restrict to the leading ``n_modes`` singular triplets."""
        if not 1 <= n_modes <= self.n_modes:
            raise ValueError(f"n_modes must lie in [1, {self.n_modes}], got {n_modes}")
        return DiagonalSpectralOperator(
            self.singular_values[:n_modes].copy(), self.ill_posedness
        )

    def optimal_truncation_index(self, delta: float, smoothness: float) -> int:
        """Theory-driven truncation index of Theorem 4.6.

        Theorem 4.6 prescribes :math:`\\sigma_N \\asymp \\delta^{p/(s+p)}`, i.e.
        :math:`N \\asymp \\delta^{-1/(s+p)}` for a power-law spectrum.
        """
        if delta <= 0.0:
            return self.n_modes
        p = self.ill_posedness
        target = delta ** (p / (smoothness + p))
        idx = int(np.searchsorted(-self.singular_values, -target, side="left"))
        return int(np.clip(idx, 1, self.n_modes))

    def _check(self, arr: FloatArray) -> None:
        if arr.shape[-1] != self.n_modes:
            raise ValueError(
                f"trailing axis must be {self.n_modes} (operator modes), got {arr.shape[-1]}"
            )


def power_law_operator(
    n_modes: int, ill_posedness: float = 1.5, scale: float = 1.0
) -> DiagonalSpectralOperator:
    """Build :math:`\\sigma_n = \\text{scale} \\cdot n^{-p}`, the spectrum used in §5.1.

    The default ``scale=1.0`` reproduces the paper's literal ``sigma_n = n**-p``
    (so :math:`\\sigma_1 = 1`).  Choosing ``scale = pi**-p`` instead gives the exact
    spectrum of :math:`(-\\Delta)^{-p/2}`; both are covered by the same code path because
    the reported error metric is relative and the noise level is specified relative to
    :math:`\\|y\\|`, which makes the results invariant to this normalisation.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if ill_posedness <= 0.0:
        raise ValueError("ill_posedness (p) must be positive")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    n = np.arange(1, n_modes + 1, dtype=np.float64)
    return DiagonalSpectralOperator(scale * n ** (-ill_posedness), ill_posedness)


def dense_kernel_matrix(basis: SineBasis, operator: DiagonalSpectralOperator) -> FloatArray:
    """Materialise the integral kernel :math:`k(x,x') = \\sum_n \\sigma_n v_n(x) v_n(x')`.

    Returned as the matrix :math:`h\\,k(x_i, x_j)` so that ``M @ f(x_j)`` approximates
    :math:`(\\mathcal{K}f)(x_i)`.  Only used for verification and illustration; the
    experiments operate in the (mathematically equivalent) diagonal representation.
    """
    if operator.n_modes > basis.n_modes:
        raise ValueError("operator has more modes than the basis can represent")
    modes = basis.evaluate_modes(basis.grid)[: operator.n_modes]
    kernel = (modes * operator.singular_values[:, None]).T @ modes
    return kernel * basis.spacing