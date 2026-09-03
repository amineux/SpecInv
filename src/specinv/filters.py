"""Classical spectral filters and per-sample oracle parameter selection.

§2.3 of arXiv:2603.20602 writes every classical regulariser as a *filter function*
:math:`g_\\alpha(\\sigma)` acting on the singular values,

.. math:: f_\\alpha^\\delta = \\sum_n g_\\alpha(\\sigma_n)\\,\\langle y^\\delta, u_n\\rangle\\, v_n .

Throughout :mod:`specinv` we use the equivalent *damping* form
:math:`\\lambda_n = \\sigma_n g_\\alpha(\\sigma_n) \\in [0,1]`, so that

.. math:: \\hat f_n = \\lambda_n \\cdot \\frac{y_n}{\\sigma_n},

which is exactly the shape of the SC-Net reconstruction Eq. (5) with
:math:`\\lambda_n = \\Psi_\\theta(y_n, \\sigma_n)`.  Putting the learned and the classical
filters in the same units is what makes the comparison of §5.3 possible: the plots overlay
:math:`\\lambda_n` for SC-Net, Tikhonov and TSVD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "OracleSelection",
    "discrepancy_principle_tikhonov",
    "landweber_damping",
    "oracle_spectral_bound",
    "oracle_tikhonov",
    "oracle_tsvd",
    "prior_wiener",
    "tikhonov_damping",
    "tsvd_damping",
    "wiener_damping",
]

FloatArray = NDArray[np.float64]


def tikhonov_damping(singular_values: FloatArray, alpha: float) -> FloatArray:
    """Tikhonov damping :math:`\\lambda_n = \\sigma_n^2/(\\sigma_n^2+\\alpha)`.

    Equivalent to the paper's :math:`g_\\alpha(\\sigma) = \\sigma/(\\sigma^2+\\alpha)`.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    s2 = np.asarray(singular_values, dtype=np.float64) ** 2
    return s2 / (s2 + alpha)


def tsvd_damping(singular_values: FloatArray, alpha: float) -> FloatArray:
    """Truncated-SVD damping: 1 where :math:`\\sigma_n \\ge \\sqrt\\alpha`, else 0."""
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    sv = np.asarray(singular_values, dtype=np.float64)
    return (sv >= np.sqrt(alpha)).astype(np.float64)


def landweber_damping(
    singular_values: FloatArray, n_iterations: int, step: float = 1.0
) -> FloatArray:
    """Landweber damping :math:`1-(1-\\tau\\sigma_n^2)^k`."""
    if n_iterations < 0:
        raise ValueError("n_iterations must be non-negative")
    sv = np.asarray(singular_values, dtype=np.float64)
    if step <= 0.0 or step > 1.0 / float(sv.max() ** 2):
        raise ValueError("step must satisfy 0 < step <= 1/sigma_1**2 for convergence")
    return 1.0 - (1.0 - step * sv**2) ** n_iterations


def wiener_damping(
    singular_values: FloatArray, signal_scale: FloatArray, noise_scale: FloatArray
) -> FloatArray:
    """Wiener/oracle damping of Definition 4.3, generalised to a per-mode noise level.

    .. math::
        \\lambda_n = \\frac{\\tau_n^2}{\\tau_n^2 + \\omega_n^2},
        \\qquad \\omega_n = \\nu_n/\\sigma_n,

    where :math:`\\tau_n` is the prior standard deviation of :math:`f_n` and :math:`\\nu_n`
    the noise standard deviation of :math:`y_n`.  Definition 4.3 is the special case of a
    flat prior and a flat noise level.  For a Gaussian prior this is the *pointwise
    Bayes-optimal* filter, hence a ceiling for any filter that acts mode-by-mode.
    """
    tau2 = np.asarray(signal_scale, dtype=np.float64) ** 2
    omega2 = (np.asarray(noise_scale, dtype=np.float64) / singular_values) ** 2
    return tau2 / (tau2 + omega2)


@dataclass(frozen=True)
class OracleSelection:
    """Result of a per-sample oracle parameter search.

    Attributes
    ----------
    reconstruction:
        Coefficients of the reconstruction, shape ``(n_samples, n_modes)``.
    damping:
        The selected damping profiles :math:`\\lambda_n`, same shape.
    parameter:
        The selected parameter per sample (``alpha`` for Tikhonov, truncation index for
        TSVD), shape ``(n_samples,)``.
    """

    reconstruction: FloatArray
    damping: FloatArray
    parameter: FloatArray


def _naive(noisy_data: FloatArray, singular_values: FloatArray) -> FloatArray:
    return np.asarray(noisy_data, dtype=np.float64) / singular_values


def oracle_tikhonov(
    noisy_data: FloatArray,
    singular_values: FloatArray,
    true_coefficients: FloatArray,
    alphas: FloatArray | None = None,
) -> OracleSelection:
    """Tikhonov with :math:`\\alpha` chosen per sample to minimise the true error.

    This is the "Oracle Tikhonov" benchmark of §5.1: it is given access to the ground
    truth, so it is an upper bound on what any practical parameter-choice rule (L-curve,
    discrepancy principle) could achieve within the Tikhonov family.
    """
    if alphas is None:
        alphas = np.geomspace(1e-18, 1e2, 361)
    naive = _naive(noisy_data, singular_values)
    truth = np.asarray(true_coefficients, dtype=np.float64)

    best_err = np.full(truth.shape[0], np.inf)
    best_rec = np.zeros_like(truth)
    best_damp = np.zeros_like(truth)
    best_par = np.zeros(truth.shape[0])
    for alpha in np.asarray(alphas, dtype=np.float64):
        damping = tikhonov_damping(singular_values, float(alpha))
        rec = damping * naive
        err = np.linalg.norm(rec - truth, axis=-1)
        improved = err < best_err
        if np.any(improved):
            best_err = np.where(improved, err, best_err)
            best_rec[improved] = rec[improved]
            best_damp[improved] = damping
            best_par[improved] = alpha
    return OracleSelection(best_rec, best_damp, best_par)


def oracle_tsvd(
    noisy_data: FloatArray,
    singular_values: FloatArray,
    true_coefficients: FloatArray,
) -> OracleSelection:
    """TSVD with the truncation index chosen per sample to minimise the true error.

    Computed exactly (not by grid search) with prefix sums: for a cut after mode ``k`` the
    squared error is ``sum_{n<=k} (naive_n - f_n)^2 + sum_{n>k} f_n^2``.
    """
    naive = _naive(noisy_data, singular_values)
    truth = np.asarray(true_coefficients, dtype=np.float64)

    inside = np.cumsum((naive - truth) ** 2, axis=-1)
    tail_total = np.sum(truth**2, axis=-1, keepdims=True)
    tail_inclusive = np.cumsum(truth**2, axis=-1)
    outside = tail_total - tail_inclusive
    total = inside + outside  # index k-1 <-> keeping modes 1..k

    # Keeping zero modes is also admissible.
    total = np.concatenate([tail_total, total], axis=-1)
    kept = np.argmin(total, axis=-1)  # number of retained modes

    n_modes = truth.shape[-1]
    mode_index = np.arange(1, n_modes + 1)
    damping = (mode_index[None, :] <= kept[:, None]).astype(np.float64)
    return OracleSelection(damping * naive, damping, kept.astype(np.float64))


def oracle_spectral_bound(
    noisy_data: FloatArray,
    singular_values: FloatArray,
    true_coefficients: FloatArray,
) -> OracleSelection:
    """The best *any* diagonal filter can do on each realisation.

    Minimises :math:`|\\lambda_n y_n/\\sigma_n - f_n|^2` over :math:`\\lambda_n \\in [0,1]`
    independently per mode, giving :math:`\\lambda_n = \\mathrm{clip}(f_n \\sigma_n/y_n, 0, 1)`.
    Not attainable by any method without the ground truth; reported as an unreachable floor
    so the gap left by SC-Net is visible.
    """
    naive = _naive(noisy_data, singular_values)
    truth = np.asarray(true_coefficients, dtype=np.float64)
    safe = np.where(np.abs(naive) < 1e-300, 1e-300, naive)
    damping = np.clip(truth / safe, 0.0, 1.0)
    return OracleSelection(damping * naive, damping, np.zeros(truth.shape[0]))


def prior_wiener(
    noisy_data: FloatArray,
    singular_values: FloatArray,
    signal_scale: FloatArray,
    noise_scale: FloatArray,
) -> OracleSelection:
    """Wiener filter built from the *prior* and the true noise level (no ground truth).

    For the Gaussian suite this is the pointwise Bayes-optimal spectral filter and hence
    the ceiling that SC-Net can be expected to approach.  It is not an oracle in the sense
    of the other baselines -- it never sees :math:`f` -- but it does receive the exact
    :math:`\\delta`, which SC-Net has to infer from the data.
    """
    naive = _naive(noisy_data, singular_values)
    damping = wiener_damping(singular_values, signal_scale, noise_scale)
    damping = np.broadcast_to(damping, naive.shape)
    return OracleSelection(
        damping * naive, np.array(damping), np.zeros(naive.shape[0])
    )


def discrepancy_principle_tikhonov(
    noisy_data: FloatArray,
    singular_values: FloatArray,
    noise_norm: float | FloatArray,
    tau: float = 1.1,
    n_bisection: int = 80,
) -> OracleSelection:
    """Tikhonov with :math:`\\alpha` from Morozov's discrepancy principle.

    Chooses :math:`\\alpha` so that
    :math:`\\|\\mathcal{K}\\mathcal{R}_\\alpha y^\\delta - y^\\delta\\| = \\tau\\delta`,
    the practical rule §4.4 points to.  Unlike :func:`oracle_tikhonov` this never sees the
    ground truth -- it only needs the noise level -- so it is the honest reference for what
    a practitioner would actually get out of Tikhonov, and the gap between the two
    quantifies how much the oracle benchmark is worth.

    Parameters
    ----------
    noise_norm:
        The *absolute* :math:`L^2` noise norm :math:`\\delta = \\|y^\\delta - \\mathcal{K}f\\|`,
        scalar or one value per sample.  This is the :math:`\\delta` of Theorem 4.6, not the
        per-mode relative level used to parameterise
        :class:`~specinv.problems.NoiseModel`; under a coloured model the two differ by a
        factor that depends on the resolution, and passing the wrong one badly
        under-regularises.
    tau:
        Safety factor :math:`\\tau > 1` of §4.4.

    Notes
    -----
    The residual is monotone in :math:`\\alpha` for a diagonal operator, so bisection on
    :math:`\\log\\alpha` converges to the unique root.
    """
    if np.any(np.asarray(noise_norm) <= 0.0):
        raise ValueError("noise_norm must be positive for the discrepancy principle")
    if tau <= 1.0:
        raise ValueError("tau must exceed 1")
    y = np.asarray(noisy_data, dtype=np.float64)
    sv = np.asarray(singular_values, dtype=np.float64)
    target = tau * np.broadcast_to(
        np.asarray(noise_norm, dtype=np.float64).reshape(-1), (y.shape[0],)
    )

    lo = np.full(y.shape[0], -40.0)
    hi = np.full(y.shape[0], 10.0)
    for _ in range(n_bisection):
        mid = 0.5 * (lo + hi)
        alpha = np.exp(mid)[:, None]
        residual = np.linalg.norm((alpha / (sv**2 + alpha)) * y, axis=-1)
        too_big = residual > target
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)

    alpha = np.exp(0.5 * (lo + hi))[:, None]
    damping = sv**2 / (sv**2 + alpha)
    return OracleSelection(damping * (y / sv), damping, alpha[:, 0])
