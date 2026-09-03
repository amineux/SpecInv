"""Error metrics and convergence-rate estimation.

Metric definition
-----------------
The paper reports "relative :math:`L^2` errors" without specifying how the per-sample
errors are pooled, and the two natural choices differ by a few percent.  We compute both
and report the first as the headline number:

``mean_relative``
    :math:`\\frac1M \\sum_i \\|\\hat f_i - f_i\\| / \\|f_i\\|` -- the mean of the per-sample
    relative error.  Reported everywhere as *the* relative :math:`L^2` error.

``aggregate_relative``
    :math:`\\big(\\sum_i \\|\\hat f_i - f_i\\|^2 / \\sum_i \\|f_i\\|^2\\big)^{1/2}` -- the
    norm-pooled ("dataset level") relative error.

Norms are Euclidean norms of coefficient vectors, which by Parseval are exactly
:math:`L^2(\\Omega)` norms and are therefore independent of the grid (see
:mod:`specinv.basis`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["ErrorSummary", "relative_errors", "summarise_errors", "RateFit", "fit_rate"]

FloatArray = NDArray[np.float64]


def relative_errors(reconstruction: FloatArray, truth: FloatArray) -> FloatArray:
    """Per-sample relative :math:`L^2` error, shape ``(n_samples,)``."""
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if reconstruction.shape != truth.shape:
        raise ValueError(
            f"shape mismatch: {reconstruction.shape} vs {truth.shape}"
        )
    num = np.linalg.norm(reconstruction - truth, axis=-1)
    den = np.linalg.norm(truth, axis=-1)
    return num / np.maximum(den, 1e-300)


@dataclass(frozen=True)
class ErrorSummary:
    """Pooled error statistics for one method at one setting."""

    mean_relative: float
    aggregate_relative: float
    median_relative: float
    std_relative: float
    stderr_relative: float
    n_samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_relative_l2": self.mean_relative,
            "aggregate_relative_l2": self.aggregate_relative,
            "median_relative_l2": self.median_relative,
            "std_relative_l2": self.std_relative,
            "stderr_relative_l2": self.stderr_relative,
            "n_samples": self.n_samples,
        }


def summarise_errors(reconstruction: FloatArray, truth: FloatArray) -> ErrorSummary:
    """Pool per-sample errors into an :class:`ErrorSummary`."""
    per_sample = relative_errors(reconstruction, truth)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    aggregate = float(
        np.sqrt(
            np.sum((reconstruction - truth) ** 2) / max(float(np.sum(truth**2)), 1e-300)
        )
    )
    n = int(per_sample.size)
    std = float(np.std(per_sample, ddof=1)) if n > 1 else 0.0
    return ErrorSummary(
        mean_relative=float(np.mean(per_sample)),
        aggregate_relative=aggregate,
        median_relative=float(np.median(per_sample)),
        std_relative=std,
        stderr_relative=std / np.sqrt(n) if n > 1 else 0.0,
        n_samples=n,
    )


@dataclass(frozen=True)
class RateFit:
    """Least-squares fit of :math:`\\log e = \\text{slope}\\cdot\\log\\delta + \\text{intercept}`.

    Attributes
    ----------
    slope:
        The estimated convergence order.
    stderr:
        Standard error of the slope from the linear fit's residuals.
    ci95:
        Normal-approximation 95% confidence interval for the slope.
    prefactor:
        :math:`e^{\\text{intercept}}`, i.e. the constant :math:`C` in
        :math:`e \\approx C\\delta^{\\text{slope}}`.
    r_squared:
        Coefficient of determination of the log-log fit.
    """

    slope: float
    intercept: float
    stderr: float
    ci95: tuple[float, float]
    prefactor: float
    r_squared: float

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "stderr": self.stderr,
            "ci95": list(self.ci95),
            "prefactor": self.prefactor,
            "r_squared": self.r_squared,
        }


def fit_rate(deltas: FloatArray, errors: FloatArray) -> RateFit:
    """Estimate the convergence order from a noise-level sweep.

    Parameters
    ----------
    deltas, errors:
        Matching one-dimensional arrays of noise levels and errors; both must be positive.
    """
    d = np.asarray(deltas, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if d.shape != e.shape or d.ndim != 1:
        raise ValueError("deltas and errors must be 1-D arrays of the same length")
    if d.size < 3:
        raise ValueError("need at least 3 points to fit a rate with an error estimate")
    if np.any(d <= 0.0) or np.any(e <= 0.0):
        raise ValueError("deltas and errors must be strictly positive")

    x = np.log(d)
    y = np.log(e)
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, residuals, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(coefficients[0]), float(coefficients[1])

    fitted = design @ coefficients
    dof = x.size - 2
    residual_ss = float(np.sum((y - fitted) ** 2))
    sigma2 = residual_ss / dof if dof > 0 else 0.0
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    stderr = float(np.sqrt(max(covariance[0, 0], 0.0)))

    total_ss = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else 1.0

    return RateFit(
        slope=slope,
        intercept=intercept,
        stderr=stderr,
        ci95=(slope - 1.96 * stderr, slope + 1.96 * stderr),
        prefactor=float(np.exp(intercept)),
        r_squared=r_squared,
    )