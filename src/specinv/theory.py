"""Closed-form predictions from §4 of arXiv:2603.20602, for comparison with measurements.

Everything here is a one-liner; the point is to have the paper's claims written down as
executable predictions so the experiments can assert against them instead of against
hard-coded numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RatePrediction",
    "deterministic_rate",
    "statistical_rate",
    "optimal_truncation_index",
    "predicted_error_terms",
]

FloatArray = NDArray[np.float64]


def deterministic_rate(smoothness: float, ill_posedness: float) -> float:
    """Theorem 4.6: :math:`s/(s+p)`.  Equals 0.5 for :math:`s=p=1.5`."""
    _check(smoothness, ill_posedness)
    return smoothness / (smoothness + ill_posedness)


def statistical_rate(smoothness: float, ill_posedness: float) -> float:
    """Order attainable under isotropic (white) noise: :math:`s/(s+p+1/2)`.

    Not stated in the paper.  It arises because spreading a fixed noise energy over all
    :math:`M` modes turns the stability term from :math:`\\delta\\,\\sigma_N^{-1}` into
    :math:`\\delta\\,\\sigma_N^{-1}N^{-1/2}\\cdot N^{1/2}`-style behaviour -- concretely
    :math:`\\nu N^{p+1/2}` -- which shifts the balance.  For :math:`s=p=1.5` it is
    :math:`3/7 \\approx 0.4286`, which coincides with the Tikhonov slope of 0.42 reported
    in §5.2.
    """
    _check(smoothness, ill_posedness)
    return smoothness / (smoothness + ill_posedness + 0.5)


def optimal_truncation_index(
    delta: float | FloatArray, smoothness: float, ill_posedness: float
) -> FloatArray:
    """Theorem 4.6's prescription :math:`N \\asymp \\delta^{-1/(s+p)}`.

    Obtained from :math:`\\sigma_N \\asymp \\delta^{p/(s+p)}` with
    :math:`\\sigma_n \\asymp n^{-p}`.
    """
    _check(smoothness, ill_posedness)
    d = np.asarray(delta, dtype=np.float64)
    if np.any(d <= 0.0):
        raise ValueError("delta must be positive")
    return d ** (-1.0 / (smoothness + ill_posedness))


@dataclass(frozen=True)
class RatePrediction:
    """The two error terms of Theorem 4.6 and their balance."""

    stability: float
    truncation: float

    @property
    def total(self) -> float:
        """Root-sum-of-squares of the two terms."""
        return float(np.hypot(self.stability, self.truncation))


def predicted_error_terms(
    delta: float,
    truncation_index: int,
    smoothness: float,
    ill_posedness: float,
) -> RatePrediction:
    """Scaling of the stability term :math:`E_1` and the truncation term :math:`E_3`.

    Up to constants, :math:`E_1 \\asymp \\delta\\,N^{p}` (Theorem 4.1 with
    :math:`\\sigma_N \\asymp N^{-p}`) and :math:`E_3 \\asymp N^{-s}` (the :math:`H^s`
    approximation rate implied by :math:`|f_n| \\sim n^{-(s+1/2)}`).
    """
    _check(smoothness, ill_posedness)
    if truncation_index < 1:
        raise ValueError("truncation_index must be >= 1")
    n = float(truncation_index)
    return RatePrediction(
        stability=float(delta * n**ill_posedness),
        truncation=float(n**-smoothness),
    )


def _check(smoothness: float, ill_posedness: float) -> None:
    if smoothness <= 0.0:
        raise ValueError("smoothness (s) must be positive")
    if ill_posedness <= 0.0:
        raise ValueError("ill_posedness (p) must be positive")