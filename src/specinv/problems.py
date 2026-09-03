"""Signal priors, noise models and the 1D inverse-problem suite of §5.1.

Signal prior
------------
§5.1 of arXiv:2603.20602 states that the ground-truth sources are drawn from a Sobolev
space :math:`H^s` with *spectral coefficients decaying as* :math:`|f_n| \\sim n^{-(s+1/2)}`.
That convention is the self-consistent one: it gives a tail (truncation) error

.. math:: \\Big(\\sum_{n>N} |f_n|^2\\Big)^{1/2} \\asymp N^{-s},

which is precisely the :math:`H^s` approximation rate, and -- balanced against the
stability term :math:`\\delta/\\sigma_N \\asymp \\delta N^{p}` of Theorem 4.1 -- yields the
paper's headline rate :math:`\\delta^{s/(s+p)}`.  (Theorem 4.6 additionally writes a source
condition :math:`|f_n| \\le C n^{-(s+p)}`; carried through its own :math:`E_3` estimate that
exponent does *not* reproduce the stated rate, so we follow §5.1.  See the README section
"Discrepancies vs. the paper".)

Noise models
------------
Which noise model one uses decides which convergence order is observable, and the paper
does not state the convention.  We implement three, all parameterised by a *relative*
level :math:`\\delta` (so that :math:`\\delta = 0.05` means "5% noise"), and all defined by
their profile in the spectral domain of the operator:

``CRITICAL``
    Per-mode standard deviation :math:`\\delta \\|y\\| n^{-1/2}`.  This is the *critical*
    colouring: it is the random noise model whose induced error matches the deterministic
    worst-case bound :math:`\\delta/\\sigma_N` of Theorem 4.1, so the observable order is
    :math:`s/(s+p)` -- the paper's :math:`O(\\delta^{0.5})` for :math:`s=p=1.5`.  Its total
    energy grows only logarithmically in the resolution, i.e. it sits exactly on the
    borderline of :math:`L^2`, which is the analytical reason the deterministic bound is
    sharp for it.  **This is the default and the model used for the headline results.**

``WHITE_ENERGY``
    Isotropic noise normalised to :math:`\\|\\varepsilon\\| = \\delta\\|y\\|` exactly.  This is
    the faithful discretisation of a *finite-energy* white perturbation.  Because the fixed
    energy is spread over all :math:`M` modes, the stability term becomes
    :math:`\\asymp \\nu N^{p+1/2}` rather than :math:`\\nu N^{p}` and the observable order
    drops to the *statistical* rate :math:`s/(s+p+1/2)` (:math:`\\approx 0.43` for
    :math:`s=p=1.5`).

``WHITE_POINTWISE``
    Per-mode standard deviation :math:`\\delta\\|y\\|`, i.e. the common
    ``y + delta*y.std()*randn(...)`` convention.  Same observable order as
    ``WHITE_ENERGY``; only the constant differs.

The distinction is not cosmetic: under the white models the observable order for *every*
optimal spectral method -- including Oracle Tikhonov and Oracle TSVD -- is
:math:`\\approx 0.43`, which is exactly the Tikhonov slope of 0.42 reported in §5.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .operators import DiagonalSpectralOperator

__all__ = [
    "AmplitudeLaw",
    "NoiseModel",
    "SobolevPrior",
    "ProblemBatch",
    "InverseProblemSuite",
]

FloatArray = NDArray[np.float64]

AmplitudeLaw = Literal["gaussian", "heavy_tail"]


class NoiseModel(str, Enum):
    """Spectral profile of the measurement noise (see module docstring)."""

    CRITICAL = "critical"
    WHITE_ENERGY = "white_energy"
    WHITE_POINTWISE = "white_pointwise"

    @property
    def colouring_exponent(self) -> float:
        """Exponent :math:`q` in the per-mode standard deviation :math:`\\propto n^{-q}`."""
        return 0.5 if self is NoiseModel.CRITICAL else 0.0

    def observable_rate(self, smoothness: float, ill_posedness: float) -> float:
        """Convergence order in :math:`\\delta` that an optimal filter attains here.

        ``CRITICAL`` gives the deterministic order :math:`s/(s+p)` of Theorem 4.6;
        the white models give the statistical order :math:`s/(s+p+1/2)`.
        """
        if self is NoiseModel.CRITICAL:
            return smoothness / (smoothness + ill_posedness)
        return smoothness / (smoothness + ill_posedness + 0.5)


@dataclass(frozen=True)
class SobolevPrior:
    """Random :math:`H^s` sources with coefficients :math:`f_n = \\xi_n n^{-(s+1/2)}`.

    Attributes
    ----------
    smoothness:
        Sobolev regularity index :math:`s`.
    amplitude:
        Distribution of the dimensionless factors :math:`\\xi_n`.  ``"gaussian"`` is the
        paper's suite.  ``"heavy_tail"`` (Student-:math:`t`, 2 dof) is an extension used to
        probe *non-linear* adaptivity: under a Gaussian prior the pointwise Bayes-optimal
        filter is linear in the observation (the Wiener filter), so a filter that reads
        :math:`y_n` cannot beat it; under a heavy tail it can.
    decay_offset:
        The :math:`+1/2` in the exponent, exposed so that the alternative source-condition
        convention :math:`n^{-(s+p)}` can be selected for study.
    """

    smoothness: float = 1.5
    amplitude: AmplitudeLaw = "gaussian"
    decay_offset: float = 0.5

    def __post_init__(self) -> None:
        if self.smoothness <= 0.0:
            raise ValueError("smoothness (s) must be positive")
        if self.amplitude not in ("gaussian", "heavy_tail"):
            raise ValueError(f"unknown amplitude law {self.amplitude!r}")

    @property
    def decay_exponent(self) -> float:
        """Coefficient decay exponent :math:`a = s + \\text{decay\\_offset}`."""
        return self.smoothness + self.decay_offset

    def coefficient_scale(self, n_modes: int) -> FloatArray:
        """Per-mode standard deviation :math:`n^{-a}` of the prior."""
        n = np.arange(1, n_modes + 1, dtype=np.float64)
        return n ** (-self.decay_exponent)

    def sample(
        self,
        n_samples: int,
        n_modes: int,
        rng: np.random.Generator,
        draw_modes: int | None = None,
    ) -> FloatArray:
        """Draw ``n_samples`` coefficient vectors of length ``n_modes``.

        Parameters
        ----------
        draw_modes:
            Draw this many random amplitudes per sample and return the leading
            ``n_modes``.  Passing a fixed ``draw_modes`` makes the result at a coarse
            resolution an exact *prefix* of the result at a finer one for the same seed,
            so that changing the resolution refines one continuum problem instead of
            substituting a different one.  Without it the row-major fill order of the
            generator would silently decorrelate the two.
        """
        if n_samples < 1 or n_modes < 1:
            raise ValueError("n_samples and n_modes must be >= 1")
        width = n_modes if draw_modes is None else max(int(draw_modes), n_modes)
        if self.amplitude == "gaussian":
            xi = rng.standard_normal((n_samples, width))
        else:
            xi = rng.standard_t(2.0, size=(n_samples, width))
        return xi[:, :n_modes] * self.coefficient_scale(n_modes)


@dataclass(frozen=True)
class ProblemBatch:
    """A batch of realisations of :math:`y^\\delta = \\mathcal{K}f + \\varepsilon`.

    All arrays are *coefficient* arrays of shape ``(n_samples, n_modes)`` in the singular
    system of the operator, so that Euclidean norms are :math:`L^2` norms (Parseval).
    """

    true_coefficients: FloatArray
    clean_data: FloatArray
    noisy_data: FloatArray
    delta: float
    noise_scale: FloatArray = field(repr=False)
    """Per-mode noise standard deviation, shape ``(n_samples, n_modes)``."""

    @property
    def n_samples(self) -> int:
        return int(self.true_coefficients.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.true_coefficients.shape[1])

    @property
    def noise_norm(self) -> FloatArray:
        """Absolute :math:`L^2` noise norm :math:`\\|\\varepsilon\\|` per sample.

        This is the :math:`\\delta` that Theorem 4.6 and the discrepancy principle are
        stated in terms of.  It is *not* equal to :attr:`delta` times :math:`\\|y\\|`: under
        a coloured noise model the per-mode level :attr:`delta` and the :math:`L^2` norm
        differ by a resolution-dependent factor (for the critical colouring, the square
        root of a harmonic sum, so it grows like :math:`\\sqrt{\\log N}`).
        """
        return np.linalg.norm(self.noisy_data - self.clean_data, axis=-1)

    @property
    def realised_noise_level(self) -> FloatArray:
        """Actual relative noise :math:`\\|\\varepsilon\\|/\\|y\\|` per sample."""
        return self.noise_norm / np.linalg.norm(self.clean_data, axis=-1)


@dataclass(frozen=True)
class InverseProblemSuite:
    """The 1D inverse-problem suite: operator + prior + noise model.

    A suite is *resolution-agnostic*: :meth:`sample` takes the number of modes, so the same
    suite generates the training data at :math:`N=256` and the evaluation data at
    :math:`N=2048`.  Because both the prior scale :math:`n^{-a}` and the noise profile
    :math:`n^{-q}` are functions of the absolute mode index, a mode carries the same
    statistics at every resolution -- this is what makes the zero-shot comparison
    meaningful rather than a change of problem.

    Stronger still, for a fixed seed the coarse sample is an exact *prefix* of the fine
    one: both the source amplitudes and the noise are drawn at the operator's full width
    and then truncated (see the ``draw_modes`` arguments).  So refining the grid reveals
    more of the same realisation rather than drawing a fresh one, and errors measured at
    two resolutions are directly comparable sample by sample.
    """

    operator: DiagonalSpectralOperator
    prior: SobolevPrior = field(default_factory=SobolevPrior)
    noise_model: NoiseModel = NoiseModel.CRITICAL

    @property
    def ill_posedness(self) -> float:
        """Operator decay exponent :math:`p`."""
        return self.operator.ill_posedness

    @property
    def smoothness(self) -> float:
        """Source regularity :math:`s`."""
        return self.prior.smoothness

    def theoretical_rate(self) -> float:
        """Convergence order predicted for this suite's noise model."""
        return self.noise_model.observable_rate(self.smoothness, self.ill_posedness)

    def deterministic_rate(self) -> float:
        """The paper's rate :math:`s/(s+p)` (Theorem 4.6), independent of noise model."""
        return self.smoothness / (self.smoothness + self.ill_posedness)

    def signal_noise_crossover(self, delta: float) -> float:
        """Mode index where the noise overtakes the signal, :math:`n_*(\\delta)`.

        The expected data magnitude is :math:`\\sigma_n \\tau_n \\asymp n^{-(p+a)}` and the
        noise magnitude is :math:`\\delta\\|y\\| n^{-q}`, so they cross at

        .. math:: n_* \\asymp \\delta^{-1/(p + a - q)} .

        Above :math:`n_*` the measured coefficients are essentially pure noise, which is
        what makes a robust noise-floor estimate possible (see :mod:`specinv.scnet`).
        """
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        exponent = self.ill_posedness + self.prior.decay_exponent
        exponent -= self.noise_model.colouring_exponent
        return float(delta ** (-1.0 / exponent))

    def recommended_noise_band(
        self, delta_min: float, n_modes: int, margin: float = 3.0
    ) -> tuple[int, int]:
        """A noise-floor band valid for every :math:`\\delta \\ge \\delta_\\min`.

        The band must sit above :math:`n_*(\\delta_\\min)` -- the crossover is largest at the
        *smallest* noise level -- with some margin, and inside the available modes.  Use
        this when sweeping :math:`\\delta` over many decades; the
        :class:`~specinv.scnet.SCNetConfig` default ``(32, 64)`` is the value it returns for
        :math:`\\delta_\\min = 10^{-3}`.

        Raises
        ------
        ValueError
            If ``n_modes`` is too small to contain a valid band, i.e. if the resolution
            cannot resolve the noise floor at ``delta_min``.
        """
        lo = int(np.ceil(margin * self.signal_noise_crossover(delta_min)))
        hi = min(2 * lo, n_modes)
        if lo >= n_modes:
            raise ValueError(
                f"n_modes={n_modes} is too small for delta_min={delta_min:g}: the noise "
                f"floor only emerges above mode {lo}. Increase the resolution."
            )
        return lo, hi

    def sample(
        self,
        n_samples: int,
        delta: float,
        rng: np.random.Generator,
        n_modes: int | None = None,
    ) -> ProblemBatch:
        """Generate a batch at relative noise level ``delta``.

        Parameters
        ----------
        n_modes:
            Resolution to sample at; defaults to the operator's mode count.  Must not
            exceed it (extend the operator to go finer).
        """
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        n_modes = self.operator.n_modes if n_modes is None else int(n_modes)
        operator = self.operator.restrict(n_modes)

        full = self.operator.n_modes
        true_coefficients = self.prior.sample(n_samples, n_modes, rng, draw_modes=full)
        clean_data = operator.apply(true_coefficients)
        noise_scale = self.noise_scale(clean_data, delta)
        noisy_data = self.corrupt(
            clean_data, delta, rng, noise_scale, draw_modes=full
        )

        return ProblemBatch(
            true_coefficients=true_coefficients,
            clean_data=clean_data,
            noisy_data=noisy_data,
            delta=float(delta),
            noise_scale=noise_scale,
        )

    def noise_scale(
        self, clean_data: FloatArray, delta: float | FloatArray
    ) -> FloatArray:
        """Per-mode noise standard deviation for the configured noise model.

        ``delta`` may be a scalar or one value per sample (shape ``(n_samples,)``).
        """
        clean_data = np.asarray(clean_data, dtype=np.float64)
        n_modes = clean_data.shape[-1]
        data_norm = np.linalg.norm(clean_data, axis=-1, keepdims=True)
        n = np.arange(1, n_modes + 1, dtype=np.float64)
        if self.noise_model is NoiseModel.CRITICAL:
            profile = n ** (-0.5)
        elif self.noise_model is NoiseModel.WHITE_POINTWISE:
            profile = np.ones_like(n)
        else:
            profile = np.full_like(n, 1.0 / np.sqrt(n_modes))
        level = np.reshape(np.asarray(delta, dtype=np.float64), (-1, 1))
        if level.size not in (1, clean_data.shape[0]):
            raise ValueError("delta must be scalar or have one entry per sample")
        return level * data_norm * profile

    def corrupt(
        self,
        clean_data: FloatArray,
        delta: float | FloatArray,
        rng: np.random.Generator,
        noise_scale: FloatArray | None = None,
        draw_modes: int | None = None,
    ) -> FloatArray:
        """Add noise at level ``delta`` to clean data coefficients."""
        clean_data = np.asarray(clean_data, dtype=np.float64)
        if noise_scale is None:
            noise_scale = self.noise_scale(clean_data, delta)
        width = max(int(draw_modes or 0), clean_data.shape[-1])
        white = rng.standard_normal((clean_data.shape[0], width))[
            :, : clean_data.shape[-1]
        ]
        noisy = clean_data + noise_scale * white
        if self.noise_model is NoiseModel.WHITE_ENERGY:
            residual = noisy - clean_data
            level = np.reshape(np.asarray(delta, dtype=np.float64), (-1, 1))
            target = level * np.linalg.norm(clean_data, axis=-1, keepdims=True)
            actual = np.linalg.norm(residual, axis=-1, keepdims=True)
            noisy = clean_data + residual * (target / np.maximum(actual, 1e-300))
        return noisy
