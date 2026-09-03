"""SC-Net: the learnable spectral filter of §3 of arXiv:2603.20602.

The reconstruction operator is Eq. (6),

.. math::
    \\mathcal{R}_\\theta(y^\\delta)
      = \\sum_{n=1}^{N} \\Psi_\\theta(y_n, \\sigma_n)\\,\\frac{\\langle y^\\delta, u_n\\rangle}{\\sigma_n}\\, v_n ,

i.e. *analysis* in the singular system, a *pointwise* learned damping factor
:math:`\\Psi_\\theta \\in [0, C_\\Psi]`, then *synthesis*.  There is no spatial layer
anywhere: the only thing that is learned is a scalar function of one spectral mode, which is
why the filter can be tabulated and plotted (:meth:`SCNet.filter_profile`) and why the model
transfers to resolutions it never saw.

Input features
--------------
Feeding the raw pair :math:`(y_n, \\sigma_n)` to an MLP, as written in §3.1.2, has two
practical problems that the paper does not address but that any implementation must:

1. **Scale.** :math:`y_n` and :math:`\\sigma_n` range over many decades, so an MLP with
   bounded activations saturates almost everywhere.
2. **Noise-level awareness.** §3.1.2 motivates :math:`\\Psi_\\theta` as reweighting "based on
   the signal-to-noise ratio", but :math:`(y_n, \\sigma_n)` alone pins the SNR down only if
   the network has memorised a single noise level.  A model that is supposed to work across
   :math:`\\delta \\in [10^{-3}, 10^{-1}]` (§5.2) needs to know where it is.

``feature_set="full"`` (the default) therefore takes logarithms of the paper's quantities
and adds two global statistics read off the data:

==================  ======================================================================
``log_sigma``       :math:`\\log_{10}\\sigma_n`
``log_coeff``       :math:`\\log_{10}|y_n|`
``log_naive``       :math:`\\log_{10}(|y_n|/\\sigma_n)`, the naive inverse of Eq. (3)
``log_amplitude``   :math:`\\log_{10}\\rho`, :math:`\\rho = \\|(y_1,\\dots,y_{32})\\|`
``log_noise``       :math:`\\hat g = \\log_{10}\\mathrm{median}_{32\\le n\\le 64}|y_n|`
``log_snr``         :math:`\\log_{10}|y_n| - \\hat g`, the per-mode SNR
==================  ======================================================================

The magnitudes are deliberately kept **absolute** rather than divided by :math:`\\|y\\|`.
That is not a detail: the prior of §5.1 fixes the coefficient scale at :math:`n^{-(s+1/2)}`
in absolute units while the noise is specified relative to :math:`\\|y\\|`, so the
Bayes-optimal damping

.. math:: \\lambda_n = \\big(1 + \\delta^2\\|y\\|^2 n^{2(s+p)}\\big)^{-1}

depends on the *absolute* noise level :math:`\\delta\\|y\\|`, not on :math:`\\delta` alone.
(Its half-power point is :math:`n = (\\delta\\|y\\|)^{-1/(s+p)}`, the truncation index of
Theorem 4.6.)
A filter fed only scale-free ratios cannot represent it and plateaus measurably above the
optimum.  Absolute features are also the literal reading of Eq. (5).

Both global statistics are computed on **fixed absolute mode bands**, never on "the top
half of the spectrum":

* :math:`\\rho` -- signal-dominated, hence a stable read-out of the source amplitude;
* :math:`\\hat g` -- modes 32-64 are noise-dominated for every :math:`\\delta \\ge 10^{-9}`
  in this suite, so their median is a robust estimate of the noise floor.  It supplies the
  noise level that Theorem 4.6 assumes known; the model is never told :math:`\\delta`.

Using absolute bands is what preserves discretisation invariance: every feature at mode
:math:`n` takes the same value whether the grid has 256 or 2048 points, so the filter is
literally the same function of :math:`n`.  A band defined as a *fraction* of the grid would
silently change meaning with the resolution and destroy the zero-shot property.

``feature_set="paper"`` implements §3.1.2 literally (inputs :math:`(y_n,\\sigma_n)`,
standardised only) and is kept for the ablation in ``experiments/``.

Spectral aperture and extrapolation
-----------------------------------
Eq. (6) sums over :math:`n = 1,\\dots,N`, and §3.1.1 describes that :math:`N` as "a
hyperparameter chosen such that :math:`\\sigma_N` remains above the machine precision
threshold" -- so the truncation index is part of the method, not of the grid.  We call it
the *aperture* and store it on the model, set by default to the training resolution.
Modes beyond the aperture get :math:`\\Psi_n = 0` exactly.

This matters on a finer grid.  At :math:`N=2048` the naive inverse of a noise-dominated
mode is amplified by :math:`\\sigma_n^{-1} \\approx 10^{5}`, so a filter leaking even
:math:`10^{-3}` there would swamp the reconstruction.  With the aperture, §3.3's claim
holds by construction: the learned map acts on coefficients, and refining the mesh changes
only the *synthesis*, which is what "the model can be evaluated on any mesh where the
:math:`v_n` can be interpolated" means.

Set ``aperture_modes=None`` to instead filter *every* available mode, which is a strictly
harder test -- the network must extrapolate to singular values below anything it trained
on and reject them on its own.  For that regime we clamp the standardised features to the
range recorded during :meth:`SCNet.calibrate`.  What clamping guarantees is precisely that
the MLP is never *evaluated* outside the box of feature values it was fitted on, so the
filter on a finer grid interpolates learned behaviour instead of extrapolating; in
particular ``log_sigma`` saturates rather than running off to minus infinity.  It does not
by itself prove the damping vanishes there -- that is the aperture's job -- so both
settings are measured in ``experiments/zero_shot.py`` and reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

__all__ = ["SCNetConfig", "SCNet", "FEATURE_NAMES"]

FloatArray = NDArray[np.float64]

FeatureSet = Literal["full", "paper"]

FEATURE_NAMES: dict[FeatureSet, tuple[str, ...]] = {
    "full": (
        "log_sigma",
        "log_coeff",
        "log_naive",
        "log_amplitude",
        "log_noise",
        "log_snr",
    ),
    "paper": ("coeff", "sigma"),
}

_EPS = 1e-30


@dataclass(frozen=True)
class SCNetConfig:
    """Configuration of the learned spectral filter.

    Attributes
    ----------
    hidden_sizes:
        Widths of the hidden layers of the pointwise MLP :math:`\\Psi_\\theta`.
    activation:
        Hidden activation.  ``"tanh"`` matches the paper's "bounded activation function".
    c_psi:
        The constant :math:`C_\\Psi \\ge 1` bounding the filter (§3.1.2, "Crucial
        Constraint").  With a sigmoid output the natural value is 1, which also makes
        :math:`\\lambda_n \\in (0,1)` directly comparable with Tikhonov/TSVD damping.
    feature_set:
        ``"full"`` for the scale-free feature set, ``"paper"`` for literal
        :math:`(y_n,\\sigma_n)` inputs.
    use_noise_estimate:
        Include the data-driven noise-floor features.  Disabling it recovers a filter that
        can only depend on :math:`(\\sigma_n, |y_n|)`.
    noise_band:
        Inclusive absolute mode range used for the noise-floor median.
    reference_modes:
        Number of leading modes used for the amplitude scale :math:`\\rho`.
    clamp_features:
        Clamp standardised features to the calibrated range (see module docstring).
    aperture_modes:
        Truncation index :math:`N` of Eq. (6).  ``"train"`` (default) fixes it to the
        training resolution at calibration time; an ``int`` sets it explicitly; ``None``
        filters every mode the evaluation grid provides.
    """

    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    activation: Literal["tanh", "relu", "gelu"] = "tanh"
    c_psi: float = 1.0
    feature_set: FeatureSet = "full"
    use_noise_estimate: bool = True
    noise_band: tuple[int, int] = (32, 64)
    reference_modes: int = 32
    clamp_features: bool = True
    feature_clip_quantile: float = 5e-4
    aperture_modes: int | Literal["train"] | None = "train"
    init_seed: int | None = 0
    """Seed for weight initialisation.

    Set so that constructing a model is deterministic regardless of the ambient RNG state;
    otherwise a run's results would depend on how much random number consumption happened
    beforehand.  Pass ``None`` to use the global generator instead.
    """

    def __post_init__(self) -> None:
        if self.c_psi < 1.0:
            raise ValueError("c_psi must be >= 1 (Sec. 3.1.2)")
        if isinstance(self.aperture_modes, int) and self.aperture_modes < 1:
            raise ValueError("aperture_modes must be >= 1, an int, 'train' or None")
        if not isinstance(self.aperture_modes, int) and self.aperture_modes not in (
            "train",
            None,
        ):
            raise ValueError("aperture_modes must be >= 1, an int, 'train' or None")
        if self.feature_set not in ("full", "paper"):
            raise ValueError(f"unknown feature_set {self.feature_set!r}")
        lo, hi = self.noise_band
        if not 1 <= lo < hi:
            raise ValueError(f"noise_band must satisfy 1 <= lo < hi, got {self.noise_band}")
        if self.reference_modes < 1:
            raise ValueError("reference_modes must be >= 1")
        if not 0.0 <= self.feature_clip_quantile < 0.5:
            raise ValueError("feature_clip_quantile must lie in [0, 0.5)")

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names of the features actually fed to the MLP."""
        names = FEATURE_NAMES[self.feature_set]
        if self.feature_set == "full" and not self.use_noise_estimate:
            names = tuple(n for n in names if n not in ("log_noise", "log_snr"))
        return names

    @property
    def uses_absolute_scale(self) -> bool:
        """Whether the feature set exposes the absolute amplitude of the data."""
        return self.feature_set == "paper" or "log_amplitude" in self.feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _make_activation(kind: str) -> nn.Module:
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}[kind]


class SCNet(nn.Module):
    """Spectral Correction Network: a pointwise, bounded, learned spectral filter.

    The module consumes and produces *coefficient* arrays in the singular system of the
    forward operator, so it is agnostic to the spatial discretisation.

    Examples
    --------
    >>> import numpy as np, torch
    >>> from specinv import SCNet, SCNetConfig, power_law_operator
    >>> op = power_law_operator(64, 1.5)
    >>> net = SCNet(SCNetConfig(hidden_sizes=(8,), noise_band=(8, 16), reference_modes=8))
    >>> y = torch.randn(4, 64) * torch.as_tensor(op.singular_values)
    >>> psi = net.filter_coefficients(y, torch.as_tensor(op.singular_values))
    >>> bool(((psi >= 0) & (psi <= 1)).all())
    True
    """

    # Declared so that static analysis sees buffers as tensors rather than as the
    # `Tensor | Module` union that nn.Module.__getattr__ returns.
    feature_mean: Tensor
    feature_std: Tensor
    feature_lo: Tensor
    feature_hi: Tensor
    calibrated: Tensor
    aperture: Tensor

    def __init__(self, config: SCNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SCNetConfig()

        layers: list[nn.Module] = []
        in_dim = self.config.n_features
        for width in self.config.hidden_sizes:
            layers.append(nn.Linear(in_dim, width))
            layers.append(_make_activation(self.config.activation))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        if self.config.init_seed is not None:
            self.reset_parameters(self.config.init_seed)

        n_feat = self.config.n_features
        self.register_buffer("feature_mean", torch.zeros(n_feat))
        self.register_buffer("feature_std", torch.ones(n_feat))
        self.register_buffer("feature_lo", torch.full((n_feat,), -float("inf")))
        self.register_buffer("feature_hi", torch.full((n_feat,), float("inf")))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))
        initial = self.config.aperture_modes
        self.register_buffer(
            "aperture",
            torch.tensor(initial if isinstance(initial, int) else 0, dtype=torch.long),
        )

    @torch.no_grad()
    def reset_parameters(self, seed: int) -> None:
        """Re-initialise the MLP from a private generator, reproducibly.

        Reproduces PyTorch's default ``nn.Linear`` initialisation -- uniform on
        :math:`\\pm 1/\\sqrt{\\text{fan\\_in}}` -- but draws from a seeded local generator so
        the result does not depend on global RNG state.
        """
        generator = torch.Generator().manual_seed(int(seed))
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                bound = 1.0 / np.sqrt(module.in_features)
                module.weight.uniform_(-bound, bound, generator=generator)
                if module.bias is not None:
                    module.bias.uniform_(-bound, bound, generator=generator)

    @property
    def effective_aperture(self) -> int | None:
        """Truncation index :math:`N` in force, or ``None`` if all modes are filtered."""
        value = int(self.aperture.item())
        return value if value > 0 else None

    def set_aperture(self, n_modes: int | None) -> None:
        """Override the truncation index :math:`N` of Eq. (6).

        ``None`` filters every mode the evaluation grid provides.  Changing the aperture
        does not retrain anything -- it selects how many singular directions the (already
        learned) filter is applied to.
        """
        if n_modes is not None and n_modes < 1:
            raise ValueError("aperture must be >= 1 or None")
        self.aperture.fill_(0 if n_modes is None else int(n_modes))

    # ------------------------------------------------------------------ features

    def raw_features(self, noisy_data: Tensor, singular_values: Tensor) -> Tensor:
        """Build the un-standardised feature tensor of shape ``(..., n_modes, n_features)``.

        Parameters
        ----------
        noisy_data:
            Data coefficients :math:`y_n^\\delta`, shape ``(batch, n_modes)``.
        singular_values:
            Singular values :math:`\\sigma_n`, shape ``(n_modes,)``.
        """
        if noisy_data.shape[-1] != singular_values.shape[-1]:
            raise ValueError(
                f"mode count mismatch: data has {noisy_data.shape[-1]}, "
                f"operator has {singular_values.shape[-1]}"
            )
        cfg = self.config
        sigma = singular_values.to(noisy_data.dtype)
        magnitude = noisy_data.abs()

        if cfg.feature_set == "paper":
            sigma_b = sigma.expand_as(noisy_data)
            return torch.stack([noisy_data, sigma_b], dim=-1)

        n_modes = noisy_data.shape[-1]
        log_coeff = torch.log10(magnitude + _EPS)
        log_sigma = torch.log10(sigma + _EPS).expand_as(log_coeff)
        log_naive = log_coeff - log_sigma
        log_amplitude = torch.log10(self._amplitude_scale(noisy_data) + _EPS)
        log_amplitude = log_amplitude.expand_as(log_coeff)

        features = [log_sigma, log_coeff, log_naive, log_amplitude]
        if cfg.use_noise_estimate:
            lo = min(cfg.noise_band[0], n_modes) - 1
            hi = min(cfg.noise_band[1], n_modes)
            band = magnitude[..., lo:hi] if hi > lo else magnitude[..., -1:]
            log_noise = torch.log10(band.median(dim=-1, keepdim=True).values + _EPS)
            log_noise = log_noise.expand_as(log_coeff)
            features += [log_noise, log_coeff - log_noise]
        return torch.stack(features, dim=-1)

    def _amplitude_scale(self, noisy_data: Tensor) -> Tensor:
        k = min(self.config.reference_modes, noisy_data.shape[-1])
        scale = torch.linalg.vector_norm(noisy_data[..., :k], dim=-1, keepdim=True)
        return torch.clamp(scale, min=1e-30)

    def _standardise(self, features: Tensor) -> Tensor:
        out = (features - self.feature_mean) / self.feature_std
        if self.config.clamp_features:
            out = torch.clamp(out, min=self.feature_lo, max=self.feature_hi)
        return out

    @torch.no_grad()
    def calibrate(self, noisy_data: Tensor, singular_values: Tensor) -> None:
        """Record feature statistics and the admissible feature range.

        Must be called once, on training data spanning the intended noise-level range,
        before training.  Stores mean/std for standardisation and the empirical
        quantile range used for clamping at evaluation time.
        """
        features = self.raw_features(noisy_data, singular_values)
        flat = features.reshape(-1, features.shape[-1]).double()
        mean = flat.mean(dim=0)
        std = torch.clamp(flat.std(dim=0), min=1e-8)
        standardised = (flat - mean) / std

        q = self.config.feature_clip_quantile
        if q > 0.0:
            lo = torch.quantile(standardised, q, dim=0)
            hi = torch.quantile(standardised, 1.0 - q, dim=0)
        else:
            lo = standardised.min(dim=0).values
            hi = standardised.max(dim=0).values

        self.feature_mean.copy_(mean.to(self.feature_mean.dtype))
        self.feature_std.copy_(std.to(self.feature_std.dtype))
        self.feature_lo.copy_(lo.to(self.feature_lo.dtype))
        self.feature_hi.copy_(hi.to(self.feature_hi.dtype))
        self.calibrated.fill_(True)

        if self.config.aperture_modes == "train":
            self.aperture.fill_(int(noisy_data.shape[-1]))

    # ------------------------------------------------------------------ forward

    def filter_coefficients(self, noisy_data: Tensor, singular_values: Tensor) -> Tensor:
        """The learned damping profile :math:`\\Psi_\\theta(y_n,\\sigma_n) \\in [0, C_\\Psi]`.

        This is the object the paper calls the *interpretable spectral filter*; it is
        directly comparable with :func:`specinv.filters.tikhonov_damping` and friends.
        Modes beyond the aperture are returned as exact zeros.
        """
        features = self._standardise(self.raw_features(noisy_data, singular_values))
        logits = self.mlp(features).squeeze(-1)
        psi = self.config.c_psi * torch.sigmoid(logits)

        aperture = self.effective_aperture
        if aperture is not None and aperture < psi.shape[-1]:
            mask = torch.zeros_like(psi)
            mask[..., :aperture] = 1.0
            psi = psi * mask
        return psi

    def forward(self, noisy_data: Tensor, singular_values: Tensor) -> Tensor:
        """Reconstruct source coefficients via Eq. (6)."""
        psi = self.filter_coefficients(noisy_data, singular_values)
        sigma = singular_values.to(noisy_data.dtype)
        return psi * (noisy_data / sigma)

    # ------------------------------------------------------- numpy conveniences

    @torch.no_grad()
    def reconstruct(self, noisy_data: FloatArray, singular_values: FloatArray) -> FloatArray:
        """NumPy wrapper around :meth:`forward`."""
        y = torch.as_tensor(np.asarray(noisy_data, dtype=np.float64), dtype=torch.float32)
        s = torch.as_tensor(np.asarray(singular_values, dtype=np.float64), dtype=torch.float32)
        return self.forward(y, s).double().cpu().numpy()

    @torch.no_grad()
    def filter_profile(
        self, noisy_data: FloatArray, singular_values: FloatArray
    ) -> FloatArray:
        """NumPy wrapper around :meth:`filter_coefficients` (for §5.3-style plots)."""
        y = torch.as_tensor(np.asarray(noisy_data, dtype=np.float64), dtype=torch.float32)
        s = torch.as_tensor(np.asarray(singular_values, dtype=np.float64), dtype=torch.float32)
        return self.filter_coefficients(y, s).double().cpu().numpy()

    # ------------------------------------------------------------------- theory

    def lipschitz_bound(self) -> float:
        """Product of layer operator norms: an upper bound on :math:`L_\\Psi`.

        Assumption 1 of §4.1 requires :math:`\\Psi_\\theta` to be bounded (guaranteed here by
        the sigmoid, :math:`C_\\Psi`) and Lipschitz in its first argument.  The product of
        spectral norms of the linear layers, times 1/4 for the output sigmoid and 1 for
        tanh/ReLU/GELU hidden units, bounds the Lipschitz constant of the standardised map.
        """
        bound = 0.25  # sigmoid derivative bound
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                bound *= float(torch.linalg.matrix_norm(module.weight, ord=2).item())
        return bound

    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)