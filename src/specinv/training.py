"""Training of SC-Net with the Sobolev-weighted loss of Eq. (7).

Loss
----
Eq. (7) of arXiv:2603.20602 is

.. math::
    \\mathcal{L}(\\theta) = \\frac1M \\sum_i \\Big(
        \\|f_\\theta(y^{(i)}) - f^{(i)}\\|_{L^2}^2
      + \\gamma \\|\\nabla f_\\theta(y^{(i)}) - \\nabla f^{(i)}\\|_{L^2}^2 \\Big),

evaluated spectrally as in the §3.2 remark: with :math:`-\\Delta v_n = \\lambda_n v_n`,

.. math:: \\|\\nabla f_\\theta - \\nabla f\\|_{L^2}^2 = \\sum_n \\lambda_n |\\hat f_n - f_n|^2 .

Two documented deviations:

* **Per-sample normalisation.**  With ``relative_loss=True`` (default) each term is divided
  by :math:`\\|f^{(i)}\\|^2`.  The prior of §5.1 has :math:`\\|f\\|` dominated by the first
  mode and therefore a heavy spread across samples, so an unnormalised loss is effectively
  a weighted fit that over-serves large-norm samples.  Normalising makes the training
  objective the quantity that is actually reported (mean relative :math:`L^2` error).
* **Gradient-term weight.**  :math:`\\lambda_n = (n\\pi)^2` grows without bound, so a raw
  :math:`\\gamma>0` lets a handful of high modes dominate.  We normalise the Sobolev weights
  by :math:`\\lambda_1`, i.e. use :math:`1 + \\gamma \\lambda_n/\\lambda_1`, keeping
  :math:`\\gamma` dimensionless and the default :math:`\\gamma` meaningful across
  resolutions.

Noise-level sampling and the log-domain loss
-------------------------------------------
§5.2 evaluates one model over five noise levels, so training draws :math:`\\delta`
log-uniformly from ``delta_range`` and the network learns to read the noise floor off the
data.  Setting ``delta_range=(d, d)`` recovers single-noise-level training.

Training one model across two decades of :math:`\\delta` has a subtlety worth naming.  The
attainable squared error at :math:`\\delta = 10^{-3}` is about a hundred times smaller than
at :math:`\\delta = 10^{-1}`, so a plain average of squared errors is almost entirely a fit
to the noisiest samples, and the filter comes out visibly too broad at the quiet end -- it
barely costs any error there, but it is the wrong filter.  With ``log_domain_loss=True``
(the default) we average :math:`\\log` of the per-sample relative loss instead, which makes
every decade of :math:`\\delta` contribute equally.

This is a *balancing* choice, not a rate assumption: it contains no reference to
:math:`s`, :math:`p` or the expected order, and it can only reduce a downward bias in the
measured convergence order that comes from under-fitting the small-:math:`\\delta` end.
``experiments/ablations.py`` reports both settings.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .problems import InverseProblemSuite
from .scnet import SCNet

__all__ = ["TrainConfig", "TrainHistory", "train_scnet", "sobolev_loss"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrainConfig:
    """Hyper-parameters for :func:`train_scnet`.

    Attributes
    ----------
    n_train, n_val:
        Dataset sizes.  The paper uses 2000 training and 500 evaluation samples (§5.1).
    n_modes:
        Training resolution.  The paper trains at :math:`N=256` (§5.4).
    delta_range:
        Log-uniform range of relative noise levels seen in training.
    gamma:
        Sobolev weight of Eq. (7).
    epochs, batch_size, learning_rate, weight_decay:
        Optimiser settings (Adam + cosine decay).
    resample_every:
        Redraw the noise (and, if ``resample_signals``, the sources) every this many
        epochs.  Fresh noise acts as the natural data augmentation for this problem.
    seed:
        Base seed; training data, validation data and initialisation are derived from it.
    """

    n_train: int = 2000
    n_val: int = 500
    n_modes: int = 256
    delta_range: tuple[float, float] = (1e-3, 1e-1)
    gamma: float = 0.1
    epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    resample_every: int = 5
    resample_signals: bool = True
    relative_loss: bool = True
    log_domain_loss: bool = True
    grad_clip: float = 5.0
    seed: int = 0
    log_every: int = 50
    device: str = "cpu"

    def __post_init__(self) -> None:
        lo, hi = self.delta_range
        if not 0.0 < lo <= hi:
            raise ValueError(f"delta_range must satisfy 0 < lo <= hi, got {self.delta_range}")
        if self.n_train < 1 or self.n_val < 1:
            raise ValueError("dataset sizes must be positive")
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative")


@dataclass
class TrainHistory:
    """Per-epoch training diagnostics."""

    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_error: list[float] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def best_val_error(self) -> float:
        return min(self.val_error) if self.val_error else math.nan


def sobolev_loss(
    reconstruction: Tensor,
    truth: Tensor,
    sobolev_weights: Tensor,
    relative: bool = True,
    log_domain: bool = False,
) -> Tensor:
    """Spectral form of Eq. (7).

    Parameters
    ----------
    reconstruction, truth:
        Coefficient tensors of shape ``(batch, n_modes)``.
    sobolev_weights:
        Per-mode weights :math:`1 + \\gamma\\lambda_n/\\lambda_1`, shape ``(n_modes,)``.
    relative:
        Divide each sample's loss by :math:`\\|f\\|^2` (see module docstring).
    log_domain:
        Average :math:`\\log` of the per-sample loss instead of the loss itself
        (see module docstring).
    """
    residual = (reconstruction - truth) ** 2
    numerator = (residual * sobolev_weights).sum(dim=-1)
    if relative:
        denominator = torch.clamp((truth**2).sum(dim=-1), min=1e-30)
        per_sample = numerator / denominator
    else:
        per_sample = numerator
    if log_domain:
        return torch.log(torch.clamp(per_sample, min=1e-24)).mean()
    return per_sample.mean()


def _sample_deltas(
    n: int, delta_range: tuple[float, float], rng: np.random.Generator
) -> FloatArray:
    lo, hi = delta_range
    if lo == hi:
        return np.full(n, lo)
    return np.asarray(np.exp(rng.uniform(np.log(lo), np.log(hi), size=n)))


def _make_dataset(
    suite: InverseProblemSuite,
    n_samples: int,
    n_modes: int,
    delta_range: tuple[float, float],
    rng: np.random.Generator,
    signals: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Build ``(noisy_data, true_coefficients)``, one noise level per sample."""
    operator = suite.operator.restrict(n_modes)
    if signals is None:
        signals = suite.prior.sample(n_samples, n_modes, rng)
    clean = operator.apply(signals)
    deltas = _sample_deltas(n_samples, delta_range, rng)
    return suite.corrupt(clean, deltas, rng), signals


def train_scnet(
    model: SCNet,
    suite: InverseProblemSuite,
    config: TrainConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainHistory:
    """Train ``model`` on ``suite`` in place and return the history.

    The model is calibrated (feature standardisation and clamping range) on the initial
    training batch before the first optimiser step.
    """
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)

    rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 9_991)

    operator = suite.operator.restrict(config.n_modes)
    sigma = torch.as_tensor(operator.singular_values, dtype=torch.float32, device=device)

    lam = (np.arange(1, config.n_modes + 1, dtype=np.float64) * np.pi) ** 2
    weights = torch.as_tensor(
        1.0 + config.gamma * lam / lam[0], dtype=torch.float32, device=device
    )

    signals = suite.prior.sample(config.n_train, config.n_modes, rng)
    noisy, signals = _make_dataset(
        suite, config.n_train, config.n_modes, config.delta_range, rng, signals
    )
    val_noisy, val_signals = _make_dataset(
        suite, config.n_val, config.n_modes, config.delta_range, val_rng
    )

    x = torch.as_tensor(noisy, dtype=torch.float32, device=device)
    t = torch.as_tensor(signals, dtype=torch.float32, device=device)
    xv = torch.as_tensor(val_noisy, dtype=torch.float32, device=device)
    tv = torch.as_tensor(val_signals, dtype=torch.float32, device=device)

    model.calibrate(x, sigma)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    history = TrainHistory()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = math.inf
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        if config.resample_every > 0 and epoch > 1 and epoch % config.resample_every == 1:
            noisy, signals = _make_dataset(
                suite,
                config.n_train,
                config.n_modes,
                config.delta_range,
                rng,
                None if config.resample_signals else signals,
            )
            x = torch.as_tensor(noisy, dtype=torch.float32, device=device)
            t = torch.as_tensor(signals, dtype=torch.float32, device=device)

        model.train()
        permutation = torch.randperm(x.shape[0], device=device)
        running = 0.0
        for start in range(0, x.shape[0], config.batch_size):
            idx = permutation[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            prediction = model(x[idx], sigma)
            loss = sobolev_loss(
                prediction,
                t[idx],
                weights,
                config.relative_loss,
                config.log_domain_loss,
            )
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            running += float(loss.item()) * idx.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            prediction = model(xv, sigma)
            rel = torch.linalg.vector_norm(prediction - tv, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(tv, dim=-1), min=1e-30
            )
            val_error = float(rel.mean().item())

        history.epochs.append(epoch)
        history.train_loss.append(running / x.shape[0])
        history.val_error.append(val_error)

        if val_error < best_val:
            best_val = val_error
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if progress is not None and (epoch % config.log_every == 0 or epoch == 1):
            progress(
                f"epoch {epoch:4d}/{config.epochs}  loss={history.train_loss[-1]:.5f}  "
                f"val_rel_l2={val_error:.5f}"
            )

    model.load_state_dict(best_state)
    history.wall_time = time.perf_counter() - started
    return history


def evaluate_on_resolutions(
    model: SCNet,
    suite: InverseProblemSuite,
    resolutions: Sequence[int],
    delta: float,
    n_samples: int,
    seed: int = 0,
) -> dict[int, FloatArray]:
    """Reconstruct at several resolutions and return per-sample relative errors."""
    out: dict[int, FloatArray] = {}
    for n_modes in resolutions:
        rng = np.random.default_rng(seed)
        batch = suite.sample(n_samples, delta, rng, n_modes=n_modes)
        operator = suite.operator.restrict(n_modes)
        reconstruction = model.reconstruct(batch.noisy_data, operator.singular_values)
        num = np.linalg.norm(reconstruction - batch.true_coefficients, axis=-1)
        den = np.linalg.norm(batch.true_coefficients, axis=-1)
        out[n_modes] = num / den
    return out