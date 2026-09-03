"""A spatial CNN control for the discretisation-invariance claim of §3.3 / §5.4.

§3.3 argues that a network learning a map :math:`\\mathbb{R}^N \\to \\mathbb{R}^N` on a fixed
grid "typically fails or degrades significantly when applied to a finer mesh", and §5.4
presents SC-Net's zero-shot stability as the contrast.  Stability on its own is not
evidence for that contrast, though -- it could just mean the problem is easy.  So we train
an ordinary 1D CNN on grid samples at :math:`N=256` and apply it at :math:`N=512,\\dots,2048`
under identical conditions.  It is the same experiment, the same data and the same metric;
only the hypothesis class changes.

The CNN receives the grid samples of :math:`y^\\delta` and predicts the grid samples of
:math:`f`.  Convolutions are shape-agnostic, so the model *runs* at any resolution; what
changes is the physical meaning of a pixel, and hence of a fixed-width kernel.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .basis import SineBasis
from .problems import InverseProblemSuite
from .training import TrainConfig, TrainHistory, _make_dataset

__all__ = ["SpatialCNNConfig", "SpatialCNN", "train_spatial_cnn"]

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpatialCNNConfig:
    """Configuration of the spatial CNN control."""

    channels: int = 48
    depth: int = 5
    kernel_size: int = 9
    dilations: tuple[int, ...] = (1, 2, 4, 8, 1)

    def __post_init__(self) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so that 'same' padding is exact")
        if self.depth < 1:
            raise ValueError("depth must be >= 1")
        if len(self.dilations) < self.depth:
            raise ValueError("need at least `depth` dilation entries")


class SpatialCNN(nn.Module):
    """Dilated 1D CNN mapping grid samples of :math:`y^\\delta` to grid samples of :math:`f`."""

    input_scale: Tensor
    output_scale: Tensor

    def __init__(self, config: SpatialCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpatialCNNConfig()
        cfg = self.config

        layers: list[nn.Module] = []
        in_channels = 1
        for i in range(cfg.depth):
            dilation = cfg.dilations[i]
            padding = dilation * (cfg.kernel_size - 1) // 2
            layers.append(
                nn.Conv1d(
                    in_channels,
                    cfg.channels,
                    cfg.kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            layers.append(nn.GELU())
            in_channels = cfg.channels
        layers.append(nn.Conv1d(in_channels, 1, 1))
        self.net = nn.Sequential(*layers)

        self.register_buffer("input_scale", torch.ones(()))
        self.register_buffer("output_scale", torch.ones(()))

    def forward(self, grid_values: Tensor) -> Tensor:
        """Map ``(batch, n_grid)`` measurement samples to ``(batch, n_grid)`` source samples."""
        x = (grid_values / self.input_scale).unsqueeze(1)
        return self.net(x).squeeze(1) * self.output_scale

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def reconstruct_coefficients(
        self, noisy_data: FloatArray, basis: SineBasis
    ) -> FloatArray:
        """Reconstruct in coefficient space so the metric matches SC-Net's.

        Synthesises the data onto ``basis``'s grid, applies the CNN, and analyses the
        prediction back into coefficients.  Both transforms are isometries, so the reported
        relative :math:`L^2` error is the same quantity as for SC-Net.
        """
        grid_values = basis.synthesize(np.asarray(noisy_data, dtype=np.float64))
        x = torch.as_tensor(grid_values, dtype=torch.float32)
        prediction = self.forward(x).double().cpu().numpy()
        return basis.analyze(prediction)


def train_spatial_cnn(
    model: SpatialCNN,
    suite: InverseProblemSuite,
    config: TrainConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainHistory:
    """Train the spatial control with the same data pipeline and budget as SC-Net."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed + 1)
    device = torch.device(config.device)
    model.to(device)

    basis = SineBasis(config.n_modes)
    rng = np.random.default_rng(config.seed)
    val_rng = np.random.default_rng(config.seed + 9_991)

    def build(n: int, generator: np.random.Generator) -> tuple[Tensor, Tensor]:
        noisy, signals = _make_dataset(
            suite, n, config.n_modes, config.delta_range, generator
        )
        x = torch.as_tensor(basis.synthesize(noisy), dtype=torch.float32, device=device)
        t = torch.as_tensor(basis.synthesize(signals), dtype=torch.float32, device=device)
        return x, t

    x, t = build(config.n_train, rng)
    xv, tv = build(config.n_val, val_rng)

    with torch.no_grad():
        model.input_scale.fill_(float(x.std().item()) or 1.0)
        model.output_scale.fill_(float(t.std().item()) or 1.0)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    history = TrainHistory()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = math.inf
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        if config.resample_every > 0 and epoch > 1 and epoch % config.resample_every == 1:
            x, t = build(config.n_train, rng)

        model.train()
        permutation = torch.randperm(x.shape[0], device=device)
        running = 0.0
        for start in range(0, x.shape[0], config.batch_size):
            idx = permutation[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            prediction = model(x[idx])
            residual = ((prediction - t[idx]) ** 2).sum(dim=-1)
            denominator = torch.clamp((t[idx] ** 2).sum(dim=-1), min=1e-30)
            loss = (residual / denominator).mean() if config.relative_loss else residual.mean()
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            running += float(loss.item()) * idx.numel()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            prediction = model(xv)
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
                f"[cnn] epoch {epoch:4d}/{config.epochs}  "
                f"loss={history.train_loss[-1]:.5f}  val_rel_l2={val_error:.5f}"
            )

    model.load_state_dict(best_state)
    history.wall_time = time.perf_counter() - started
    return history