"""Shared plumbing for the reproduction scripts.

Keeps the experiment scripts themselves short enough to read as a description of what was
measured: everything about *how* a method is evaluated lives here, so all experiments
necessarily use identical data, metrics and pooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from .. import __version__
from ..filters import (
    discrepancy_principle_tikhonov,
    oracle_spectral_bound,
    oracle_tikhonov,
    oracle_tsvd,
    prior_wiener,
)
from ..metrics import ErrorSummary, summarise_errors
from ..problems import InverseProblemSuite, NoiseModel, ProblemBatch, SobolevPrior
from ..operators import power_law_operator
from ..scnet import SCNet, SCNetConfig
from ..training import TrainConfig, train_scnet

__all__ = [
    "METHOD_LABELS",
    "PAPER_DELTAS",
    "PAPER_REFERENCE",
    "add_common_arguments",
    "build_suite",
    "evaluate_full_aperture",
    "evaluate_methods",
    "environment_info",
    "output_dir",
    "train_model",
    "write_csv",
    "write_json",
]

FloatArray = NDArray[np.float64]

#: Noise levels swept in §5.2.
PAPER_DELTAS: tuple[float, ...] = (1e-1, 5e-2, 1e-2, 5e-3, 1e-3)

#: Figures quoted in the paper, for side-by-side reporting.
PAPER_REFERENCE: dict[str, Any] = {
    "scnet_rate_slope": 0.50,
    "oracle_tikhonov_rate_slope": 0.42,
    "theoretical_rate_s_p_1p5": 0.50,
    "zero_shot_error_n256": 0.2415,
    "zero_shot_error_n2048": 0.2292,
    "train_resolution": 256,
    "test_resolutions": [512, 1024, 2048],
    "n_train": 2000,
    "n_test": 500,
}

METHOD_LABELS: dict[str, str] = {
    "scnet": "SC-Net (learned spectral filter)",
    "scnet_full_aperture": "SC-Net, aperture widened to all modes",
    "oracle_tikhonov": "Oracle Tikhonov (per-sample optimal alpha)",
    "oracle_tsvd": "Oracle TSVD (per-sample optimal cutoff)",
    "tikhonov_discrepancy": "Tikhonov + discrepancy principle (non-oracle)",
    "prior_wiener": "Prior Wiener (pointwise Bayes ceiling)",
    "oracle_spectral_bound": "Oracle spectral bound (unattainable floor)",
    "naive_inverse": "Naive pseudo-inverse (no regularisation)",
    "spatial_cnn": "Spatial CNN (fixed-grid control)",
}


def build_suite(
    n_modes: int = 2048,
    ill_posedness: float = 1.5,
    smoothness: float = 1.5,
    noise_model: str | NoiseModel = NoiseModel.CRITICAL,
    amplitude: str = "gaussian",
) -> InverseProblemSuite:
    """Instantiate the §5.1 suite.

    ``n_modes`` is the *largest* resolution the suite must support (2048 for the zero-shot
    experiment); individual samples are drawn at whatever resolution is requested.
    """
    return InverseProblemSuite(
        operator=power_law_operator(n_modes, ill_posedness),
        prior=SobolevPrior(smoothness=smoothness, amplitude=amplitude),  # type: ignore[arg-type]
        noise_model=NoiseModel(noise_model),
    )


def train_model(
    suite: InverseProblemSuite,
    train_config: TrainConfig,
    net_config: SCNetConfig | None = None,
    verbose: bool = True,
) -> tuple[SCNet, dict[str, Any]]:
    """Train an SC-Net and return it with a JSON-serialisable training summary."""
    model = SCNet(net_config or SCNetConfig())
    printer = (lambda msg: print(f"    {msg}", flush=True)) if verbose else None
    history = train_scnet(model, suite, train_config, progress=printer)
    summary = {
        "n_parameters": model.n_parameters(),
        "lipschitz_bound": model.lipschitz_bound(),
        "epochs": train_config.epochs,
        "n_train": train_config.n_train,
        "train_resolution": train_config.n_modes,
        "delta_range": list(train_config.delta_range),
        "gamma": train_config.gamma,
        "final_val_relative_l2": history.val_error[-1] if history.val_error else None,
        "best_val_relative_l2": history.best_val_error,
        "wall_time_seconds": history.wall_time,
        "feature_names": list(model.config.feature_names),
        "noise_band": list(model.config.noise_band),
    }
    return model, summary


@dataclass(frozen=True)
class MethodOutputs:
    """Reconstructions and damping profiles for every evaluated method."""

    reconstructions: dict[str, FloatArray]
    dampings: dict[str, FloatArray]

    def summarise(self, truth: FloatArray) -> dict[str, ErrorSummary]:
        return {k: summarise_errors(v, truth) for k, v in self.reconstructions.items()}


def evaluate_methods(
    model: SCNet | None,
    suite: InverseProblemSuite,
    batch: ProblemBatch,
    methods: Sequence[str] | None = None,
) -> MethodOutputs:
    """Run every requested method on one batch.

    All methods act on the same realisations, so differences are attributable to the
    methods and not to the draw.
    """
    n_modes = batch.n_modes
    operator = suite.operator.restrict(n_modes)
    sv = operator.singular_values
    y = batch.noisy_data
    truth = batch.true_coefficients

    wanted = list(
        methods
        if methods is not None
        else [
            "scnet",
            "oracle_tikhonov",
            "oracle_tsvd",
            "tikhonov_discrepancy",
            "prior_wiener",
            "oracle_spectral_bound",
        ]
    )
    reconstructions: dict[str, FloatArray] = {}
    dampings: dict[str, FloatArray] = {}

    for name in wanted:
        if name == "scnet":
            if model is None:
                continue
            reconstructions[name] = model.reconstruct(y, sv)
            dampings[name] = model.filter_profile(y, sv)
        elif name == "oracle_tikhonov":
            sel = oracle_tikhonov(y, sv, truth)
            reconstructions[name], dampings[name] = sel.reconstruction, sel.damping
        elif name == "oracle_tsvd":
            sel = oracle_tsvd(y, sv, truth)
            reconstructions[name], dampings[name] = sel.reconstruction, sel.damping
        elif name == "tikhonov_discrepancy":
            if batch.delta <= 0.0:
                continue
            sel = discrepancy_principle_tikhonov(y, sv, batch.noise_norm)
            reconstructions[name], dampings[name] = sel.reconstruction, sel.damping
        elif name == "prior_wiener":
            sel = prior_wiener(
                y, sv, suite.prior.coefficient_scale(n_modes), batch.noise_scale
            )
            reconstructions[name], dampings[name] = sel.reconstruction, sel.damping
        elif name == "oracle_spectral_bound":
            sel = oracle_spectral_bound(y, sv, truth)
            reconstructions[name], dampings[name] = sel.reconstruction, sel.damping
        elif name == "naive_inverse":
            reconstructions[name] = operator.naive_inverse(y)
            dampings[name] = np.ones_like(y)
        else:
            raise ValueError(f"unknown method {name!r}")

    return MethodOutputs(reconstructions, dampings)


def evaluate_full_aperture(
    model: SCNet, suite: InverseProblemSuite, batch: ProblemBatch
) -> ErrorSummary:
    """Evaluate ``model`` with its aperture widened to every available mode.

    Same weights, no retraining: only the truncation index :math:`N` of Eq. (6) changes,
    which forces the learned filter to extrapolate to unseen singular values.
    """
    sv = suite.operator.restrict(batch.n_modes).singular_values
    trained_aperture = model.effective_aperture
    model.set_aperture(None)
    try:
        reconstruction = model.reconstruct(batch.noisy_data, sv)
    finally:
        model.set_aperture(trained_aperture)
    return summarise_errors(reconstruction, batch.true_coefficients)


def environment_info() -> dict[str, Any]:
    """Record enough of the environment to make a result traceable."""
    return {
        "specinv_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "paper": "arXiv:2603.20602",
    }


def output_dir(path: str | Path) -> Path:
    """Create and return an output directory."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as indented JSON, converting NumPy scalars."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"not JSON serialisable: {type(obj)}")

    path.write_text(json.dumps(payload, indent=2, default=default) + "\n")
    print(f"  wrote {path}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a list of uniform dicts as CSV."""
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the arguments shared by all experiment scripts."""
    parser.add_argument(
        "--results-dir",
        default="results",
        help="directory for JSON/CSV output (default: results)",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="training epochs (default: 300)"
    )
    parser.add_argument(
        "--n-train", type=int, default=2000, help="training samples (paper: 2000)"
    )
    parser.add_argument(
        "--n-test", type=int, default=500, help="test samples (paper: 500)"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short run for smoke-testing (fewer epochs and samples)",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="skip matplotlib figure generation"
    )
    return parser


def apply_quick(args: argparse.Namespace) -> argparse.Namespace:
    """Shrink the workload when ``--quick`` is given."""
    if args.quick:
        args.epochs = min(args.epochs, 30)
        args.n_train = min(args.n_train, 400)
        args.n_test = min(args.n_test, 200)
    return args