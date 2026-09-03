"""§5.4 -- zero-shot super-resolution / resolution transfer.

Reproduces Figure 3: a model trained only at :math:`N=256` is evaluated, without any
retraining or fine-tuning, at :math:`N \\in \\{256, 512, 1024, 2048\\}`.  The paper reports
relative :math:`L^2` errors of 0.2415 on the training grid and 0.2292 at :math:`N=2048`.

Three things make this a real test rather than a formality:

1. **A control.**  An ordinary spatial CNN is trained on exactly the same data at
   :math:`N=256` and evaluated on the same finer grids.  Without it, stable errors could
   just mean the problem is insensitive to resolution.
2. **A fixed continuum problem.**  The prior and the noise profile are functions of the
   absolute mode index, so refining the grid adds modes to the *same* continuum problem
   instead of substituting a different one.  Errors are relative :math:`L^2` norms computed
   from coefficients, which by Parseval are grid-independent (see :mod:`specinv.basis`).
3. **Two apertures.**  With the truncation index of Eq. (6) held at its trained value
   (``scnet``), invariance is structural: the learned map acts on the leading 256
   coefficients and a finer mesh only changes the synthesis.  That is the paper's claim,
   and reporting it alone would overstate the evidence, so we also evaluate with the
   filter applied to *every* mode the finer grid provides (``scnet_full_aperture``).  The
   second setting forces the network to extrapolate to singular values four decades below
   anything it trained on, and to reject them unprompted; at :math:`N=2048` a leak of
   :math:`10^{-3}` in the damping would be enough to destroy the reconstruction.

The paper does not state which :math:`\\delta` §5.4 used, so we report the whole table at
the round value :math:`\\delta = 0.1` and, for reference, at the :math:`\\delta` recovered by
matching the paper's training-grid error.

Usage
-----
    python -m specinv.experiments.zero_shot --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..basis import SineBasis
from ..baselines import SpatialCNN, SpatialCNNConfig, train_spatial_cnn
from ..metrics import summarise_errors
from ..scnet import SCNetConfig
from ..training import TrainConfig
from ._common import (
    METHOD_LABELS,
    PAPER_REFERENCE,
    add_common_arguments,
    apply_quick,
    build_suite,
    environment_info,
    evaluate_full_aperture,
    evaluate_methods,
    output_dir,
    train_model,
    write_csv,
    write_json,
)

RESOLUTIONS: tuple[int, ...] = (256, 512, 1024, 2048)
TRAIN_RESOLUTION = 256


def evaluate_at_resolutions(
    model: Any,
    spatial: SpatialCNN | None,
    suite: Any,
    delta: float,
    resolutions: tuple[int, ...],
    n_test: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, float]]]:
    """Evaluate every method at every resolution on matched realisations."""
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[int, float]] = {}

    for n_modes in resolutions:
        rng = np.random.default_rng(seed + 777)
        batch = suite.sample(n_test, delta, rng, n_modes=n_modes)
        outputs = evaluate_methods(model, suite, batch)
        summaries = outputs.summarise(batch.true_coefficients)

        # Same weights, aperture widened to every available mode (see module docstring).
        summaries["scnet_full_aperture"] = evaluate_full_aperture(model, suite, batch)

        if spatial is not None:
            basis = SineBasis(n_modes)
            prediction = spatial.reconstruct_coefficients(batch.noisy_data, basis)
            summaries["spatial_cnn"] = summarise_errors(
                prediction, batch.true_coefficients
            )

        realised = float(np.mean(batch.realised_noise_level))
        for method, summary in summaries.items():
            curves.setdefault(method, {})[n_modes] = summary.mean_relative
            rows.append(
                {
                    "delta_per_mode": delta,
                    "relative_l2_noise": realised,
                    "resolution": n_modes,
                    "method": method,
                    "trained_resolution": TRAIN_RESOLUTION,
                    **summary.as_dict(),
                }
            )
        print(
            f"    N={n_modes:<5d} "
            + "  ".join(f"{m}={s.mean_relative:.4f}" for m, s in summaries.items())
        )
    return rows, curves


def calibrated_delta(
    prefactor: float, slope: float, target_error: float
) -> float:
    """Invert :math:`e = C\\delta^{\\text{slope}}` for :math:`\\delta`.

    Used only to report the table at the noise level implied by the paper's own number,
    since §5.4 does not state :math:`\\delta`.
    """
    return float((target_error / prefactor) ** (1.0 / slope))


def make_figure(payload: dict[str, Any], path: Path) -> None:
    """Figure 3: error against resolution for SC-Net and the spatial control."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable, skipping figure")
        return

    curves = payload["primary"]["curves"]
    resolutions = payload["resolutions"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    styles: list[tuple[str, dict[str, Any]]] = [
        ("scnet", dict(color="#1f4e9c", marker="o", lw=2.2)),
        ("scnet_full_aperture", dict(color="#3f8ee0", marker="*", ls="-", lw=1.6, ms=9)),
        ("oracle_tikhonov", dict(color="#e07b26", marker="s", ls="--")),
        ("oracle_tsvd", dict(color="#2f7d3a", marker="^", ls="-.")),
    ]
    for method, style in styles:
        if method in curves:
            values = [curves[method][n] for n in resolutions]
            ax.plot(resolutions, values, label=METHOD_LABELS.get(method, method), **style)

    ax.axhline(
        PAPER_REFERENCE["zero_shot_error_n256"],
        color="k",
        ls=":",
        lw=1.2,
        label="paper, $N=256$ (0.2415)",
    )
    ax.axhline(
        PAPER_REFERENCE["zero_shot_error_n2048"],
        color="gray",
        ls=":",
        lw=1.2,
        label="paper, $N=2048$ (0.2292)",
    )
    ax.axvline(TRAIN_RESOLUTION, color="#1f4e9c", alpha=0.2, lw=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(n) for n in resolutions])
    ax.set_xlabel("evaluation resolution $N$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(f"Zero-shot transfer, $\\delta={payload['primary']['delta']}$")
    ax.set_ylim(0.0, 0.45)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")

    if "spatial_cnn" in curves:
        ax2.plot(
            resolutions,
            [curves["scnet"][n] for n in resolutions],
            "o-",
            color="#1f4e9c",
            lw=2.2,
            label="SC-Net (spectral)",
        )
        ax2.plot(
            resolutions,
            [curves["spatial_cnn"][n] for n in resolutions],
            "s--",
            color="#b02418",
            lw=2.0,
            label="Spatial CNN (fixed grid)",
        )
        ax2.set_yscale("log")
        ax2.set_xscale("log", base=2)
        ax2.set_xticks(resolutions)
        ax2.set_xticklabels([str(n) for n in resolutions])
        ax2.axvline(TRAIN_RESOLUTION, color="#1f4e9c", alpha=0.2, lw=8)
        ax2.set_xlabel("evaluation resolution $N$")
        ax2.set_ylabel(r"relative $L^2$ error (log)")
        ax2.set_title("Operator learning vs. fixed-grid learning")
        ax2.grid(True, which="both", alpha=0.25)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    parser.add_argument(
        "--delta", type=float, default=0.1, help="relative noise level (default: 0.1)"
    )
    parser.add_argument(
        "--skip-cnn", action="store_true", help="skip the spatial CNN control"
    )
    args = apply_quick(parser.parse_args(argv))

    directory = output_dir(args.results_dir)
    figures = output_dir(directory / "figures")

    suite = build_suite(n_modes=max(RESOLUTIONS))
    train_config = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_test,
        n_modes=TRAIN_RESOLUTION,
        delta_range=(1e-3, 1e-1),
        epochs=args.epochs,
        seed=args.seed,
        gamma=0.0,
    )

    print(f"\n[zero-shot] training SC-Net at N={TRAIN_RESOLUTION}")
    model, train_summary = train_model(suite, train_config, SCNetConfig(), verbose=True)

    spatial: SpatialCNN | None = None
    spatial_summary: dict[str, Any] | None = None
    if not args.skip_cnn:
        print(f"\n[zero-shot] training spatial CNN control at N={TRAIN_RESOLUTION}")
        spatial = SpatialCNN(SpatialCNNConfig())
        history = train_spatial_cnn(
            spatial, suite, train_config, progress=lambda m: print(f"    {m}", flush=True)
        )
        spatial_summary = {
            "n_parameters": spatial.n_parameters(),
            "best_val_relative_l2": history.best_val_error,
            "wall_time_seconds": history.wall_time,
        }

    print(f"\n[zero-shot] evaluating at delta={args.delta}")
    rows, curves = evaluate_at_resolutions(
        model, spatial, suite, args.delta, RESOLUTIONS, args.n_test, args.seed
    )

    scnet_curve = curves["scnet"]
    base = scnet_curve[TRAIN_RESOLUTION]
    finest = scnet_curve[max(RESOLUTIONS)]

    # Sec. 5.4 does not state its noise level, so additionally report the table at the
    # delta implied by the paper's own training-grid error, inverting e = C*delta^slope
    # with the measured error at args.delta fixing C.
    matched: dict[str, Any] | None = None
    slope_estimate = 0.49
    fine_delta = calibrated_delta(
        base / args.delta**slope_estimate,
        slope_estimate,
        PAPER_REFERENCE["zero_shot_error_n256"],
    )
    if 1e-3 <= fine_delta <= 1e-1:
        print(
            f"\n[zero-shot] evaluating at delta={fine_delta:.4f} "
            "(matched to the paper's N=256 error of 0.2415)"
        )
        matched_rows, matched_curves = evaluate_at_resolutions(
            model, spatial, suite, fine_delta, RESOLUTIONS, args.n_test, args.seed
        )
        rows += matched_rows
        matched = {"delta": fine_delta, "curves": matched_curves}

    payload: dict[str, Any] = {
        "experiment": "zero_shot_resolution_transfer",
        "paper_section": "5.4",
        "environment": environment_info(),
        "paper_reference": PAPER_REFERENCE,
        "resolutions": list(RESOLUTIONS),
        "trained_resolution": TRAIN_RESOLUTION,
        "training": train_summary,
        "spatial_cnn_training": spatial_summary,
        "primary": {"delta": args.delta, "curves": curves},
        "matched_to_paper": matched,
        "summary": {
            "scnet_error_at_train_resolution": base,
            "scnet_error_at_finest_resolution": finest,
            "relative_drift_train_to_finest": abs(finest - base) / base,
            "paper_error_n256": PAPER_REFERENCE["zero_shot_error_n256"],
            "paper_error_n2048": PAPER_REFERENCE["zero_shot_error_n2048"],
        },
    }
    full = curves["scnet_full_aperture"]
    payload["summary"]["scnet_full_aperture_error_at_train_resolution"] = full[
        TRAIN_RESOLUTION
    ]
    payload["summary"]["scnet_full_aperture_error_at_finest_resolution"] = full[
        max(RESOLUTIONS)
    ]
    payload["summary"]["scnet_full_aperture_drift"] = (
        abs(full[max(RESOLUTIONS)] - full[TRAIN_RESOLUTION]) / full[TRAIN_RESOLUTION]
    )
    if spatial is not None:
        cnn = curves["spatial_cnn"]
        payload["summary"]["spatial_cnn_error_at_train_resolution"] = cnn[TRAIN_RESOLUTION]
        payload["summary"]["spatial_cnn_error_at_finest_resolution"] = cnn[max(RESOLUTIONS)]
        payload["summary"]["spatial_cnn_degradation_factor"] = (
            cnn[max(RESOLUTIONS)] / cnn[TRAIN_RESOLUTION]
        )

    write_json(directory / "zero_shot.json", payload)
    write_csv(directory / "zero_shot.csv", rows)
    if not args.no_figures:
        make_figure(payload, figures / "fig3_zero_shot.png")

    print(
        f"\nSC-Net: {base:.4f} at N={TRAIN_RESOLUTION} -> {finest:.4f} at "
        f"N={max(RESOLUTIONS)} (drift {100*abs(finest-base)/base:.1f}%); "
        f"paper: 0.2415 -> 0.2292"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())