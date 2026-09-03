"""§5.2 -- convergence order in the noise level.

Reproduces Figure 1: relative :math:`L^2` error against :math:`\\delta` on a log-log scale,
for SC-Net and the Oracle baselines, and reports the fitted slope.

The paper's claim is that SC-Net attains :math:`O(\\delta^{s/(s+p)}) = O(\\delta^{0.5})` for
:math:`s=p=1.5`.  We run the sweep three ways:

* on the paper's noise grid :math:`\\{10^{-1}, 5\\cdot10^{-2}, 10^{-2}, 5\\cdot10^{-3},
  10^{-3}\\}` under the ``critical`` noise model -- the direct comparison;
* on an *extended* grid reaching :math:`10^{-8}`, where the asymptotic regime is actually
  entered.  On the paper's grid the optimal truncation index is only 2-12 modes, so its
  integer quantisation biases any fitted slope downwards; the extended sweep is what shows
  the order converging to 0.5.
* under the ``white_energy`` noise model, which yields the *statistical* order
  :math:`s/(s+p+1/2) \\approx 0.43` for every method.  This is reported because it explains
  the paper's own Oracle Tikhonov slope of 0.42.

Usage
-----
    python -m specinv.experiments.convergence --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import fit_rate
from ..problems import NoiseModel
from ..scnet import SCNetConfig
from ..theory import deterministic_rate, optimal_truncation_index, statistical_rate
from ..training import TrainConfig
from ._common import (
    METHOD_LABELS,
    PAPER_DELTAS,
    PAPER_REFERENCE,
    add_common_arguments,
    apply_quick,
    build_suite,
    environment_info,
    evaluate_methods,
    output_dir,
    train_model,
    write_csv,
    write_json,
)

EXTENDED_DELTAS: tuple[float, ...] = (
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    3e-6,
    1e-6,
    3e-7,
    1e-7,
    3e-8,
    1e-8,
)


def run_sweep(
    name: str,
    noise_model: NoiseModel,
    deltas: tuple[float, ...],
    n_modes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train one model and sweep it over ``deltas``."""
    print(f"\n[{name}] noise_model={noise_model.value} resolution={n_modes}")
    suite = build_suite(n_modes=max(n_modes, 2048), noise_model=noise_model)

    delta_min, delta_max = min(deltas), max(deltas)
    band = suite.recommended_noise_band(delta_min, n_modes)
    print(
        f"    signal/noise crossover at delta={delta_min:g} is mode "
        f"{suite.signal_noise_crossover(delta_min):.0f}; noise band {band}"
    )

    train_config = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_test,
        n_modes=n_modes,
        delta_range=(delta_min, delta_max),
        epochs=args.epochs,
        seed=args.seed,
        gamma=0.0,
    )
    model, train_summary = train_model(
        suite, train_config, SCNetConfig(noise_band=band), verbose=True
    )

    rows: list[dict[str, Any]] = []
    per_method: dict[str, list[float]] = {}
    realised_levels: list[float] = []
    for delta in deltas:
        rng = np.random.default_rng(args.seed + 10_000)
        batch = suite.sample(args.n_test, delta, rng, n_modes=n_modes)
        outputs = evaluate_methods(model, suite, batch)
        summaries = outputs.summarise(batch.true_coefficients)
        realised = float(np.mean(batch.realised_noise_level))
        realised_levels.append(realised)
        for method, summary in summaries.items():
            per_method.setdefault(method, []).append(summary.mean_relative)
            rows.append(
                {
                    "sweep": name,
                    "noise_model": noise_model.value,
                    "delta_per_mode": delta,
                    "relative_l2_noise": realised,
                    "method": method,
                    **summary.as_dict(),
                }
            )
        print(
            f"    delta={delta:<9g} "
            + "  ".join(
                f"{m}={s.mean_relative:.4f}" for m, s in summaries.items()
            )
        )

    fits = {
        method: fit_rate(np.asarray(deltas), np.asarray(errors)).as_dict()
        for method, errors in per_method.items()
    }
    print("    fitted slopes: " + "  ".join(f"{m}={f['slope']:.4f}" for m, f in fits.items()))

    return {
        "noise_model": noise_model.value,
        "resolution": n_modes,
        "deltas": list(deltas),
        "delta_definition": (
            "per-mode noise scale: std(eps_n) = delta * ||y|| * n^(-q) with q the "
            "colouring exponent of the noise model"
        ),
        "relative_l2_noise": realised_levels,
        "noise_band": list(band),
        "training": train_summary,
        "errors": {m: e for m, e in per_method.items()},
        "rate_fits": fits,
        "rows": rows,
        "predicted_rate": suite.noise_model.observable_rate(
            suite.smoothness, suite.ill_posedness
        ),
        "optimal_truncation_index": optimal_truncation_index(
            np.asarray(deltas), suite.smoothness, suite.ill_posedness
        ).tolist(),
    }


def make_figure(payload: dict[str, Any], path: Path) -> None:
    """Figure 1: log-log convergence plot for the paper's noise grid."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable, skipping figure")
        return

    sweep = payload["sweeps"]["paper_grid"]
    deltas = np.asarray(sweep["deltas"])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    styles = {
        "scnet": dict(color="#1f4e9c", marker="o", lw=2.2, zorder=5),
        "oracle_tikhonov": dict(color="#e07b26", marker="s", ls="--", lw=1.8),
        "oracle_tsvd": dict(color="#2f7d3a", marker="^", ls="-.", lw=1.6),
        "tikhonov_discrepancy": dict(color="#8a8a8a", marker="v", ls=":", lw=1.4),
        "prior_wiener": dict(color="#7a4fa3", marker="d", ls="--", lw=1.4),
        "oracle_spectral_bound": dict(color="#b02418", marker="x", ls=":", lw=1.4),
    }
    for method, errors in sweep["errors"].items():
        slope = sweep["rate_fits"][method]["slope"]
        ax.loglog(
            deltas,
            errors,
            label=f"{METHOD_LABELS.get(method, method)} [slope {slope:.3f}]",
            **styles.get(method, {}),
        )

    reference = 0.77 * deltas ** payload["theory"]["deterministic_rate"]
    ax.loglog(deltas, reference, color="k", lw=1.0, alpha=0.5, label=r"$\propto\delta^{0.5}$ (theory)")

    ax.set_xlabel(r"relative noise level $\delta$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("Convergence order (critical noise model, $s=p=1.5$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def make_extended_figure(payload: dict[str, Any], path: Path) -> None:
    """Slope-vs-window plot showing the order approaching 0.5 as delta shrinks."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if "extended_grid" not in payload["sweeps"]:
        return

    sweep = payload["sweeps"]["extended_grid"]
    deltas = np.asarray(sweep["deltas"])
    errors = np.asarray(sweep["errors"]["scnet"])
    window = 5
    centres, slopes = [], []
    for start in range(len(deltas) - window + 1):
        sl = slice(start, start + window)
        centres.append(float(np.exp(np.mean(np.log(deltas[sl])))))
        slopes.append(fit_rate(deltas[sl], errors[sl]).slope)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.semilogx(centres, slopes, "o-", color="#1f4e9c", label="SC-Net local slope")
    ax.axhline(
        payload["theory"]["deterministic_rate"],
        color="k",
        ls="--",
        label=r"theory $s/(s+p)=0.5$",
    )
    ax.axhline(
        payload["theory"]["statistical_rate"],
        color="#b02418",
        ls=":",
        label=r"white-noise order $s/(s+p+1/2)\approx0.43$",
    )
    ax.set_xlabel(r"window centre in $\delta$")
    ax.set_ylabel("local convergence order")
    ax.set_title("Convergence order approaches 0.5 as $\\delta\\to0$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    )
    parser.add_argument(
        "--skip-extended",
        action="store_true",
        help="skip the extended (small-delta) sweep",
    )
    parser.add_argument(
        "--skip-white",
        action="store_true",
        help="skip the white-noise-model sweep",
    )
    args = apply_quick(parser.parse_args(argv))

    directory = output_dir(args.results_dir)
    figures = output_dir(directory / "figures")

    payload: dict[str, Any] = {
        "experiment": "convergence_rate",
        "paper_section": "5.2",
        "environment": environment_info(),
        "paper_reference": PAPER_REFERENCE,
        "theory": {
            "deterministic_rate": deterministic_rate(1.5, 1.5),
            "statistical_rate": statistical_rate(1.5, 1.5),
            "smoothness": 1.5,
            "ill_posedness": 1.5,
        },
        "sweeps": {},
    }

    payload["sweeps"]["paper_grid"] = run_sweep(
        "paper_grid", NoiseModel.CRITICAL, PAPER_DELTAS, 256, args
    )
    if not args.skip_extended:
        payload["sweeps"]["extended_grid"] = run_sweep(
            "extended_grid", NoiseModel.CRITICAL, EXTENDED_DELTAS, 2048, args
        )
    if not args.skip_white:
        payload["sweeps"]["white_noise"] = run_sweep(
            "white_noise", NoiseModel.WHITE_ENERGY, PAPER_DELTAS, 256, args
        )

    rows = [row for sweep in payload["sweeps"].values() for row in sweep["rows"]]
    for sweep in payload["sweeps"].values():
        sweep.pop("rows")

    write_json(directory / "convergence.json", payload)
    write_csv(directory / "convergence.csv", rows)
    if not args.no_figures:
        make_figure(payload, figures / "fig1_convergence.png")
        make_extended_figure(payload, figures / "fig1b_local_slope.png")

    scnet_slope = payload["sweeps"]["paper_grid"]["rate_fits"]["scnet"]["slope"]
    print(
        f"\nSC-Net order on the paper's grid: {scnet_slope:.4f} "
        f"(paper 0.50, theory {deterministic_rate(1.5, 1.5):.2f})"
    )
    if "extended_grid" in payload["sweeps"]:
        ext = payload["sweeps"]["extended_grid"]["rate_fits"]["scnet"]["slope"]
        print(f"SC-Net order on the extended grid:  {ext:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())