"""Ablations: which parts of the implementation the reported results actually depend on.

Not in the paper.  Several of the choices in :mod:`specinv.scnet` are things the paper
leaves unspecified, and it would be dishonest to report headline numbers without showing
how much they rest on those choices.  Each row changes exactly one thing:

``reference``
    The configuration used for the headline results.
``paper_features``
    :math:`\\Psi_\\theta(y_n,\\sigma_n)` with the raw inputs of §3.1.2, standardised only.
``no_noise_estimate``
    Drops the data-driven noise-floor features, leaving a filter that can only see
    :math:`(\\sigma_n, |y_n|)` -- the literal information content of Eq. (5).
``no_feature_clamp``
    Removes the extrapolation guard.  Expected to leave the in-distribution numbers intact
    and break the zero-shot transfer, which is exactly the point of the guard.
``sobolev_gamma=*``
    The weight :math:`\\gamma` of Eq. (7).  Since the reported metric is an :math:`L^2`
    error and the gradient term optimises an :math:`H^1` objective, :math:`\\gamma>0` can
    only trade the reported number away; this quantifies by how much.
``heavy_tail_prior``
    Replaces the Gaussian amplitudes by Student-:math:`t`.  Under a Gaussian prior the
    pointwise Bayes-optimal filter is *linear* in the observation, so no filter reading
    :math:`y_n` can beat the Wiener ceiling; under a heavy tail it can, and SC-Net does.
    This is the cleanest evidence that the learned filter is genuinely adaptive rather
    than a re-derivation of Wiener.

Usage
-----
    python -m specinv.experiments.ablations --results-dir results
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from ..metrics import summarise_errors
from ..scnet import SCNetConfig
from ..training import TrainConfig
from ._common import (
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

EVAL_DELTAS: tuple[float, ...] = (0.1, 0.01)
TRAIN_RESOLUTION = 256
TRANSFER_RESOLUTION = 2048


def variants() -> dict[str, tuple[SCNetConfig, dict[str, Any], str]]:
    """Name -> (net config, train-config overrides, suite overrides)."""
    return {
        "reference": (SCNetConfig(), {}, "gaussian"),
        "paper_features": (SCNetConfig(feature_set="paper"), {}, "gaussian"),
        "no_noise_estimate": (SCNetConfig(use_noise_estimate=False), {}, "gaussian"),
        "no_feature_clamp": (SCNetConfig(clamp_features=False), {}, "gaussian"),
        "sobolev_gamma_0.01": (SCNetConfig(), {"gamma": 0.01}, "gaussian"),
        "sobolev_gamma_0.1": (SCNetConfig(), {"gamma": 0.1}, "gaussian"),
        "sobolev_gamma_1.0": (SCNetConfig(), {"gamma": 1.0}, "gaussian"),
        "heavy_tail_prior": (SCNetConfig(), {}, "heavy_tail"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    args = apply_quick(parser.parse_args(argv))
    directory = output_dir(args.results_dir)

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for name, (net_config, overrides, amplitude) in variants().items():
        print(f"\n[ablation] {name}")
        suite = build_suite(n_modes=TRANSFER_RESOLUTION, amplitude=amplitude)
        train_config = TrainConfig(
            n_train=args.n_train,
            n_val=args.n_test,
            n_modes=TRAIN_RESOLUTION,
            delta_range=(1e-3, 1e-1),
            epochs=args.epochs,
            seed=args.seed,
            gamma=float(overrides.get("gamma", 0.0)),
        )
        model, train_summary = train_model(suite, train_config, net_config, verbose=False)

        entry: dict[str, Any] = {"training": train_summary, "errors": {}}
        for delta in EVAL_DELTAS:
            for resolution in (TRAIN_RESOLUTION, TRANSFER_RESOLUTION):
                rng = np.random.default_rng(args.seed + 555)
                batch = suite.sample(args.n_test, delta, rng, n_modes=resolution)
                outputs = evaluate_methods(
                    model, suite, batch, methods=["scnet", "oracle_tikhonov", "prior_wiener"]
                )
                summaries = outputs.summarise(batch.true_coefficients)
                summaries["scnet_full_aperture"] = evaluate_full_aperture(
                    model, suite, batch
                )
                key = f"delta={delta:g},N={resolution}"
                entry["errors"][key] = {
                    m: s.mean_relative for m, s in summaries.items()
                }
                rows.append(
                    {
                        "variant": name,
                        "delta": delta,
                        "resolution": resolution,
                        **{f"{m}_relative_l2": s.mean_relative for m, s in summaries.items()},
                        "beats_oracle_tikhonov": bool(
                            summaries["scnet"].mean_relative
                            < summaries["oracle_tikhonov"].mean_relative
                        ),
                    }
                )
                print(
                    f"    delta={delta:<7g} N={resolution:<5d} "
                    + "  ".join(f"{m}={s.mean_relative:.4f}" for m, s in summaries.items())
                )
        base = entry["errors"][f"delta=0.1,N={TRAIN_RESOLUTION}"]["scnet"]
        fine = entry["errors"][f"delta=0.1,N={TRANSFER_RESOLUTION}"]["scnet"]
        entry["zero_shot_drift"] = abs(fine - base) / base
        # The extrapolation guard only bites when the aperture is widened, so the drift
        # that discriminates between variants is the full-aperture one.
        base_full = entry["errors"][f"delta=0.1,N={TRAIN_RESOLUTION}"]["scnet_full_aperture"]
        fine_full = entry["errors"][f"delta=0.1,N={TRANSFER_RESOLUTION}"][
            "scnet_full_aperture"
        ]
        entry["zero_shot_drift_full_aperture"] = abs(fine_full - base_full) / base_full
        results[name] = entry

    payload = {
        "experiment": "ablations",
        "paper_section": "n/a (implementation study)",
        "environment": environment_info(),
        "train_resolution": TRAIN_RESOLUTION,
        "transfer_resolution": TRANSFER_RESOLUTION,
        "eval_deltas": list(EVAL_DELTAS),
        "variants": results,
    }
    write_json(directory / "ablations.json", payload)
    write_csv(directory / "ablations.csv", rows)

    print("\nzero-shot drift (N=256 -> N=2048) at delta=0.1:")
    print(f"  {'variant':<22s} {'trained aperture':>17s} {'full aperture':>15s}")
    for name, entry in results.items():
        print(
            f"  {name:<22s} {100*entry['zero_shot_drift']:16.2f}% "
            f"{100*entry['zero_shot_drift_full_aperture']:14.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())