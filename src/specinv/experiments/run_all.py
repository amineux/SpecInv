"""Run every reproduction script and check the paper's claims against the measurements.

Writes ``results/summary.json``, whose ``criteria`` block states, for each claim of
arXiv:2603.20602 that this repository set out to reproduce, the measured value, the
threshold applied, and whether it passed.  The exit status is non-zero if any criterion
fails, so this doubles as an end-to-end regression check.

Usage
-----
    python -m specinv.experiments.run_all --results-dir results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ablations, convergence, filters, zero_shot
from ._common import PAPER_REFERENCE, add_common_arguments, apply_quick, output_dir


@dataclass(frozen=True)
class Criterion:
    """One reproduction target, its measurement and its verdict."""

    name: str
    description: str
    measured: float | bool
    target: str
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "measured": self.measured,
            "target": self.target,
            "passed": self.passed,
        }


def build_criteria(
    convergence_payload: dict[str, Any],
    zero_shot_payload: dict[str, Any],
    filters_payload: dict[str, Any],
) -> list[Criterion]:
    """Evaluate the reproduction targets from the experiment outputs."""
    paper_grid = convergence_payload["sweeps"]["paper_grid"]
    scnet_slope = paper_grid["rate_fits"]["scnet"]["slope"]
    theory = convergence_payload["theory"]["deterministic_rate"]

    extended = convergence_payload["sweeps"].get("extended_grid")
    extended_slope = (
        extended["rate_fits"]["scnet"]["slope"] if extended is not None else scnet_slope
    )

    zs = zero_shot_payload["summary"]
    base = zs["scnet_error_at_train_resolution"]
    finest = zs["scnet_error_at_finest_resolution"]
    drift = zs["relative_drift_train_to_finest"]

    matched = zero_shot_payload.get("matched_to_paper")
    matched_finest = (
        matched["curves"]["scnet"][str(max(zero_shot_payload["resolutions"]))]
        if matched
        else finest
    )

    beats_tikhonov = all(
        s < t
        for s, t in zip(
            paper_grid["errors"]["scnet"],
            paper_grid["errors"]["oracle_tikhonov"],
            strict=True,
        )
    )
    worst_margin = min(
        (t - s) / t
        for s, t in zip(
            paper_grid["errors"]["scnet"],
            paper_grid["errors"]["oracle_tikhonov"],
            strict=True,
        )
    )
    checks = filters_payload["claim_checks"]

    return [
        Criterion(
            "convergence_order_paper_grid",
            "Fitted order of SC-Net on the paper's noise grid is within 0.05 of s/(s+p)=0.5",
            scnet_slope,
            f"|slope - {theory:.2f}| <= 0.05",
            abs(scnet_slope - theory) <= 0.05,
        ),
        Criterion(
            "convergence_order_asymptotic",
            "Fitted order on the extended grid (delta down to 1e-8) is within 0.02 of 0.5",
            extended_slope,
            f"|slope - {theory:.2f}| <= 0.02",
            abs(extended_slope - theory) <= 0.02,
        ),
        Criterion(
            "zero_shot_error_magnitude",
            "Zero-shot error at N=2048 matches the paper's 0.2292 within 15%",
            matched_finest,
            f"|e - {PAPER_REFERENCE['zero_shot_error_n2048']}| / "
            f"{PAPER_REFERENCE['zero_shot_error_n2048']} <= 0.15",
            abs(matched_finest - PAPER_REFERENCE["zero_shot_error_n2048"])
            / PAPER_REFERENCE["zero_shot_error_n2048"]
            <= 0.15,
        ),
        Criterion(
            "zero_shot_stability",
            "Error drifts by <5% from the N=256 training grid to N=2048",
            drift,
            "drift <= 0.05",
            drift <= 0.05,
        ),
        Criterion(
            "zero_shot_stability_full_aperture",
            "With the aperture widened to all 2048 modes, the error still drifts by <15%",
            zs["scnet_full_aperture_drift"],
            "drift <= 0.15",
            zs["scnet_full_aperture_drift"] <= 0.15,
        ),
        Criterion(
            "beats_oracle_tikhonov",
            "SC-Net beats Oracle Tikhonov at every noise level on the suite",
            beats_tikhonov,
            "all(scnet < oracle_tikhonov)",
            bool(beats_tikhonov),
        ),
        Criterion(
            "beats_oracle_tikhonov_margin",
            "Worst-case relative margin over Oracle Tikhonov across noise levels",
            worst_margin,
            "margin > 0",
            worst_margin > 0.0,
        ),
        Criterion(
            "filter_is_interpretable",
            "Learned filter is a bounded damping profile in [0,1] that can be tabulated",
            checks["filter_is_bounded_in_unit_interval"],
            "True",
            bool(checks["filter_is_bounded_in_unit_interval"]),
        ),
        Criterion(
            "filter_preserves_low_modes",
            "Learned damping exceeds 0.9 on the leading mode at every noise level",
            checks["preserves_leading_modes"],
            "True",
            bool(checks["preserves_leading_modes"]),
        ),
        Criterion(
            "filter_suppresses_high_modes",
            "Learned damping is below 0.05 at mode 32 at every noise level",
            checks["suppresses_high_modes"],
            "True",
            bool(checks["suppresses_high_modes"]),
        ),
        Criterion(
            "filter_sharper_than_tikhonov",
            "Learned transition band is narrower than Tikhonov's at every noise level",
            checks["sharper_than_tikhonov"],
            "True",
            bool(checks["sharper_than_tikhonov"]),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = add_common_arguments(
        argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
    )
    parser.add_argument(
        "--skip-ablations", action="store_true", help="skip the ablation study"
    )
    args = apply_quick(parser.parse_args(argv))
    directory = output_dir(args.results_dir)

    forwarded = [
        "--results-dir", str(args.results_dir),
        "--epochs", str(args.epochs),
        "--n-train", str(args.n_train),
        "--n-test", str(args.n_test),
        "--seed", str(args.seed),
    ]
    if args.no_figures:
        forwarded.append("--no-figures")

    print("=" * 78)
    print("SpecInv: reproducing arXiv:2603.20602 (SC-Net)")
    print("=" * 78)

    convergence.main(forwarded)
    zero_shot.main(forwarded)
    filters.main(forwarded)
    if not args.skip_ablations:
        ablations.main(forwarded)

    def load(name: str) -> dict[str, Any]:
        return json.loads((Path(args.results_dir) / name).read_text())

    criteria = build_criteria(
        load("convergence.json"), load("zero_shot.json"), load("filters.json")
    )

    summary = {
        "paper": "arXiv:2603.20602",
        "paper_title": (
            "Interpretable Operator Learning for Inverse Problems via Adaptive Spectral "
            "Filtering: Convergence and Discretization Invariance"
        ),
        "paper_reference_values": PAPER_REFERENCE,
        "criteria": [c.as_dict() for c in criteria],
        "all_passed": all(c.passed for c in criteria),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 78)
    print("REPRODUCTION CRITERIA")
    print("=" * 78)
    for c in criteria:
        measured = f"{c.measured:.4f}" if isinstance(c.measured, float) else str(c.measured)
        print(f"  [{'PASS' if c.passed else 'FAIL'}]  {c.name:<34s} {measured:>10s}   ({c.target})")
    print("=" * 78)
    print(f"  wrote {directory / 'summary.json'}")

    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())