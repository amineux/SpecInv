"""One joint path: KerOp forward spectrum → SpecInv spectral filter.

Team verdict (locked): keep this path only. KerOp exports a forward spectrum;
SpecInv consumes it and emits the learned filter. This script runs **both**
bars on that path and prints red / green honestly.

It is not a new theorem and it does not invent a factor (no 100x language).

Resolution of the spectrum artifact
-----------------------------------
1. ``--spectrum PATH``
2. live export from an imported KerOp (sibling checkout, ``--kerop-root``, or
   ``KEROP_ROOT``)
3. the committed contract fixture ``fixtures/kerop_spectrum_v1.npz``

KerOp **must** be importable. If it is not, this script exits non-zero and
does **not** fall back to ``power_law_operator`` / the old 1-D suite.

Usage
-----
    python -m specinv.experiments.joint --results-dir results/joint
    python -m specinv.experiments.joint --quick --results-dir /tmp/joint-quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..filters import oracle_tikhonov
from ..metrics import fit_rate, summarise_errors
from ..problems import InverseProblemSuite, SobolevPrior
from ..scnet import SCNetConfig
from ..spectrum_contract import (
    JOINT_B,
    JOINT_N_MODES,
    JOINT_OUTPUT_DIM,
    JOINT_R,
    JOINT_SEED,
    JointPathError,
    SpectrumArtifact,
    default_fixture_path,
    export_kerop_spectrum,
    fingerprint,
    load_spectrum_artifact,
    operator_from_artifact,
    require_kerop,
    write_spectrum_artifact,
)
from ..training import TrainConfig
from . import convergence, filters, run_all, zero_shot
from ._common import apply_quick, environment_info, output_dir, train_model, write_json

# KerOp's committed spectral-quick median (results/walltime_spectral.json).
# Recorded so this path cannot silently drop that bar. Not a new claim.
KEROP_PUBLISHED_MEDIAN_SPEEDUP = 27.71727018410563
# Fail the KerOp bar if the matched-risk win on this path falls out of class.
KEROP_MIN_MEDIAN_SPEEDUP = 8.0

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty()


def _paint(label: str, color: str) -> str:
    if not _use_color():
        return label
    return f"{color}{label}{_RESET}"


def _print_verdict(passed: bool, name: str, detail: str, *, graded: bool = True) -> None:
    if passed:
        tag = _paint("GREEN", _GREEN)
    elif graded:
        tag = _paint("RED", _RED)
    else:
        tag = _paint("AMBER", _YELLOW)
    print(f"  [{tag}]  {name}: {detail}", flush=True)


def resolve_spectrum(args: argparse.Namespace) -> tuple[SpectrumArtifact, Path, str]:
    """Load or export the KerOp artifact. ``require_kerop`` has already run."""
    if args.spectrum is not None:
        path = Path(args.spectrum)
        artifact = load_spectrum_artifact(path)
        return artifact, path, "path"

    fixture = default_fixture_path()
    if args.use_fixture:
        if not fixture.is_file():
            raise JointPathError(f"--use-fixture set but fixture missing: {fixture}")
        return load_spectrum_artifact(fixture), fixture, "committed_fixture"

    # Prefer a live export so the bytes are KerOp's, not a static table.
    live = export_kerop_spectrum(kerop_root=args.kerop_root)
    dest = Path(args.results_dir) / "kerop_spectrum_v1.npz"
    write_spectrum_artifact(dest, live)
    if fixture.is_file():
        pinned = load_spectrum_artifact(fixture)
        if not np.allclose(pinned.eigenvalues, live.eigenvalues, rtol=1e-12, atol=1e-14):
            raise JointPathError(
                "committed fixture eigenvalues disagree with a live KerOp export "
                "of the same contract; regenerate the fixture with "
                "`python -m specinv.spectrum_contract`"
            )
    return live, dest, "kerop_export"


def _suite_from_artifact(artifact: SpectrumArtifact, n_modes: int) -> InverseProblemSuite:
    return InverseProblemSuite(
        operator=operator_from_artifact(artifact, n_modes=n_modes),
        prior=SobolevPrior(smoothness=1.5),
    )


def invoke_kerop_bar(
    artifact: SpectrumArtifact,
    *,
    quick: bool,
    kerop_root: str | Path | None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run KerOp's matched-excess-risk comparison on the **same** spectral model."""
    require_kerop(kerop_root)
    from kerop.experiments import run_walltime_benchmark

    n_modes = 64 if quick else 512
    n_modes = min(n_modes, artifact.n)
    task_kwargs: dict[str, Any] = {
        "r": artifact.r,
        "b": artifact.b,
        "n_modes": n_modes,
        "output_dim": artifact.output_dim,
        "n_test": 150 if quick else 1000,
    }
    payload = run_walltime_benchmark(
        task="spectral",
        train_sizes=(80, 120) if quick else (150, 300, 600),
        lambda_grid=(1e-2, 1e-3) if quick else (3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5),
        feature_multipliers=(1.0,) if quick else (0.5, 1.0, 2.0),
        iteration_grid=(64,) if quick else (64, 256),
        n_targets=3 if quick else 5,
        seed=int(artifact.seed),
        task_kwargs=task_kwargs,
        verbose=verbose,
    )
    verdict = payload["verdict"]
    median = verdict.get("median_speedup_at_matched_risk")
    faster = bool(verdict.get("random_features_faster_at_every_target"))
    if quick:
        # Smoke only: require a real matched-risk win, do not grade the ~28x bar.
        passed = bool(median is not None and median > 1.0 and faster)
        graded = False
        reason = (
            f"quick smoke median={median:.2f}x, RF faster at every target={faster}; "
            "the ~28x bar is not graded in --quick"
        )
    else:
        passed = bool(median is not None and float(median) >= KEROP_MIN_MEDIAN_SPEEDUP and faster)
        graded = True
        measured = "none" if median is None else f"{float(median):.2f}x"
        reason = (
            f"median matched-excess-risk speedup {measured} "
            f"(require >= {KEROP_MIN_MEDIAN_SPEEDUP:.1f}x; "
            f"KerOp published spectral-quick median is {KEROP_PUBLISHED_MEDIAN_SPEEDUP:.1f}x, "
            "not re-claimed here); "
            f"RF faster at every target={faster}"
        )
    return {
        "task": payload.get("settings", {}).get("task_name"),
        "quick": quick,
        "graded": graded,
        "passed": passed,
        "reason": reason,
        "verdict": verdict,
        "n_modes": n_modes,
        "operator_id": artifact.operator_id,
        "published_spectral_quick_median": KEROP_PUBLISHED_MEDIAN_SPEEDUP,
        "required_median_speedup": None if quick else KEROP_MIN_MEDIAN_SPEEDUP,
    }


def apply_filter_to_path_target(
    artifact: SpectrumArtifact,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train SC-Net on the artifact operator and invert one batch it generated."""
    n_modes = min(256, artifact.n)
    suite = _suite_from_artifact(artifact, artifact.n)
    train_config = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_test,
        n_modes=n_modes,
        delta_range=(1e-3, 1e-1),
        epochs=args.epochs,
        seed=args.seed,
        gamma=0.0,
    )
    print(
        f"\n[joint] training SC-Net on KerOp spectrum {artifact.operator_id} "
        f"(N={n_modes}, epochs={args.epochs})",
        flush=True,
    )
    model, train_summary = train_model(suite, train_config, SCNetConfig(), verbose=True)
    rng = np.random.default_rng(args.seed + 19)
    batch = suite.sample(args.n_test, 0.05, rng, n_modes=n_modes)
    sv = suite.operator.restrict(n_modes).singular_values
    reconstruction = model.reconstruct(batch.noisy_data, sv)
    damping = model.filter_profile(batch.noisy_data, sv)
    scnet = summarise_errors(reconstruction, batch.true_coefficients)
    tik = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
    tik_err = summarise_errors(tik.reconstruction, batch.true_coefficients)
    beats = bool(scnet.mean_relative < tik_err.mean_relative)
    return {
        "operator_id": artifact.operator_id,
        "fingerprint": fingerprint(artifact),
        "n_modes": n_modes,
        "n_test": args.n_test,
        "delta": 0.05,
        "training": train_summary,
        "scnet_mean_relative": scnet.mean_relative,
        "oracle_tikhonov_mean_relative": tik_err.mean_relative,
        "beats_oracle_tikhonov_on_path_target": beats,
        "filter_min": float(np.min(damping)),
        "filter_max": float(np.max(damping)),
        "filter_is_bounded": bool(np.min(damping) >= 0.0 and np.max(damping) <= 1.0 + 1e-12),
    }


def _quick_specinv_bars(
    artifact: SpectrumArtifact, args: argparse.Namespace, path_target: dict[str, Any]
) -> dict[str, Any]:
    """Reduced SpecInv checks. Must not be reported as 11/11."""
    suite = _suite_from_artifact(artifact, artifact.n)
    n_train_res = min(128, artifact.n)
    train_config = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_test,
        n_modes=n_train_res,
        delta_range=(1e-3, 1e-1),
        epochs=args.epochs,
        seed=args.seed,
        gamma=0.0,
    )
    model, _ = train_model(suite, train_config, SCNetConfig(), verbose=False)
    deltas = (1e-1, 1e-2, 1e-3)
    scnet_err = []
    tik_err = []
    for delta in deltas:
        rng = np.random.default_rng(args.seed + int(1.0 / delta))
        batch = suite.sample(args.n_test, delta, rng, n_modes=n_train_res)
        sv = suite.operator.restrict(n_train_res).singular_values
        rec = model.reconstruct(batch.noisy_data, sv)
        scnet_err.append(summarise_errors(rec, batch.true_coefficients).mean_relative)
        tik = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
        tik_err.append(summarise_errors(tik.reconstruction, batch.true_coefficients).mean_relative)
    slope = fit_rate(np.asarray(deltas), np.asarray(scnet_err)).slope
    beats = all(s < t for s, t in zip(scnet_err, tik_err, strict=True))
    fine = min(n_train_res * 2, artifact.n)
    rng = np.random.default_rng(args.seed + 777)
    fine_batch = suite.sample(args.n_test, 0.1, rng, n_modes=fine)
    sv_fine = suite.operator.restrict(fine).singular_values
    e_fine = summarise_errors(
        model.reconstruct(fine_batch.noisy_data, sv_fine), fine_batch.true_coefficients
    ).mean_relative
    e_coarse = scnet_err[0]
    drift = abs(e_fine - e_coarse) / max(e_coarse, 1e-12)
    return {
        "graded_11_of_11": False,
        "note": "quick mode is not the 11/11 bar; do not cite these numbers as a joint result",
        "rate_slope_three_point": slope,
        "beats_oracle_tikhonov": beats,
        "zero_shot_drift_2x": drift,
        "path_target": path_target,
        "passed_smoke": bool(
            beats
            and path_target["beats_oracle_tikhonov_on_path_target"]
            and path_target["filter_is_bounded"]
            and drift < 0.25
        ),
    }


def _full_specinv_bars(
    artifact: SpectrumArtifact,
    artifact_path: Path,
    args: argparse.Namespace,
    path_target: dict[str, Any],
) -> dict[str, Any]:
    """Run the existing 11/11 suite on the artifact operator."""
    forwarded = [
        "--results-dir",
        str(args.results_dir),
        "--epochs",
        str(args.epochs),
        "--n-train",
        str(args.n_train),
        "--n-test",
        str(args.n_test),
        "--seed",
        str(args.seed),
        "--spectrum",
        str(artifact_path),
        "--skip-ablations",
    ]
    if args.no_figures:
        forwarded.append("--no-figures")
    print("\n[joint] SpecInv 11/11 on the KerOp-generated operator", flush=True)
    # Individual scripts so a failure still writes whatever it finished.
    convergence.main(forwarded)
    zero_shot.main(forwarded)
    filters.main(forwarded)

    def load(name: str) -> dict[str, Any]:
        return json.loads((Path(args.results_dir) / name).read_text())

    criteria = run_all.build_criteria(
        load("convergence.json"), load("zero_shot.json"), load("filters.json")
    )
    all_passed = all(c.passed for c in criteria) and bool(
        path_target["beats_oracle_tikhonov_on_path_target"]
    )
    return {
        "graded_11_of_11": True,
        "criteria": [c.as_dict() for c in criteria],
        "all_passed": all_passed,
        "n_passed": sum(1 for c in criteria if c.passed),
        "n_criteria": len(criteria),
        "path_target": path_target,
        "operator_id": artifact.operator_id,
        "fingerprint": fingerprint(artifact),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results/joint")
    parser.add_argument("--spectrum", type=Path, default=None, help="KerOp v1 .npz or .json")
    parser.add_argument("--kerop-root", type=Path, default=None)
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="load the committed contract fixture instead of re-exporting",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--skip-kerop-bar",
        action="store_true",
        help="do not invoke the KerOp bar (the joint path then FAILs — the bar cannot be dropped)",
    )
    args = apply_quick(parser.parse_args(argv))
    directory = output_dir(args.results_dir)

    print("=" * 78)
    print("Joint path: KerOp forward spectrum → SpecInv spectral filter")
    print("Not a new theorem. No factor claimed.")
    print("=" * 78)

    try:
        require_kerop(args.kerop_root)
    except JointPathError as error:
        _print_verdict(False, "kerop_import", str(error))
        print("FAIL: joint script cannot substitute the old 1D suite.", flush=True)
        return 2

    try:
        artifact, artifact_path, resolved = resolve_spectrum(args)
    except JointPathError as error:
        _print_verdict(False, "spectrum_artifact", str(error))
        return 2

    print(
        f"  artifact  {artifact_path}  ({resolved})\n"
        f"  operator  {artifact.operator_id}\n"
        f"  n={artifact.n}  M={artifact.feature_count}  p={artifact.ill_posedness:.4g}  "
        f"fingerprint={fingerprint(artifact)}",
        flush=True,
    )
    if artifact.source != "kerop.SpectralOperatorModel":
        _print_verdict(
            False,
            "spectrum_artifact",
            f"source={artifact.source!r} is not a KerOp export; refusing to call this joint",
        )
        return 2

    # Always invert a target drawn from the artifact operator.
    path_target = apply_filter_to_path_target(artifact, args)
    _print_verdict(
        bool(path_target["beats_oracle_tikhonov_on_path_target"]),
        "specinv_path_target",
        (
            f"SC-Net {path_target['scnet_mean_relative']:.4f} vs "
            f"Oracle Tikhonov {path_target['oracle_tikhonov_mean_relative']:.4f} "
            f"on a batch sampled from the KerOp spectrum"
        ),
    )

    if args.skip_kerop_bar:
        kerop_bar = {
            "passed": False,
            "graded": True,
            "reason": "--skip-kerop-bar drops the matched-excess-risk requirement; that is a FAIL",
        }
    else:
        print(
            "\n[joint] KerOp matched-excess-risk bar on the same SpectralOperatorModel", flush=True
        )
        kerop_bar = invoke_kerop_bar(artifact, quick=args.quick, kerop_root=args.kerop_root)
    _print_verdict(
        bool(kerop_bar["passed"]),
        "kerop_matched_excess_risk",
        str(kerop_bar["reason"]),
        graded=bool(kerop_bar.get("graded", True)),
    )

    if args.quick:
        specinv_bar = _quick_specinv_bars(artifact, args, path_target)
        specinv_pass = False
        _print_verdict(
            False,
            "specinv_11_of_11",
            "not graded in --quick (that would be a false joint result)",
            graded=False,
        )
        _print_verdict(
            bool(specinv_bar["passed_smoke"]),
            "specinv_quick_smoke",
            (
                f"three-point slope={specinv_bar['rate_slope_three_point']:.3f}, "
                f"beats Tikhonov={specinv_bar['beats_oracle_tikhonov']}, "
                f"2x drift={specinv_bar['zero_shot_drift_2x']:.4f}"
            ),
            graded=False,
        )
    else:
        specinv_bar = _full_specinv_bars(artifact, artifact_path, args, path_target)
        specinv_pass = bool(specinv_bar["all_passed"])
        n_ok = specinv_bar["n_passed"]
        n_all = specinv_bar["n_criteria"]
        _print_verdict(
            specinv_pass,
            "specinv_11_of_11",
            f"{n_ok}/{n_all} reproduction criteria on the KerOp-generated operator"
            + (
                ""
                if path_target["beats_oracle_tikhonov_on_path_target"]
                else "; path-target Tikhonov comparison failed"
            ),
        )
        for row in specinv_bar["criteria"]:
            _print_verdict(
                bool(row["passed"]), row["name"], f"{row['measured']}  ({row['target']})"
            )

    joint_pass = bool(kerop_bar.get("graded") and kerop_bar["passed"] and specinv_pass)
    summary = {
        "path": "kerop.forward_spectrum -> specinv.spectral_filter",
        "claim": None,
        "note": (
            "Engineering joint path only. Not a new theorem. "
            "No factor is claimed beyond the measured bars."
        ),
        "quick": bool(args.quick),
        "environment": environment_info(args.spectrum or artifact_path),
        "artifact": artifact.metadata(),
        "artifact_path": str(artifact_path),
        "resolved_via": resolved,
        "kerop_bar": kerop_bar,
        "specinv_bar": specinv_bar,
        "path_target": path_target,
        "joint_defaults": {
            "r": JOINT_R,
            "b": JOINT_B,
            "n_modes": JOINT_N_MODES,
            "output_dim": JOINT_OUTPUT_DIM,
            "seed": JOINT_SEED,
        },
        "all_passed": joint_pass,
    }
    write_json(directory / "joint.json", summary)

    print("=" * 78)
    if args.quick:
        print(
            _paint("AMBER", _YELLOW)
            + "  --quick is a smoke of the joint wiring, not a graded joint result."
        )
        return 0 if kerop_bar["passed"] and specinv_bar.get("passed_smoke") else 1
    print(
        (_paint("GREEN", _GREEN) if joint_pass else _paint("RED", _RED))
        + ("  JOINT PASS" if joint_pass else "  JOINT FAIL")
    )
    print("=" * 78)
    return 0 if joint_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
