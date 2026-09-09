"""Joint KerOp → SpecInv lock: load KerOp's filter_contract/v1, then grade both bars.

KerOp writes ``kerop.filter_contract/v1`` as ``filter_contract_v1.{json,npz}``.
This script loads that pair and nothing else. It does not mint a parallel
``kerop.specinv.spectrum/v1`` artifact and does not fall back to SpecInv's
built-in 1-D operator.

KerOp's recorded-bar operator ids today:

* ``kerop.spectral`` — product-set eigenvalues of SpectralOperatorModel;
  this is the SpecInv 11/11 target (must have ≥ 2048 modes).
* ``kerop.poisson`` — 1-D Dirichlet-Poisson wall-time spectrum (12 modes).
  Not FEM. Too short for SpecInv 11/11. KerOp may rename this id; aliases
  are accepted.

Usage::

    specinv-joint --no-figures --results-dir results/joint
    specinv-joint --contract /path/to/filter_contract_v1
    specinv-joint --kerop-root /path/to/KerOp
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from specinv.experiments._common import environment_info, write_json
from specinv.spectrum_contract import (
    FILTER_CONTRACT_SCHEMA,
    FilterContract,
    JointPathError,
    KerOpOperator,
    find_kerop_contract_files,
    load_filter_contract,
    maybe_write_kerop_contract,
    resolve_contract_stem,
    select_operator,
    walltime_task_name,
)

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_AMBER = "\033[33m"
ANSI_RESET = "\033[0m"

# KerOp's published spectral-quick median (results/walltime_spectral.json).
# Printed as context only — the live matched-risk number is this run's median.
KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X = 27.717
SPECINV_REQUIRED_PASS = 11
SPECINV_REQUIRED_TOTAL = 11


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _verdict_line(ok: bool, text: str, *, graded: bool) -> str:
    if not graded:
        return f"{ANSI_AMBER}{text}{ANSI_RESET}"
    color = ANSI_GREEN if ok else ANSI_RED
    return f"{color}{text}{ANSI_RESET}"


def _kerop_import_error() -> str:
    return (
        "KerOp is not importable. The joint lock requires KerOp's "
        "filter_contract/v1 files and a live matched-risk bar. "
        "Install KerOp or pass --contract / --kerop-root / KEROP_ROOT. "
        "No 1-D fallback."
    )


def resolve_contract(
    *,
    contract_arg: Path | None,
    kerop_root: Path | None,
    results_dir: Path,
    write_if_missing: bool,
) -> tuple[Path, FilterContract]:
    """Resolve KerOp's filter_contract/v1 pair. Fail-closed; no 1-D fallback."""
    if contract_arg is not None:
        stem = resolve_contract_stem(contract_arg)
        return stem, load_filter_contract(stem)

    found = find_kerop_contract_files(kerop_root)
    if found is not None:
        return found, load_filter_contract(found)

    if write_if_missing:
        written = maybe_write_kerop_contract(results_dir, kerop_root=kerop_root)
        if written is None:
            raise JointPathError(
                "KerOp could not write filter_contract/v1. " + _kerop_import_error()
            )
        return written, load_filter_contract(written)

    raise JointPathError(
        "KerOp filter_contract/v1 files were not found. Looked for "
        "filter_contract_v1.{json,npz} under --kerop-root / KEROP_ROOT / "
        "a sibling KerOp checkout / this repo's fixtures/. Pass --contract "
        "pointing at KerOp's export, or --write-contract after installing "
        "KerOp. No fallback to SpecInv's built-in 1-D operator."
    )


def _reproduce_walltime_kwargs(operator: KerOpOperator) -> dict[str, Any]:
    """Map KerOp's reproduce block onto ``run_walltime_benchmark`` kwargs."""
    repro = operator.reproduce
    kwargs: dict[str, Any] = {
        "task": walltime_task_name(operator.operator_id),
        "seed": int(operator.seed),
        "verbose": True,
    }
    if "train_sizes" in repro:
        kwargs["train_sizes"] = tuple(int(x) for x in repro["train_sizes"])
    if "lambda_grid" in repro:
        kwargs["lambda_grid"] = tuple(float(x) for x in repro["lambda_grid"])
    if "feature_multipliers" in repro:
        kwargs["feature_multipliers"] = tuple(float(x) for x in repro["feature_multipliers"])
    if "iteration_grid" in repro:
        kwargs["iteration_grid"] = tuple(int(x) for x in repro["iteration_grid"])
    task_kwargs = dict(repro.get("task_kwargs") or {})
    extra = {k: v for k, v in task_kwargs.items() if k != "kind"}
    if extra:
        kwargs["task_kwargs"] = extra
    return kwargs


def _import_walltime() -> Any:
    try:
        from kerop.experiments import run_walltime_benchmark

        return run_walltime_benchmark
    except ImportError:
        pass
    try:
        from kerop.benchmarks.walltime import run_walltime_benchmark

        return run_walltime_benchmark
    except ImportError as exc:
        raise JointPathError(_kerop_import_error()) from exc


def run_kerop_bar(
    contract: FilterContract,
    target: KerOpOperator,
    *,
    skip: bool,
    quick: bool,
) -> dict[str, Any]:
    """Live KerOp matched-risk bar on the graded operator KerOp put in the contract."""
    if skip:
        return {
            "status": "fail",
            "reason": (
                "--skip-kerop-bar refuses the KerOp half of the lock. The "
                "joint claim requires a live matched-risk number, not a skip."
            ),
            "graded": True,
            "pass": False,
        }

    run_walltime_benchmark = _import_walltime()
    kwargs = _reproduce_walltime_kwargs(target)
    print(
        f"  KerOp wall-time: operator_id={target.operator_id} "
        f"task={kwargs['task']} n_modes={target.n_modes} kwargs={kwargs}"
    )
    payload = run_walltime_benchmark(**kwargs)
    verdict = dict(payload.get("verdict") or {})
    median = verdict.get("median_speedup_at_matched_risk")
    if median is None:
        raise JointPathError(
            f"KerOp wall-time returned no matched-risk median for {target.operator_id}."
        )
    median_f = float(median)
    rf_faster = bool(verdict.get("random_features_faster_at_every_target"))
    recorded = None
    rec = target.recorded_bar or contract.recorded_bar
    for key in (
        "median_speedup_at_matched_risk",
        "spectral_median_speedup_at_matched_risk",
        "rf_vs_naive_median",
    ):
        if rec.get(key) is not None:
            recorded = float(rec[key])
            break
    print(
        f"    live median RF vs exact KRR at matched risk: {median_f:.3f}x  "
        f"recorded_bar={recorded}  rf_faster_every_target={rf_faster}"
    )
    print(
        f"    published spectral-quick median {KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X}x "
        "is context only -- this run claims the live number, not ~28x"
    )
    return {
        "status": "pass" if rf_faster else "fail",
        "pass": rf_faster,
        "graded": not quick,
        "quick": quick,
        "published_spectral_quick_median_x": KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X,
        "live_spectral_median_x": median_f,
        "rf_faster_at_every_target": rf_faster,
        "operator_id": target.operator_id,
        "n_modes": target.n_modes,
        "n_samples": target.n_samples,
        "recorded_bar_median": recorded,
        "verdict": verdict,
        "reproduce": kwargs,
        "note": (
            "Live median is this run's number. The published spectral-quick "
            f"median is {KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X}x -- do not "
            "re-claim that figure if the live number is lower. "
            "kerop.poisson is 1-D Dirichlet wall-time, not FEM."
        ),
    }


def run_specinv_bar(
    stem: Path,
    operator: KerOpOperator,
    *,
    skip_ablations: bool,
    no_figures: bool,
    results_dir: Path,
    quick: bool,
) -> dict[str, Any]:
    """Run SpecInv 11/11 on the eigenvalues KerOp wrote for this operator."""
    if operator.n_modes < 2048:
        raise JointPathError(
            f"Operator {operator.operator_id!r} has n_modes={operator.n_modes}; "
            "SpecInv 11/11 needs ≥ 2048 eigenvalues. Use kerop.spectral "
            "(or another KerOp operator that actually exports that many)."
        )

    from specinv.experiments import run_all as run_all_mod

    suite_dir = results_dir / "specinv_suite"
    argv = [
        "--spectrum",
        str(stem),
        "--operator-id",
        operator.operator_id,
        "--results-dir",
        str(suite_dir),
        "--seed",
        str(operator.seed),
    ]
    if skip_ablations:
        argv.append("--skip-ablations")
    if no_figures:
        argv.append("--no-figures")
    if quick:
        argv.append("--quick")

    print(f"  SpecInv suite argv: {argv}")
    exit_code = run_all_mod.main(argv)
    summary_path = suite_dir / "summary.json"
    if not summary_path.is_file():
        raise JointPathError(f"SpecInv suite did not write {summary_path}")
    summary = json.loads(summary_path.read_text())
    criteria = list(summary.get("criteria") or [])
    n_pass = sum(1 for c in criteria if c.get("passed") or c.get("pass"))
    n_total = len(criteria)
    required_pass = 3 if quick else SPECINV_REQUIRED_PASS
    required_total = 3 if quick else SPECINV_REQUIRED_TOTAL
    ok = (exit_code == 0) and bool(summary.get("all_passed")) and n_pass >= required_pass
    return {
        "status": "pass" if ok else "fail",
        "pass": ok,
        "graded": not quick,
        "quick": quick,
        "n_pass": n_pass,
        "n_total": n_total,
        "required_pass": required_pass,
        "required_total": required_total,
        "operator_id": operator.operator_id,
        "n_modes": operator.n_modes,
        "smoothness": operator.smoothness,
        "ill_posedness": operator.ill_posedness,
        "suite_dir": str(suite_dir),
        "criteria": [
            {
                "id": c.get("name") or c.get("id"),
                "pass": bool(c.get("passed") or c.get("pass")),
                "detail": c.get("target") or c.get("detail"),
                "measured": c.get("measured"),
            }
            for c in criteria
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint KerOp → SpecInv lock. Loads kerop.filter_contract/v1 only."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/joint"),
        help="directory for joint.json (default: results/joint)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="amber smoke run; not a graded 11/11 lock",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    parser.add_argument(
        "--contract",
        "--spectrum",
        dest="contract",
        type=Path,
        default=None,
        help=(
            "Path to KerOp's filter_contract_v1 stem or .json/.npz file. "
            "--spectrum is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--kerop-root",
        type=Path,
        default=None,
        help="KerOp checkout that already has (or can write) the contract.",
    )
    parser.add_argument(
        "--operator-id",
        default="kerop.spectral",
        help="KerOp operator used as the SpecInv 11/11 target (default: kerop.spectral).",
    )
    parser.add_argument(
        "--write-contract",
        action="store_true",
        help=(
            "If the contract files are missing, ask KerOp to write them into "
            "--results-dir. Never mints a SpecInv-side spectrum schema."
        ),
    )
    parser.add_argument(
        "--skip-kerop-bar",
        action="store_true",
        help="Refuse the KerOp half of the lock (joint result is FAIL).",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Forwarded to specinv-run-all (skips the two ablation scripts).",
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    _print_banner("KerOp → SpecInv joint lock")
    print(f"schema required: {FILTER_CONTRACT_SCHEMA}")
    print("graded path: KerOp filter_contract_v1.{json,npz} — not a SpecInv mint")

    stem, contract = resolve_contract(
        contract_arg=args.contract,
        kerop_root=args.kerop_root,
        results_dir=results_dir,
        write_if_missing=bool(args.write_contract),
    )
    specinv_op = select_operator(
        contract,
        operator_id=args.operator_id,
        require_min_modes=2048,
    )
    print(f"loaded: {stem}.json + {stem}.npz")
    print(f"schema: {contract.schema}")
    print(f"producer: {contract.producer} {contract.producer_version}")
    print(
        f"SpecInv target: {specinv_op.operator_id}  n_modes={specinv_op.n_modes}  "
        f"s={specinv_op.smoothness}  p={specinv_op.ill_posedness}"
    )
    for op in contract.operators:
        print(
            f"  contract operator {op.operator_id}: kind={op.kind} "
            f"n_modes={op.n_modes} n_samples={op.n_samples} note={op.note!r}"
        )

    _print_banner("KerOp matched-risk bar (live)")
    kerop_bar = run_kerop_bar(
        contract,
        specinv_op,
        skip=bool(args.skip_kerop_bar),
        quick=bool(args.quick),
    )
    print(
        _verdict_line(
            bool(kerop_bar.get("pass")),
            f"KerOp bar: {str(kerop_bar['status']).upper()}  "
            f"live spectral median={kerop_bar.get('live_spectral_median_x')}x  "
            f"(published spectral-quick {KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X}x "
            "is context, not this run's claim)",
            graded=bool(kerop_bar.get("graded")),
        )
    )

    _print_banner("SpecInv 11/11 on KerOp's spectrum")
    specinv_bar = run_specinv_bar(
        stem,
        specinv_op,
        skip_ablations=bool(args.skip_ablations),
        no_figures=bool(args.no_figures),
        results_dir=results_dir,
        quick=bool(args.quick),
    )
    print(
        _verdict_line(
            bool(specinv_bar.get("pass")),
            f"SpecInv bar: {str(specinv_bar['status']).upper()}  "
            f"{specinv_bar['n_pass']}/{specinv_bar['n_total']}",
            graded=bool(specinv_bar.get("graded")),
        )
    )

    skip_fail = bool(args.skip_kerop_bar) or not bool(kerop_bar.get("pass"))
    if skip_fail:
        joint_status = "fail"
        joint_line = "JOINT FAIL"
        graded = True
        joint_ok = False
    elif args.quick:
        joint_status = "ungraded"
        joint_line = "JOINT UNGRADED (--quick; not the 11/11 lock)"
        graded = False
        joint_ok = bool(kerop_bar.get("pass")) and bool(specinv_bar.get("pass"))
    elif bool(kerop_bar.get("pass")) and bool(specinv_bar.get("pass")):
        joint_status = "pass"
        joint_line = "JOINT PASS"
        graded = True
        joint_ok = True
    else:
        joint_status = "fail"
        joint_line = "JOINT FAIL"
        graded = True
        joint_ok = False

    _print_banner(joint_line)
    print(
        _verdict_line(joint_ok, joint_line, graded=graded)
        + ("" if graded else f"  {ANSI_AMBER}(amber: not a graded 11/11 result){ANSI_RESET}")
    )

    payload = {
        "created_at": _now(),
        "schema": contract.schema,
        "contract_stem": str(stem),
        "producer": contract.producer,
        "producer_version": contract.producer_version,
        "specinv_operator_id": specinv_op.operator_id,
        "quick": bool(args.quick),
        "graded": graded,
        "joint_status": joint_status,
        "joint_pass": joint_ok and graded,
        "kerop": kerop_bar,
        "specinv": specinv_bar,
        "environment": environment_info(str(stem), operator_id=specinv_op.operator_id),
        "notes": [
            "Graded path loads kerop.filter_contract/v1 written by KerOp.",
            "SpecInv does not mint kerop.specinv.spectrum/v1 as the joint artifact.",
            "kerop.poisson is 1-D Dirichlet wall-time (not FEM).",
            (
                f"KerOp published spectral-quick median {KEROP_PUBLISHED_SPECTRAL_QUICK_MEDIAN_X}x "
                "is not re-claimed; live median is kerop.live_spectral_median_x."
            ),
        ],
    }
    write_json(results_dir / "joint.json", payload)
    print(f"wrote {results_dir / 'joint.json'}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = _run(args)
    except JointPathError as exc:
        print(f"{ANSI_RED}JOINT FAIL{ANSI_RESET}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{ANSI_RED}JOINT FAIL{ANSI_RESET}: {exc}", file=sys.stderr)
        return 2
    if payload["joint_status"] == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
