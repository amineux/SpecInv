"""§5.3 -- interpretability of the learned spectral filter.

Reproduces Figure 2 and the claims attached to it.  Because SC-Net's only learned object is
the scalar damping :math:`\\Psi_\\theta(y_n,\\sigma_n) \\in [0,1]`, it can be tabulated mode by
mode and overlaid directly on the classical filters of §2.3.  This script writes the
filter profiles to CSV so they can be inspected without rerunning anything, and checks the
three specific claims of §5.3:

1. the filter is :math:`\\approx 1` on the leading modes (no bias where the SNR is high);
2. it decays to :math:`\\approx 0` on high modes (noise suppression);
3. the transition is *sharper* than Tikhonov's -- quantified here by the width of the
   band where :math:`0.1 \\le \\lambda_n \\le 0.9` and by the mass of the filter tail, which
   is what the paper means by Tikhonov's "heavy tail".

It also records the filter's dependence on the noise level, which is the part of the
architecture that Definition 4.3 predicts: the learned cutoff should track
:math:`n_*(\\delta) \\asymp \\delta^{-1/(s+p)}`.

Usage
-----
    python -m specinv.experiments.filters --results-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..filters import oracle_tikhonov, oracle_tsvd, tikhonov_damping, tsvd_damping
from ..scnet import SCNetConfig
from ..theory import optimal_truncation_index
from ..training import TrainConfig
from ._common import (
    add_common_arguments,
    apply_quick,
    build_suite,
    environment_info,
    output_dir,
    train_model,
    write_csv,
    write_json,
)

PROFILE_DELTAS: tuple[float, ...] = (0.1, 0.05, 0.01, 0.001)
N_MODES = 256
N_PROFILE_MODES = 48


def transition_width(damping: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float:
    """Number of modes spanned by the band ``lo <= lambda <= hi``.

    A sharp (TSVD-like) filter has a width near 0; Tikhonov's rational profile is wider.
    Computed by interpolating the crossing points so the value is not quantised to
    integers.
    """
    n = np.arange(1, damping.size + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(damping)

    def crossing(level: float) -> float:
        below = np.nonzero(monotone <= level)[0]
        if below.size == 0:
            return float(n[-1])
        first = int(below[0])
        if first == 0:
            return float(n[0])
        y0, y1 = monotone[first - 1], monotone[first]
        if y0 == y1:
            return float(n[first])
        return float(n[first - 1] + (y0 - level) / (y0 - y1))

    return crossing(lo) - crossing(hi)


def tail_mass(damping: np.ndarray, cutoff: float) -> float:
    """Total damping assigned above the mode where the filter first drops below 0.5.

    Quantifies the "heavy tail" of §5.3: how much weight a filter still gives to modes it
    has nominally rejected.
    """
    n = np.arange(1, damping.size + 1, dtype=np.float64)
    return float(np.sum(damping[n > cutoff]))


def half_power_mode(damping: np.ndarray) -> float:
    """Mode index where the damping crosses 1/2 -- the filter's effective cutoff."""
    n = np.arange(1, damping.size + 1, dtype=np.float64)
    monotone = np.minimum.accumulate(damping)
    below = np.nonzero(monotone <= 0.5)[0]
    if below.size == 0:
        return float(n[-1])
    first = int(below[0])
    if first == 0:
        return float(n[0])
    y0, y1 = monotone[first - 1], monotone[first]
    if y0 == y1:
        return float(n[first])
    return float(n[first - 1] + (y0 - 0.5) / (y0 - y1))


def make_figure(payload: dict[str, Any], path: Path) -> None:
    """Figure 2: learned filter vs. Tikhonov vs. TSVD."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable, skipping figure")
        return

    profiles = payload["profiles"]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))

    # (a) The profile at the paper's delta = 5%, on a log mode axis so the transition is
    # not squeezed into three pixels.
    reference = profiles["0.05"]
    modes = np.asarray(reference["modes"], dtype=float)
    ax = axes[0]
    ax.semilogx(
        modes, reference["scnet"], "o-", color="#b02418", lw=2.3, ms=4, label="SC-Net (learned)"
    )
    ax.semilogx(
        modes,
        reference["oracle_tikhonov"],
        "s--",
        color="#1f4e9c",
        lw=1.8,
        ms=3.5,
        label="Oracle Tikhonov",
    )
    ax.semilogx(
        modes, reference["oracle_tsvd"], ":", color="k", lw=2.0, label="Oracle TSVD (ideal step)"
    )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("spectral index $n$ (log)")
    ax.set_ylabel(r"damping $\lambda_n$")
    ax.set_title(r"(a) Learned filter at $\delta=5\%$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # (b) The same profiles on a log damping axis: this is where Tikhonov's "heavy tail"
    # of Sec. 5.3 is visible and the learned filter's faster decay is unmistakable.
    ax = axes[1]
    floor = 1e-7
    ax.loglog(
        modes,
        np.maximum(reference["scnet"], floor),
        "o-",
        color="#b02418",
        lw=2.3,
        ms=4,
        label="SC-Net (learned)",
    )
    ax.loglog(
        modes,
        np.maximum(reference["oracle_tikhonov"], floor),
        "s--",
        color="#1f4e9c",
        lw=1.8,
        ms=3.5,
        label=r"Oracle Tikhonov ($\propto n^{-2p}$ tail)",
    )
    ax.set_ylim(1e-6, 2.0)
    ax.set_xlabel("spectral index $n$ (log)")
    ax.set_ylabel(r"damping $\lambda_n$ (log)")
    ax.set_title("(b) Tail behaviour at $\\delta=5\\%$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # (c) Adaptivity: the learned cutoff must move with the noise level.
    ax = axes[2]
    colours = ["#b02418", "#e07b26", "#2f7d3a", "#1f4e9c"]
    for colour, delta in zip(colours, PROFILE_DELTAS, strict=False):
        entry = profiles[str(delta)]
        ax.semilogx(
            entry["modes"],
            entry["scnet"],
            "-",
            color=colour,
            lw=2.2,
            label=rf"SC-Net, $\delta={delta}$",
        )
        ax.semilogx(
            entry["modes"], entry["oracle_tikhonov"], "--", color=colour, lw=1.0, alpha=0.7
        )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("spectral index $n$ (log)")
    ax.set_ylabel(r"damping $\lambda_n$")
    ax.set_title("(c) Cutoff tracks the noise level\n(dashed = Oracle Tikhonov)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

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
    args = apply_quick(parser.parse_args(argv))

    directory = output_dir(args.results_dir)
    figures = output_dir(directory / "figures")

    suite = build_suite(n_modes=2048)
    train_config = TrainConfig(
        n_train=args.n_train,
        n_val=args.n_test,
        n_modes=N_MODES,
        delta_range=(1e-3, 1e-1),
        epochs=args.epochs,
        seed=args.seed,
        gamma=0.0,
    )
    print("\n[filters] training SC-Net")
    model, train_summary = train_model(suite, train_config, SCNetConfig(), verbose=True)

    sv = suite.operator.restrict(N_MODES).singular_values
    profiles: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    claims: dict[str, Any] = {}

    for delta in PROFILE_DELTAS:
        rng = np.random.default_rng(args.seed + 31)
        batch = suite.sample(args.n_test, delta, rng, n_modes=N_MODES)

        learned = model.filter_profile(batch.noisy_data, sv).mean(axis=0)
        oracle_tik = oracle_tikhonov(batch.noisy_data, sv, batch.true_coefficients)
        alpha = float(np.median(oracle_tik.parameter))
        tik = tikhonov_damping(sv, alpha)
        oracle_cut = float(
            np.median(oracle_tsvd(batch.noisy_data, sv, batch.true_coefficients).parameter)
        )
        tsvd = tsvd_damping(sv, float(sv[max(int(round(oracle_cut)) - 1, 0)] ** 2))

        entry = {
            "modes": list(range(1, N_PROFILE_MODES + 1)),
            "scnet": learned[:N_PROFILE_MODES].tolist(),
            "oracle_tikhonov": tik[:N_PROFILE_MODES].tolist(),
            "oracle_tsvd": tsvd[:N_PROFILE_MODES].tolist(),
            "median_tikhonov_alpha": alpha,
            "median_oracle_tsvd_cutoff": oracle_cut,
        }
        profiles[str(delta)] = entry

        learned_cut = half_power_mode(learned)
        # "Suppresses high modes" has to be judged relative to the filter's own cutoff:
        # at delta=1e-3 the optimal cutoff sits near mode 10, so a fixed probe mode would
        # test different things at different noise levels.
        far_mode = min(int(np.ceil(4.0 * learned_cut)), N_MODES) - 1
        claims[str(delta)] = {
            "scnet_leading_mode_damping": float(learned[0]),
            "scnet_damping_at_mode_32": float(learned[31]),
            "scnet_damping_at_4x_cutoff": float(learned[far_mode]),
            "probe_mode_4x_cutoff": far_mode + 1,
            "scnet_half_power_mode": learned_cut,
            "tikhonov_half_power_mode": half_power_mode(tik),
            "theory_optimal_truncation_index": float(
                optimal_truncation_index(delta, suite.smoothness, suite.ill_posedness)
            ),
            "scnet_transition_width": transition_width(learned),
            "tikhonov_transition_width": transition_width(tik),
            "scnet_tail_mass": tail_mass(learned, learned_cut),
            "tikhonov_tail_mass": tail_mass(tik, learned_cut),
        }
        c = claims[str(delta)]
        print(
            f"    delta={delta:<7g} lambda_1={c['scnet_leading_mode_damping']:.4f}  "
            f"cutoff={learned_cut:.2f} (tikh {c['tikhonov_half_power_mode']:.2f}, "
            f"theory~{c['theory_optimal_truncation_index']:.1f})  "
            f"width={c['scnet_transition_width']:.2f} vs {c['tikhonov_transition_width']:.2f}  "
            f"tail={c['scnet_tail_mass']:.3f} vs {c['tikhonov_tail_mass']:.3f}"
        )

        for i in range(N_PROFILE_MODES):
            rows.append(
                {
                    "delta": delta,
                    "mode": i + 1,
                    "singular_value": sv[i],
                    "scnet_damping": learned[i],
                    "oracle_tikhonov_damping": tik[i],
                    "oracle_tsvd_damping": tsvd[i],
                }
            )

    payload: dict[str, Any] = {
        "experiment": "spectral_filter_interpretability",
        "paper_section": "5.3",
        "environment": environment_info(),
        "training": train_summary,
        "resolution": N_MODES,
        "profiles": profiles,
        "claims": claims,
        "claim_checks": {
            "filter_is_bounded_in_unit_interval": bool(
                all(
                    0.0 <= v <= 1.0
                    for entry in profiles.values()
                    for v in entry["scnet"]
                )
            ),
            "preserves_leading_modes": bool(
                all(c["scnet_leading_mode_damping"] > 0.9 for c in claims.values())
            ),
            "suppresses_high_modes": bool(
                all(c["scnet_damping_at_4x_cutoff"] < 0.05 for c in claims.values())
            ),
            "sharper_than_tikhonov": bool(
                all(
                    c["scnet_transition_width"] < c["tikhonov_transition_width"]
                    for c in claims.values()
                )
            ),
            "lighter_tail_than_tikhonov": bool(
                all(c["scnet_tail_mass"] < c["tikhonov_tail_mass"] for c in claims.values())
            ),
        },
    }

    write_json(directory / "filters.json", payload)
    write_csv(directory / "filter_profiles.csv", rows)
    if not args.no_figures:
        make_figure(payload, figures / "fig2_learned_filter.png")

    print("\nclaim checks:")
    checks: dict[str, bool] = payload["claim_checks"]
    for key, value in checks.items():
        print(f"  {'PASS' if value else 'FAIL'}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())