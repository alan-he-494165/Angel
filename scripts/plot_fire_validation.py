#!/usr/bin/env python3
"""Plot FiRE validation results against the FCI/CBS reference.

Reads only the CSVs written by ``scripts/validate_fire.py`` (and, for the
reference uncertainty, ``scripts/fci_reference.py``), so it needs pandas and
matplotlib but neither JAX nor PySCF.  Two figures:

* one run      -- convergence traces + final energies vs HF/CBS and FCI/cc-pVDZ
* two runs     -- the same run at two settings, e.g. two base learning rates
                  (pass ``--compare`` with the second run's directory)

Examples
--------
    python scripts/plot_fire_validation.py --run out --out out/fire_validation.png
    python scripts/plot_fire_validation.py --run out --compare /tmp/fv_lr005 \\
        --labels "$lr_0=0.1$" "$lr_0=0.005$" --out out/fire_lr_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

META_GREY = "#6e6e6e"
SYS_COLOUR = {"H2": "#1b6ca8", "LiH": "#c77400", "Be": "#7a7a7a"}
SYS_LABEL = {"H2": "H$_2$ (2 e$^-$)", "LiH": "LiH (4 e$^-$)", "Be": "Be (4 e$^-$)"}
CHEMICAL_ACCURACY_MHARTREE = 1.6


def apply_style() -> None:
    """Inline rcParams; no external style dependency."""

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


def panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.16, 1.06, letter, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="left",
    )


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one validation directory into (summary, traces).

    Accepts either a directory holding ``fire_validation.csv`` directly or one
    holding sub-directories that each do (as written by two-part sweeps).
    """

    summaries, traces = [], []
    candidates = [run_dir, *sorted(p for p in run_dir.iterdir() if p.is_dir())]
    for d in candidates:
        s, t = d / "fire_validation.csv", d / "fire_validation_traces.csv"
        if s.exists() and t.exists():
            summaries.append(pd.read_csv(s))
            traces.append(pd.read_csv(t))
    if not summaries:
        raise SystemExit(f"no fire_validation.csv under {run_dir}")
    summary = pd.concat(summaries, ignore_index=True).drop_duplicates("system", keep="last")
    trace = pd.concat(traces, ignore_index=True)
    reference = dict(zip(summary.system, summary.reference_hartree))
    trace["error_mhartree"] = (
        trace.energy_hartree - trace.system.map(reference)
    ) * 1e3
    return summary, trace


def binned(trace: pd.DataFrame, system: str, width: int = 100):
    g = trace[trace.system == system].sort_values("step")
    err = g.error_mhartree.to_numpy()
    n = len(err) // width * width
    if n == 0:
        return g.step.to_numpy(), err
    return (
        g.step.to_numpy()[:n].reshape(-1, width).mean(1),
        err[:n].reshape(-1, width).mean(1),
    )


def order_of(summary: pd.DataFrame) -> list[str]:
    known = [s for s in ("H2", "LiH", "Be") if s in set(summary.system)]
    return known + [s for s in summary.system if s not in known]


def uncertainty_band(ax, systems: list[str], summary: pd.DataFrame) -> None:
    unc = dict(zip(summary.system, summary.reference_uncertainty_mhartree))
    for s in systems:
        y, u = systems.index(s), unc[s]
        ax.add_patch(
            mpl.patches.Rectangle(
                (-u, y - 0.42), 2 * u, 0.84, color="#d9d9d9", alpha=0.85, lw=0, zorder=1
            )
        )


def lollipops(ax, systems, series) -> None:
    """series: list of (label, colour, marker size, value_getter).

    Returns nothing; sets an x range with headroom so no marker is clipped.
    """

    n = len(series)
    values = [v for _, _, _, get in series for v in (get(s) for s in systems) if v is not None]
    for j, (label, colour, size, getter) in enumerate(series):
        offset = (j - (n - 1) / 2) * (0.26 if n == 2 else 0.22)
        for s in systems:
            value = getter(s)
            if value is None:
                continue
            y = systems.index(s) + offset
            ax.plot([0, value], [y, y], color=colour, lw=0.9, alpha=0.6, zorder=2)
            ax.plot(
                [value], [y], "o", color=colour, ms=size, zorder=4,
                label=label if s == systems[0] else None,
            )
    lo, hi = min(values), max(values)
    ax.set_xlim(min(-2.0, lo * 3 if lo < 0 else -2.0), max(10.0, hi * 3))


def figure(runs, labels, out_path: Path) -> None:
    apply_style()
    summary, trace = runs[0]
    systems = order_of(summary)
    compare = len(runs) > 1

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.35 if compare else 1.3, 1]}
    )

    ax.axhspan(
        -CHEMICAL_ACCURACY_MHARTREE, CHEMICAL_ACCURACY_MHARTREE,
        color="#cfe3cf", alpha=0.7, lw=0, zorder=0,
    )
    for s in systems:
        if compare:
            step, err = binned(runs[1][1], s)
            ax.plot(step, err, color=SYS_COLOUR.get(s, "#444444"), lw=1.0, alpha=0.45, zorder=2)
        step, err = binned(trace, s)
        ax.plot(
            step, err, color=SYS_COLOUR.get(s, "#444444"), lw=1.5 if compare else 1.3,
            zorder=3, label=SYS_LABEL.get(s, s),
        )
    if compare:
        ax.plot([], [], color=META_GREY, lw=1.5, label=labels[0])
        ax.plot([], [], color=META_GREY, lw=1.0, alpha=0.45, label=labels[1])
    ax.axhline(0, color=META_GREY, lw=0.6, zorder=1)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_yticks([-1000, -10, 0, 10, 1000])
    ax.set_xlabel("optimization step")
    ax.set_ylabel("energy $-$ FCI/CBS  (mE$_h$)")
    ax.set_title(
        "Convergence at two base learning rates" if compare
        else "Convergence toward the FCI basis-set limit",
        loc="left",
    )
    ax.margins(x=0.03)
    ax.legend(
        loc="upper right", fontsize=5.8 if compare else 6,
        ncol=2 if compare else 1, columnspacing=1.0,
    )
    ax.text(
        0.03, 0.42, "chemical accuracy", transform=ax.transAxes,
        fontsize=6, color="#3f6b3f", va="top",
    )

    if compare:
        series = [
            (labels[1], "#9a9a9a", 4.5,
             lambda s: float(runs[1][0].set_index("system").fire_minus_reference_mhartree[s])),
            (labels[0], "#b3272d", 5.5,
             lambda s: float(summary.set_index("system").fire_minus_reference_mhartree[s])),
        ]
        title = "Final energy after optimization"
    else:
        table = summary.set_index("system")

        def rel(column):
            return lambda s: (float(table[column][s]) - float(table.reference_hartree[s])) * 1e3

        series = [
            ("HF/CBS", "#b0b0b0", 4.5, rel("hf_cbs_hartree")),
            ("FCI/cc-pVDZ", "#6f9ad3", 4.5, rel("fci_basis_hartree")),
            ("FiRE (this work)", "#b3272d", 4.5,
             lambda s: float(table.fire_minus_reference_mhartree[s])),
        ]
        title = "Final energies against the FCI limit"

    uncertainty_band(ax2, systems, summary)
    lollipops(ax2, systems, series)
    ax2.axvline(0, color=META_GREY, lw=0.8, zorder=3)
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.set_yticks(range(len(systems)))
    ax2.set_yticklabels([SYS_LABEL.get(s, s) for s in systems])
    ax2.set_ylim(-0.62, len(systems) - 0.25)
    ax2.invert_yaxis()
    ax2.set_xlabel("energy $-$ FCI/CBS  (mE$_h$)")
    ax2.set_title(title, loc="left")
    ax2.legend(loc="lower left", fontsize=6, handletextpad=0.4, borderaxespad=0.3)
    ax2.text(
        0.02, 0.99, "grey band: FCI/CBS reference uncertainty",
        transform=ax2.transAxes, fontsize=5.5, color="#666666", va="top", ha="left",
    )

    panel_letter(ax, "a")
    panel_letter(ax2, "b")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"wrote {out_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, default=Path("out"), help="Validation output directory.")
    parser.add_argument("--compare", type=Path, default=None, help="Second run to overlay.")
    parser.add_argument(
        "--labels", type=str, nargs=2, default=("run", "comparison"),
        help="Legend labels for --run and --compare, in that order.",
    )
    parser.add_argument("--out", type=Path, default=Path("out/fire_validation.png"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [load_run(args.run)]
    if args.compare is not None:
        runs.append(load_run(args.compare))
    figure(runs, list(args.labels), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
