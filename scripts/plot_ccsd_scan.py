#!/usr/bin/env python3
"""Collect DLPNO-CCSD(T) energies from the ORCA benzene-dimer scan and plot them
alongside the MACE / DFT / FCI relative-energy scan.

Coupled-cluster energies are read from the ORCA ``*.property.txt`` files written
by each ``ORCA_scripts/benzene_dimer_parallel_<sep>A`` job.  Two correlated
levels are reported:

* ``ccsd``    = reference + correlation - (T)          [DLPNO-CCSD]
* ``ccsd_t``  = reference + correlation                [DLPNO-CCSD(T)]

All curves are reported relative to a common reference separation (2.80 A by
default), matching the convention of ``results/mace_fci_scan/relative_energy_scan.csv``.

Example
-------
    python scripts/plot_ccsd_scan.py
    python scripts/plot_ccsd_scan.py --reference-distance 6.00 --no-ccsd-no-t
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

HARTREE_TO_KCAL_MOL = 627.5094740631
REPO_ROOT = Path(__file__).resolve().parents[1]

# Directory names look like "benzene_dimer_parallel_3p40A" -> 3.40 Angstrom.
DIR_DISTANCE_RE = re.compile(r"_(\d+)p(\d+)A$")
# Inside a $Block, an ArrayOfDoubles entry is followed by an index/value line:
#     &refEnergy [&Type "ArrayOfDoubles", &Dim (1,1)] "Reference Energy"
#                                                              0
#
#     0                                     -4.6154301961298870e+02
ARRAY_VALUE_RE = re.compile(r"^\s*0\s+(-?\d+\.\d+[eE][+-]\d+)\s*$", re.MULTILINE)


@dataclass
class CoupledClusterPoint:
    """Energies extracted from one ORCA single-point property file."""

    distance_angstrom: float
    directory: str
    hf_energy_hartree: float
    corr_energy_hartree: float
    triples_energy_hartree: float
    ccsd_t_energy_hartree: float

    @property
    def ccsd_energy_hartree(self) -> float:
        """DLPNO-CCSD total energy, i.e. CCSD(T) with the triples removed."""
        return self.ccsd_t_energy_hartree - self.triples_energy_hartree


@dataclass
class ScanRow:
    """One separation, with every available method on a common reference."""

    distance_angstrom: float
    hf_energy_hartree: float
    ccsd_energy_hartree: float
    ccsd_t_energy_hartree: float
    hf_relative_kcal_mol: float
    ccsd_relative_kcal_mol: float
    ccsd_t_relative_kcal_mol: float
    mace_relative_kcal_mol: float | None
    dft_relative_kcal_mol: float | None
    fci_relative_kcal_mol: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--orca-dir",
        type=Path,
        default=REPO_ROOT / "ORCA_scripts",
        help="Directory holding the per-separation ORCA job directories.",
    )
    parser.add_argument(
        "--dir-glob",
        default="benzene_dimer_parallel_*A",
        help="Glob for the per-separation ORCA job directories.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=REPO_ROOT / "results" / "mace_fci_scan" / "relative_energy_scan.csv",
        help="Existing MACE/DFT/FCI scan CSV to overlay. Pass an absent path to skip the overlay.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ccsd_scan",
        help="Directory for the generated CSV and plot.",
    )
    parser.add_argument("--csv-name", default="ccsd_relative_energy_scan.csv")
    parser.add_argument("--plot-name", default="ccsd_relative_energy_scan.png")
    parser.add_argument(
        "--reference-distance",
        type=float,
        default=2.80,
        help="Separation (Angstrom) used as the zero of every relative-energy curve.",
    )
    parser.add_argument(
        "--distance-tolerance",
        type=float,
        default=1e-3,
        help="Tolerance (Angstrom) when matching ORCA separations to CSV separations.",
    )
    parser.add_argument(
        "--no-ccsd-no-t",
        action="store_true",
        help="Plot only DLPNO-CCSD(T), omitting the triples-free DLPNO-CCSD curve.",
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help=(
            "Also plot the Hartree-Fock reference curve. Off by default: it spans roughly "
            "twice the range of the correlated curves and compresses them."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def distance_from_dirname(name: str) -> float | None:
    """'benzene_dimer_parallel_3p40A' -> 3.40; None if the name does not encode one."""
    match = DIR_DISTANCE_RE.search(name)
    if match is None:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def extract_block(text: str, block_name: str) -> str | None:
    """Return the body of an ORCA property block such as '$MDCI_Energies'."""
    start = text.find(f"${block_name}")
    if start == -1:
        return None
    end = text.find("$End", start)
    if end == -1:
        return None
    return text[start:end]


def read_array_scalar(block: str, key: str) -> float:
    """Read the single value of a (1,1) ArrayOfDoubles entry named `key`."""
    key_pos = block.find(f"&{key} ")
    if key_pos == -1:
        raise ValueError(f"Property block has no entry '&{key}'.")
    match = ARRAY_VALUE_RE.search(block, key_pos)
    if match is None:
        raise ValueError(f"No numeric value found after '&{key}'.")
    return float(match.group(1))


def parse_property_file(path: Path, distance_angstrom: float) -> CoupledClusterPoint:
    text = path.read_text()

    mdci = extract_block(text, "MDCI_Energies")
    if mdci is None:
        raise ValueError(f"{path} contains no $MDCI_Energies block (job likely did not finish).")

    hf = read_array_scalar(mdci, "refEnergy")
    corr = read_array_scalar(mdci, "corrEnergy")
    triples = read_array_scalar(mdci, "triplesEnergy")
    total = read_array_scalar(mdci, "totalEnergy")

    # ORCA's corrEnergy already includes the (T) correction; verify before trusting
    # the CCSD = total - triples decomposition below.
    if not math.isclose(hf + corr, total, abs_tol=1e-8):
        raise ValueError(
            f"{path}: refEnergy + corrEnergy ({hf + corr:.10f}) does not match "
            f"totalEnergy ({total:.10f}); the energy decomposition is not as expected."
        )

    return CoupledClusterPoint(
        distance_angstrom=distance_angstrom,
        directory=path.parent.name,
        hf_energy_hartree=hf,
        corr_energy_hartree=corr,
        triples_energy_hartree=triples,
        ccsd_t_energy_hartree=total,
    )


def collect_points(orca_dir: Path, dir_glob: str) -> list[CoupledClusterPoint]:
    points: list[CoupledClusterPoint] = []
    for job_dir in sorted(orca_dir.glob(dir_glob)):
        if not job_dir.is_dir():
            continue
        distance = distance_from_dirname(job_dir.name)
        if distance is None:
            print(f"  skip {job_dir.name}: no separation encoded in directory name")
            continue
        property_files = sorted(job_dir.glob("*.property.txt"))
        if not property_files:
            print(f"  skip {job_dir.name}: no *.property.txt found")
            continue
        if len(property_files) > 1:
            print(f"  warn {job_dir.name}: {len(property_files)} property files, using {property_files[0].name}")
        try:
            points.append(parse_property_file(property_files[0], distance))
        except ValueError as error:
            print(f"  skip {job_dir.name}: {error}")
    points.sort(key=lambda point: point.distance_angstrom)
    return points


def load_reference_csv(path: Path) -> dict[str, dict[float, float]]:
    """Map method -> {distance: relative kcal/mol} from the MACE/DFT/FCI scan CSV."""
    columns = ("mace_relative_kcal_mol", "dft_relative_kcal_mol", "fci_relative_kcal_mol")
    table: dict[str, dict[float, float]] = {column: {} for column in columns}
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            distance = float(record["order_value_angstrom"])
            for column in columns:
                value = record.get(column, "")
                if value not in ("", None):
                    table[column][distance] = float(value)
    return table


def lookup(series: dict[float, float], distance: float, tolerance: float) -> float | None:
    for key, value in series.items():
        if abs(key - distance) <= tolerance:
            return value
    return None


def rebase(series: dict[float, float], reference: float, tolerance: float) -> dict[float, float]:
    """Shift a relative-energy series so that `reference` is the zero point."""
    offset = lookup(series, reference, tolerance)
    if offset is None:
        return dict(series)
    return {distance: value - offset for distance, value in series.items()}


def build_rows(
    points: list[CoupledClusterPoint],
    reference_table: dict[str, dict[float, float]],
    reference_distance: float,
    tolerance: float,
) -> list[ScanRow]:
    anchor = next(
        (point for point in points if abs(point.distance_angstrom - reference_distance) <= tolerance),
        None,
    )
    if anchor is None:
        available = ", ".join(f"{point.distance_angstrom:.2f}" for point in points)
        raise ValueError(f"No ORCA point at {reference_distance:.2f} A. Available: {available}")

    rebased = {
        column: rebase(series, reference_distance, tolerance) for column, series in reference_table.items()
    }

    rows: list[ScanRow] = []
    for point in points:
        rows.append(
            ScanRow(
                distance_angstrom=point.distance_angstrom,
                hf_energy_hartree=point.hf_energy_hartree,
                ccsd_energy_hartree=point.ccsd_energy_hartree,
                ccsd_t_energy_hartree=point.ccsd_t_energy_hartree,
                hf_relative_kcal_mol=(point.hf_energy_hartree - anchor.hf_energy_hartree) * HARTREE_TO_KCAL_MOL,
                ccsd_relative_kcal_mol=(point.ccsd_energy_hartree - anchor.ccsd_energy_hartree)
                * HARTREE_TO_KCAL_MOL,
                ccsd_t_relative_kcal_mol=(point.ccsd_t_energy_hartree - anchor.ccsd_t_energy_hartree)
                * HARTREE_TO_KCAL_MOL,
                mace_relative_kcal_mol=lookup(rebased["mace_relative_kcal_mol"], point.distance_angstrom, tolerance),
                dft_relative_kcal_mol=lookup(rebased["dft_relative_kcal_mol"], point.distance_angstrom, tolerance),
                fci_relative_kcal_mol=lookup(rebased["fci_relative_kcal_mol"], point.distance_angstrom, tolerance),
            )
        )
    return rows


def write_csv(rows: list[ScanRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [field.name for field in fields(ScanRow)]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if getattr(row, name) is None else getattr(row, name) for name in header])


def plot_rows(rows: list[ScanRow], path: Path, args: argparse.Namespace) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    base, small, tick = 10, 9, 8
    mpl.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "legend.fontsize": small,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "savefig.bbox": "tight",
        }
    )

    distances = [row.distance_angstrom for row in rows]

    # DLPNO-CCSD(T) is the focal series: saturated, heavier. Comparators are lighter.
    series = [
        ("ccsd_t_relative_kcal_mol", "DLPNO-CCSD(T)/aug-cc-pVTZ", "#0b3d91", "o", 2.2, 1.0, 5),
    ]
    if not args.no_ccsd_no_t:
        series.append(("ccsd_relative_kcal_mol", "DLPNO-CCSD (no (T))", "#4c86c6", "v", 1.4, 0.95, 4))
    if args.hf:
        series.append(("hf_relative_kcal_mol", "Hartree-Fock/aug-cc-pVTZ", "#9aa5b1", "d", 1.2, 0.9, 4))
    series += [
        ("mace_relative_kcal_mol", "MACE-OFF", "#d95f02", "s", 1.4, 0.9, 4),
        ("dft_relative_kcal_mol", "$\\omega$B97M-V/def2-TZVPD", "#7570b3", "^", 1.4, 0.9, 4),
        ("fci_relative_kcal_mol", "FCI(12e,12o)", "#1b9e77", "P", 1.4, 0.9, 4),
    ]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)

    plotted = 0
    for attr, label, color, marker, width, alpha, msize in series:
        pairs = [(d, getattr(row, attr)) for d, row in zip(distances, rows) if getattr(row, attr) is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(
            xs,
            ys,
            marker=marker,
            markersize=msize,
            linewidth=width,
            alpha=alpha,
            color=color,
            label=label,
            zorder=3 if attr == "ccsd_t_relative_kcal_mol" else 2,
        )
        plotted += 1
    if plotted == 0:
        raise ValueError("Nothing to plot.")

    ax.set_xlabel("Interplanar separation (\u00c5)")
    ax.set_ylabel(f"Energy relative to {args.reference_distance:.2f} \u00c5 (kcal/mol)")
    ax.set_title("Parallel benzene dimer: relative energy scan")
    ax.margins(0.05)
    # Sits in the empty band between the correlated curves and the active-space FCI curve.
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.40))
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    orca_dir = args.orca_dir.expanduser().resolve()
    print(f"Reading ORCA property files from {orca_dir}")
    points = collect_points(orca_dir, args.dir_glob)
    if not points:
        raise SystemExit(f"No usable ORCA property files under {orca_dir}/{args.dir_glob}")
    print(f"Parsed {len(points)} coupled-cluster points.")

    reference_csv = args.reference_csv.expanduser().resolve()
    if reference_csv.is_file():
        reference_table = load_reference_csv(reference_csv)
        print(f"Overlaying MACE/DFT/FCI curves from {reference_csv}")
    else:
        reference_table = {
            "mace_relative_kcal_mol": {},
            "dft_relative_kcal_mol": {},
            "fci_relative_kcal_mol": {},
        }
        print(f"No reference CSV at {reference_csv}; plotting coupled-cluster curves only.")

    rows = build_rows(points, reference_table, args.reference_distance, args.distance_tolerance)

    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name

    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    plot_rows(rows, plot_path, args)
    print(f"Wrote {plot_path}")

    print()
    print(f"{'d (A)':>7} {'CCSD(T)':>10} {'CCSD':>10} {'HF':>10} {'MACE':>10} {'DFT':>10} {'FCI':>10}")
    for row in rows:
        def fmt(value: float | None) -> str:
            return "        --" if value is None else f"{value:10.3f}"

        print(
            f"{row.distance_angstrom:7.2f} "
            f"{row.ccsd_t_relative_kcal_mol:10.3f} "
            f"{row.ccsd_relative_kcal_mol:10.3f} "
            f"{row.hf_relative_kcal_mol:10.3f} "
            f"{fmt(row.mace_relative_kcal_mol)} "
            f"{fmt(row.dft_relative_kcal_mol)} "
            f"{fmt(row.fci_relative_kcal_mol)}"
        )


if __name__ == "__main__":
    main()
