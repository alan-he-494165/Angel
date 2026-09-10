#!/usr/bin/env python3
"""Generate parallel benzene dimer XYZ structures over a spacing scan."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


ANGSTROM = "A"
DEFAULT_SPACINGS = (2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.5, 5.0, 6.0)


def benzene_monomer(z: float) -> list[tuple[str, float, float, float]]:
    """Return a planar C6H6 monomer centered at the origin in the xy plane."""
    carbon_carbon = 1.397
    carbon_hydrogen = 1.090
    carbon_radius = carbon_carbon
    hydrogen_radius = carbon_radius + carbon_hydrogen

    atoms: list[tuple[str, float, float, float]] = []
    for index in range(6):
        angle = 2.0 * math.pi * index / 6.0
        atoms.append(
            (
                "C",
                carbon_radius * math.cos(angle),
                carbon_radius * math.sin(angle),
                z,
            )
        )

    for index in range(6):
        angle = 2.0 * math.pi * index / 6.0
        atoms.append(
            (
                "H",
                hydrogen_radius * math.cos(angle),
                hydrogen_radius * math.sin(angle),
                z,
            )
        )

    return atoms


def parallel_dimer(spacing: float) -> list[tuple[str, float, float, float]]:
    """Return two eclipsed, parallel benzene rings separated along z."""
    lower_ring = benzene_monomer(-spacing / 2.0)
    upper_ring = benzene_monomer(spacing / 2.0)
    return lower_ring + upper_ring


def format_xyz(atoms: list[tuple[str, float, float, float]], comment: str) -> str:
    lines = [str(len(atoms)), comment]
    lines.extend(
        f"{element:2s} {x: .8f} {y: .8f} {z: .8f}"
        for element, x, y, z in atoms
    )
    return "\n".join(lines) + "\n"


def spacing_label(spacing: float) -> str:
    return f"{spacing:.2f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate eclipsed, parallel benzene dimer XYZ structures for a "
            "pi-stacking separation scan."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Angel" / "Structure" / "C6H6_dimer",
        help="Directory for generated XYZ files.",
    )
    parser.add_argument(
        "--spacings",
        type=float,
        nargs="+",
        default=DEFAULT_SPACINGS,
        help="Ring-center separations in angstrom.",
    )
    parser.add_argument(
        "--combined-name",
        default="benzene_dimer_parallel_scan.xyz",
        help="Filename for the multi-frame XYZ scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: list[str] = []
    for spacing in args.spacings:
        atoms = parallel_dimer(spacing)
        comment = (
            f"parallel benzene dimer; ring_center_spacing={spacing:.2f} "
            f"{ANGSTROM}; eclipsed sandwich geometry"
        )
        xyz_text = format_xyz(atoms, comment)
        filename = f"benzene_dimer_parallel_{spacing_label(spacing)}A.xyz"
        (output_dir / filename).write_text(xyz_text, encoding="utf-8")
        combined_frames.append(xyz_text)

    (output_dir / args.combined_name).write_text("".join(combined_frames), encoding="utf-8")

    print(f"Wrote {len(args.spacings)} structures to {output_dir}")
    print(f"Wrote combined scan to {output_dir / args.combined_name}")


if __name__ == "__main__":
    main()
