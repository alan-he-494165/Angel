#!/usr/bin/env python3
"""Compute MACE, DFT, and active-space FCI energies for an XYZ scan."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HARTREE_TO_EV = 27.211386245981
EV_TO_KCAL_MOL = 23.060547830619
HARTREE_TO_KCAL_MOL = HARTREE_TO_EV * EV_TO_KCAL_MOL


@dataclass(frozen=True)
class ScanFrame:
    label: str
    atoms: object
    order_value: float


@dataclass(frozen=True)
class EnergyRow:
    index: int
    label: str
    order_value_angstrom: float
    mace_energy_ev: float
    dft_energy_hartree: float
    fci_energy_hartree: float
    mace_relative_kcal_mol: float
    dft_relative_kcal_mol: float
    fci_relative_kcal_mol: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CUDA MACE, wB97M-V/def2-TZVPD DFT, and 12-electron/12-orbital "
            "active-space FCI energies for an XYZ scan, then plot relative energies "
            "against the shortest-distance structure."
        )
    )
    parser.add_argument(
        "xyz",
        type=Path,
        nargs="?",
        help="XYZ file, multi-frame XYZ file, or directory of XYZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mace_fci_scan"),
        help="Directory for CSV and plot outputs.",
    )
    parser.add_argument(
        "--mace-model-path",
        type=Path,
        default=Path("mace_model/MACE-OFF23_large.model"),
        help="Path to a local MACE model. If omitted, installed MACE factory calculators are tried.",
    )
    parser.add_argument(
        "--mace-model",
        default="medium",
        help="Factory model name used when --mace-model-path is omitted.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for MACE. Defaults to cuda.",
    )
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA for MACE before running.",
    )
    parser.add_argument(
        "--fci-basis",
        default="def2-TZVPD",
        help="PySCF orbital basis for active-space FCI.",
    )
    parser.add_argument(
        "--active-electrons",
        type=int,
        default=12,
        help="Number of active electrons for FCI.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=int,
        default=12,
        help="Number of active spatial orbitals for FCI.",
    )
    parser.add_argument(
        "--dft-functional",
        default="wB97M-V",
        help="PySCF DFT functional. Defaults to wB97M-V.",
    )
    parser.add_argument(
        "--dft-basis",
        default="def2-TZVPD",
        help="PySCF DFT basis. Defaults to def2-TZVPD.",
    )
    parser.add_argument(
        "--dft-grid-level",
        type=int,
        default=3,
        help="PySCF numerical integration grid level for DFT.",
    )
    parser.add_argument(
        "--dft-conv-tol",
        type=float,
        default=1e-9,
        help="PySCF DFT SCF convergence tolerance.",
    )
    parser.add_argument(
        "--dft-max-cycle",
        type=int,
        default=100,
        help="Maximum number of PySCF DFT SCF iterations.",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge.",
    )
    parser.add_argument(
        "--spin",
        type=int,
        default=0,
        help="PySCF spin value, 2S.",
    )
    parser.add_argument(
        "--fci-cache-dir",
        type=Path,
        default=Path("results/electronic_fci_cache"),
        help="Cache directory for active-space Hamiltonians and CI vectors.",
    )
    parser.add_argument(
        "--no-fci-cache",
        action="store_true",
        help="Disable FCI cache reads and writes.",
    )
    parser.add_argument(
        "--fragment-split",
        type=int,
        default=None,
        help=(
            "Atom index where fragment B starts for ordering by center-of-mass distance. "
            "Defaults to half the atom count."
        ),
    )
    parser.add_argument(
        "--order",
        choices=("distance", "input"),
        default="distance",
        help="Sort structures by fragment center distance or preserve input order.",
    )
    parser.add_argument(
        "--plot-name",
        default="relative_energy_scan.png",
        help="Output plot filename.",
    )
    parser.add_argument(
        "--csv-name",
        default="relative_energy_scan.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--reuse-csv",
        type=Path,
        default=None,
        help="Existing CSV for reused energies; defaults to output-dir/csv-name.",
    )
    parser.add_argument(
        "--reuse-mace",
        action="store_true",
        help="Reuse MACE energies from the existing CSV instead of recalculating.",
    )
    parser.add_argument(
        "--reuse-dft",
        action="store_true",
        help="Reuse DFT energies from the existing CSV instead of recalculating.",
    )
    parser.add_argument(
        "--reuse-fci",
        action="store_true",
        help="Reuse FCI energies from the existing CSV instead of recalculating.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    return [float(item) if re.fullmatch(r"\d+(?:\.\d+)?", item) else item.lower() for item in re.split(r"(\d+(?:\.\d+)?)", path.name)]


def load_scan_frames(path: Path, fragment_split: int | None, order: str) -> list[ScanFrame]:
    try:
        from ase.io import read
    except ImportError as error:
        raise ImportError("This script requires ASE: install the 'ase' package.") from error

    source = path.expanduser()
    if source.is_dir():
        atoms_items = []
        for xyz_path in sorted(source.glob("*.xyz"), key=natural_key):
            frames = read(xyz_path, index=":")
            atoms_items.extend((f"{xyz_path.stem}:{index}", atoms) for index, atoms in enumerate(frames))
    else:
        frames = read(source, index=":")
        atoms_items = [(f"{source.stem}:{index}", atoms) for index, atoms in enumerate(frames)]

    scan_frames = [
        ScanFrame(label=label, atoms=atoms, order_value=ordering_value(atoms, fragment_split))
        for label, atoms in atoms_items
    ]
    if order == "distance":
        return sorted(scan_frames, key=lambda item: item.order_value)
    return scan_frames


def ordering_value(atoms: object, fragment_split: int | None) -> float:
    import numpy as np

    positions = np.asarray(atoms.get_positions(), dtype=float)
    natoms = len(positions)
    split = fragment_split if fragment_split is not None else natoms // 2
    if split <= 0 or split >= natoms:
        raise ValueError(f"fragment split must be between 1 and {natoms - 1}; got {split}.")
    first = positions[:split].mean(axis=0)
    second = positions[split:].mean(axis=0)
    return float(np.linalg.norm(second - first))


def build_mace_calculator(args: argparse.Namespace):
    if args.require_cuda:
        try:
            import torch
        except ImportError as error:
            raise ImportError("CUDA was requested, but PyTorch is not installed.") from error
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for MACE, but torch.cuda.is_available() is false.")

    try:
        from mace.calculators import MACECalculator
    except ImportError:
        MACECalculator = None

    if args.mace_model_path is not None:
        if MACECalculator is None:
            raise ImportError("Loading --mace-model-path requires mace.calculators.MACECalculator.")
        return MACECalculator(
            model_paths=str(args.mace_model_path.expanduser()),
            device=args.device,
            default_dtype="float64",
        )

    errors: list[str] = []
    for factory_name in ("mace_off", "mace_mp"):
        try:
            module = __import__("mace.calculators", fromlist=[factory_name])
            factory = getattr(module, factory_name)
            return factory(model=args.mace_model, device=args.device, default_dtype="float64")
        except Exception as error:
            errors.append(f"{factory_name}: {error}")
    detail = "\n".join(errors)
    raise RuntimeError(
        "Could not construct a MACE calculator. Pass --mace-model-path or install a "
        f"MACE package with a supported factory calculator.\n{detail}"
    )


def compute_mace_energies(frames: Iterable[ScanFrame], calculator: object) -> list[float]:
    energies = []
    for frame in frames:
        frame.atoms.calc = calculator
        energies.append(float(frame.atoms.get_potential_energy()))
    return energies


def atoms_to_atom_string(atoms: object) -> str:
    import numpy as np

    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.get_positions(), dtype=float)
    return "\n".join(
        f"{symbol} {x:.14g} {y:.14g} {z:.14g}"
        for symbol, (x, y, z) in zip(symbols, positions)
    )


def compute_dft_energies(frames: Iterable[ScanFrame], args: argparse.Namespace) -> list[float]:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from angel_src.electronic.fci import gpu4pyscf_available
    except ImportError:
        gpu4pyscf_available = lambda: False

    use_gpu = gpu4pyscf_available()
    try:
        from pyscf import dft, gto
    except ImportError as error:
        raise ImportError("DFT evaluation requires PySCF.") from error

    backend = "GPU4PySCF" if use_gpu else "CPU PySCF"
    print(f"DFT backend: {backend}", flush=True)
    energies = []
    for index, frame in enumerate(frames, start=1):
        mol = gto.M(
            atom=atoms_to_atom_string(frame.atoms),
            basis=args.dft_basis,
            charge=args.charge,
            spin=args.spin,
            unit="Angstrom",
            symmetry=False,
            verbose=0,
        )
        mf = dft.RKS(mol)
        mf.xc = args.dft_functional
        mf.grids.level = args.dft_grid_level
        mf.conv_tol = args.dft_conv_tol
        mf.max_cycle = args.dft_max_cycle
        if use_gpu:
            try:
                mf = mf.to_gpu()
            except Exception as error:
                use_gpu = False
                backend = "CPU PySCF"
                print(
                    f"GPU4PySCF setup failed for {frame.label}; falling back to CPU: {error}",
                    flush=True,
                )
                mf = dft.RKS(mol)
                mf.xc = args.dft_functional
                mf.grids.level = args.dft_grid_level
                mf.conv_tol = args.dft_conv_tol
                mf.max_cycle = args.dft_max_cycle
        print(f"  DFT structure {index}/{len(frames)} using {backend}: {frame.label}", flush=True)
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError(f"DFT calculation did not converge for {frame.label}.")
        energies.append(float(energy))
    return energies


def compute_fci_energy(frame: ScanFrame, args: argparse.Namespace) -> float:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from angel_src.electronic.fci import get_electronic_structure

    electronic = get_electronic_structure(
        atoms_to_atom_string(frame.atoms),
        args.fci_basis,
        molecule_name=frame.label,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
        charge=args.charge,
        spin=args.spin,
        calculate_energy=True,
        use_cache=not args.no_fci_cache,
        cache_dir=args.fci_cache_dir,
        use_gpu=True,
    )
    if electronic.energy_hartree is None:
        raise RuntimeError(f"FCI energy was not returned for {frame.label}.")
    return float(electronic.energy_hartree)


def build_rows(
    frames: list[ScanFrame],
    mace_energies: list[float],
    dft_energies: list[float],
    fci_energies: list[float],
) -> list[EnergyRow]:
    mace_reference = mace_energies[0]
    dft_reference = dft_energies[0]
    fci_reference = fci_energies[0]
    rows = []
    for index, (frame, mace_energy, dft_energy, fci_energy) in enumerate(
        zip(frames, mace_energies, dft_energies, fci_energies)
    ):
        rows.append(
            EnergyRow(
                index=index,
                label=frame.label,
                order_value_angstrom=frame.order_value,
                mace_energy_ev=mace_energy,
                dft_energy_hartree=dft_energy,
                fci_energy_hartree=fci_energy,
                mace_relative_kcal_mol=(mace_energy - mace_reference) * EV_TO_KCAL_MOL,
                dft_relative_kcal_mol=(dft_energy - dft_reference) * HARTREE_TO_KCAL_MOL,
                fci_relative_kcal_mol=(fci_energy - fci_reference) * HARTREE_TO_KCAL_MOL,
            )
        )
    return rows


def load_reused_energies(
    path: Path,
    frames: list[ScanFrame],
    column: str,
    method_name: str,
) -> list[float]:
    if not path.exists():
        raise FileNotFoundError(f"Reuse CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"Reuse CSV {path} has no '{column}' column for {method_name}.")
        rows = list(reader)
    by_label = {row["label"]: row for row in rows if row.get("label")}
    missing = [frame.label for frame in frames if frame.label not in by_label]
    if missing:
        raise ValueError(f"Reuse CSV {path} is missing structures for {method_name}: {missing}")
    return [float(by_label[frame.label][column]) for frame in frames]


def load_rows_from_csv(path: Path) -> list[EnergyRow]:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = set(EnergyRow.__dataclass_fields__)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV {path} is missing columns: {sorted(missing)}")
        return [
            EnergyRow(
                index=int(row["index"]),
                label=row["label"],
                order_value_angstrom=float(row["order_value_angstrom"]),
                mace_energy_ev=float(row["mace_energy_ev"]),
                dft_energy_hartree=float(row["dft_energy_hartree"]),
                fci_energy_hartree=float(row["fci_energy_hartree"]),
                mace_relative_kcal_mol=float(row["mace_relative_kcal_mol"]),
                dft_relative_kcal_mol=float(row["dft_relative_kcal_mol"]),
                fci_relative_kcal_mol=float(row["fci_relative_kcal_mol"]),
            )
            for row in reader
        ]


def write_csv(rows: list[EnergyRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(EnergyRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_rows(rows: list[EnergyRow], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Plotting requires matplotlib.") from error

    x = [row.order_value_angstrom for row in rows]
    plt.figure(figsize=(7.0, 4.5))
    plt.plot(x, [row.mace_relative_kcal_mol for row in rows], marker="o", label="MACE")
    plt.plot(
        x,
        [row.dft_relative_kcal_mol for row in rows],
        marker="^",
        label="wB97M-V/def2-TZVPD",
    )
    plt.plot(x, [row.fci_relative_kcal_mol for row in rows], marker="s", label="FCI(12e,12o)")
    plt.axhline(0.0, color="0.3", linewidth=0.8)
    plt.xlabel("Fragment center distance (Angstrom)")
    plt.ylabel("Relative energy vs first structure (kcal/mol)")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    csv_path = output_dir / args.csv_name
    plot_path = output_dir / args.plot_name
    reuse_csv = (args.reuse_csv or csv_path).expanduser().resolve()
    reuse_all = args.reuse_mace and args.reuse_dft and args.reuse_fci

    if args.xyz is None:
        if not reuse_all:
            raise ValueError("Provide an XYZ input unless --reuse-mace, --reuse-dft, and --reuse-fci are all set.")
        rows = load_rows_from_csv(reuse_csv)
    else:
        frames = load_scan_frames(args.xyz, args.fragment_split, args.order)
        if not frames:
            raise ValueError(f"No XYZ frames found in {args.xyz}.")

        if args.reuse_dft:
            dft_energies = load_reused_energies(reuse_csv, frames, "dft_energy_hartree", "DFT")
        else:
            print(
                f"Computing {args.dft_functional}/{args.dft_basis} DFT energies for "
                f"{len(frames)} structures...",
                flush=True,
            )
            dft_energies = compute_dft_energies(frames, args)

        if args.reuse_mace:
            mace_energies = load_reused_energies(reuse_csv, frames, "mace_energy_ev", "MACE")
        else:
            calculator = build_mace_calculator(args)
            print(f"Computing MACE energies for {len(frames)} structures on {args.device}...", flush=True)
            mace_energies = compute_mace_energies(frames, calculator)

        if args.reuse_fci:
            fci_energies = load_reused_energies(reuse_csv, frames, "fci_energy_hartree", "FCI")
        else:
            fci_energies = []
            for index, frame in enumerate(frames, start=1):
                print(
                    f"Computing active-space FCI energy {index}/{len(frames)} "
                    f"({frame.label}, separation={frame.order_value:.3f} Angstrom)...",
                    flush=True,
                )
                fci_energies.append(compute_fci_energy(frame, args))

        rows = build_rows(frames, mace_energies, dft_energies, fci_energies)

    write_csv(rows, csv_path)
    plot_rows(rows, plot_path)

    print(f"Reference structure: {rows[0].label}")
    print(f"Wrote energies to {csv_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
