#!/usr/bin/env python3
"""Validate the FiRE solver on small all-electron systems.

For each system the script runs the label-free variational optimization and
writes one row per system to ``fire_validation.csv`` plus the full energy
traces to ``fire_validation_traces.csv``.  PySCF HF/FCI energies in a finite
basis and the FCI/CBS ground truth (scripts/fci_reference.py) are recorded in the same
table for context only - neither is used during the optimization.

Usage::

    python scripts/validate_fire.py --out out/fire_validation --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The comparison target is full CI at the complete basis set limit, produced by
# scripts/fci_reference.py and read from --reference-csv.  The literature
# non-relativistic energies below are kept only as an independent check on that
# extrapolation and are never the target.
SYSTEMS = [
    {
        "name": "H2",
        "symbols": ["H", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        "charge": 0,
        "spin": 0,
        "exact_hartree": -1.174476,
        "exact_source": "Kolos-Wolniewicz, R = 1.4 a0",
    },
    {
        "name": "LiH",
        "symbols": ["Li", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]],
        "charge": 0,
        "spin": 0,
        "exact_hartree": -8.070548,
        "exact_source": "Cencek-Rychlewski, R = 3.015 a0",
    },
    {
        "name": "Be",
        "symbols": ["Be"],
        "coords": [[0.0, 0.0, 0.0]],
        "charge": 0,
        "spin": 0,
        "exact_hartree": -14.667356,
        "exact_source": "Chakravorty et al. 1993",
    },
]


def load_fci_reference(path: Path) -> dict[str, dict]:
    """Read the FCI/CBS table written by scripts/fci_reference.py."""

    import csv as _csv

    if not path.exists():
        raise SystemExit(
            f"FCI reference table {path} not found - run scripts/fci_reference.py first, "
            "or point --reference-csv at an existing table."
        )
    with path.open() as handle:
        return {row["system"]: row for row in _csv.DictReader(handle)}


def reference_energies(system, basis: str, ecp: str | None = None) -> tuple[float, float]:
    """Return (HF, FCI) energies in ``basis`` - context only, never a label."""

    from pyscf import fci, gto, scf

    mol = gto.Mole()
    mol.atom = system.as_pyscf_atom()
    mol.unit = "Bohr"
    mol.basis = basis
    mol.charge = system.charge
    mol.spin = system.spin
    if ecp is not None:
        mol.ecp = ecp
    mol.verbose = 0
    mol.build()
    mean_field = scf.RHF(mol).run()
    solver = fci.FCI(mean_field)
    solver.max_cycle = 200
    fci_energy = solver.kernel()[0]
    return float(mean_field.e_tot), float(fci_energy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("out/fire_validation"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--determinants", type=int, default=4)
    parser.add_argument("--pretrain-steps", type=int, default=200)
    parser.add_argument("--basis", type=str, default="cc-pvdz", help="Basis for the HF/FCI context values.")
    parser.add_argument("--pretrain-basis", type=str, default="cc-pvdz")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="SPRING base learning rate lr_0; defaults to the OptConfig value.",
    )
    parser.add_argument(
        "--lr-decay-time",
        type=float,
        default=None,
        help="Learning-rate decay time; defaults to steps/4 so that short CPU runs settle.",
    )
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("out/fci_reference.csv"),
        help="FCI/CBS ground-truth table from scripts/fci_reference.py.",
    )
    parser.add_argument(
        "--ecp",
        type=str,
        default=None,
        help=(
            "Effective core potential family (e.g. ccecp).  Off by default: "
            "the all-electron path is the reference one.  With an ECP the "
            "context basis and the FCI reference table must both be the "
            "matching ECP ones, since the total energies are on a different "
            "scale from the all-electron values."
        ),
    )
    parser.add_argument(
        "--clip-width",
        type=float,
        default=None,
        help=(
            "Local-energy clipping width in MAD units for the gradient; "
            "defaults to the OptConfig value (5).  ECP runs are heavy-tailed "
            "early in training and are more robust at 2."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-blocks", type=int, default=0)
    parser.add_argument("--eval-block-steps", type=int, default=10)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated subset of system names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ecp is not None:
        # Keep the whole comparison on one Hamiltonian: context basis,
        # pretraining basis and reference table all follow the ECP.
        if not args.basis.startswith(args.ecp):
            args.basis = f"{args.ecp}-{args.basis}"
        if args.reference_csv == Path("out/fci_reference.csv"):
            args.reference_csv = Path(f"out/fci_reference_{args.ecp}.csv")

    from dataclasses import replace

    from angel_src.NQS.fire.config import small_config
    from angel_src.NQS.fire.solver import run_fire_vmc
    from angel_src.NQS.fire.system import system_from_arrays

    fci_reference = load_fci_reference(args.reference_csv)
    wanted = set(args.only.split(",")) if args.only else None
    rows: list[dict] = []
    traces: list[dict] = []

    for spec in SYSTEMS:
        if wanted is not None and spec["name"] not in wanted:
            continue
        system = system_from_arrays(
            spec["symbols"],
            np.asarray(spec["coords"], dtype=float),
            charge=spec["charge"],
            spin=spec["spin"],
            units="bohr",
            name=spec["name"],
            ecp=args.ecp,
        )
        base = small_config()
        config = replace(
            base,
            ansatz=replace(base.ansatz, n_determinants=args.determinants),
            mcmc=replace(base.mcmc, batch_size=args.batch_size),
            opt=replace(
                base.opt,
                steps=args.steps,
                n_eval_steps=max(args.steps // 10, 10),
                learning_rate=args.learning_rate if args.learning_rate is not None else base.opt.learning_rate,
                lr_decay_time=args.lr_decay_time or max(args.steps / 4.0, 1.0),
                clip_width=args.clip_width if args.clip_width is not None else base.opt.clip_width,
                eval_blocks=args.eval_blocks,
                eval_block_steps=args.eval_block_steps,
            ),
            pretrain=replace(base.pretrain, steps=args.pretrain_steps, basis=args.pretrain_basis),
            ecp=args.ecp,
            seed=args.seed,
            log_every=max(args.steps // 10, 1),
        )

        hf_energy, fci_energy = reference_energies(system, args.basis, args.ecp)
        print(f"[{spec['name']}] HF/{args.basis} = {hf_energy:.6f}  FCI/{args.basis} = {fci_energy:.6f}", flush=True)

        start = time.perf_counter()
        result = run_fire_vmc(
            system,
            config,
            pretrain=args.pretrain_steps > 0,
            output_dir=args.out / spec["name"],
            callback=lambda step, record: print(
                f"[{spec['name']}] {step:6d} " + " ".join(f"{k}={v:.5f}" for k, v in record.items()),
                flush=True,
            ),
        )
        elapsed = time.perf_counter() - start

        if args.checkpoint_dir is not None:
            from angel_src.NQS.fire.checkpoint import save_params

            save_params(
                args.checkpoint_dir / spec["name"], result.params,
                config=config, system=system,
            )

        reference_row = fci_reference[spec["name"]]
        exact = float(reference_row["fci_cbs_hartree"])
        hf_cbs = float(reference_row["hf_cbs_hartree"])
        rows.append(
            {
                "system": spec["name"],
                "n_electrons": system.n_electrons,
                "fire_energy_hartree": result.energy_hartree,
                "fire_error_hartree": result.energy_error_hartree,
                "fire_variance_hartree2": result.energy_variance_hartree2,
                "eval_energy_hartree": result.energy_eval_hartree,
                "eval_error_hartree": result.energy_eval_error_hartree,
                "eval_blocks": result.eval_blocks_used,
                "eval_autocorr_time": result.eval_autocorr_time,
                "eval_minus_reference_mhartree": (
                    (result.energy_eval_hartree - exact) * 1e3
                    if result.eval_blocks_used else float("nan")
                ),
                "fire_extrapolated_hartree": result.energy_extrapolated_hartree,
                "hf_basis_hartree": hf_energy,
                "fci_basis_hartree": fci_energy,
                "reference_hartree": exact,
                "reference_kind": "FCI/CBS composite (scripts/fci_reference.py)",
                # Under an ECP there is no all-electron literature value to
                # compare the composite against, so the extrapolation
                # uncertainty is left blank rather than faked.
                "reference_uncertainty_mhartree": (
                    abs(float(reference_row["cbs_minus_literature_mhartree"]))
                    if reference_row["cbs_minus_literature_mhartree"]
                    else ""
                ),
                "hf_cbs_hartree": hf_cbs,
                "literature_hartree": (
                    float(reference_row["literature_hartree"])
                    if reference_row["literature_hartree"]
                    else ""
                ),
                "literature_source": (
                    spec["exact_source"] if args.ecp is None else reference_row["literature_source"]
                ),
                "basis": args.basis,
                "core_treatment": args.ecp or "all-electron",
                "clip_width": config.opt.clip_width,
                "fire_minus_reference_mhartree": (result.energy_hartree - exact) * 1e3,
                "fire_extrap_minus_reference_mhartree": (result.energy_extrapolated_hartree - exact) * 1e3,
                "correlation_recovered_percent": 100.0
                * (hf_cbs - result.energy_hartree)
                / (hf_cbs - exact),
                "steps": args.steps,
                "learning_rate": config.opt.learning_rate,
                "lr_decay_time": config.opt.lr_decay_time,
                "pretrain_steps": args.pretrain_steps,
                "batch_size": args.batch_size,
                "n_determinants": args.determinants,
                "wall_time_seconds": round(elapsed, 1),
            }
        )
        for step, energy, variance, gradient in zip(
            result.history["step"],
            result.history["energy_hartree"],
            result.history["variance_hartree2"],
            result.history["grad_norm_sq"],
        ):
            traces.append(
                {
                    "system": spec["name"],
                    "step": step,
                    "energy_hartree": energy,
                    "variance_hartree2": variance,
                    "grad_norm_sq": gradient,
                    "reference_hartree": exact,
                }
            )

    table_path = args.out / "fire_validation.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    trace_path = args.out / "fire_validation_traces.csv"
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)

    print(f"\nwrote {table_path} and {trace_path}")
    for row in rows:
        eval_text = (
            f", eval_energy_hartree = {row['eval_energy_hartree']:.6f}"
            if row["eval_blocks"] else ""
        )
        print(
            f"{row['system']:>5}  E = {row['fire_energy_hartree']:.6f}  "
            f"({row['fire_minus_reference_mhartree']:+.2f} mEh vs FCI/CBS, "
            f"{row['correlation_recovered_percent']:.1f} % correlation)"
            f"{eval_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
