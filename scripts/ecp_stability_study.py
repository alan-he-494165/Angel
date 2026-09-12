#!/usr/bin/env python3
"""Robustness of the FiRE optimization with and without an ECP.

The ECP option is correct at the estimator level -- ``scripts/
validate_ecp_operator.py`` reproduces PySCF's own ECP integrals on a
Hartree-Fock trial to well under a milli-hartree -- but the nonlocal
projector evaluates ``Psi(r')/Psi(r)`` on a quadrature sphere, and a
neural ansatz early in training can put a spurious node close to the
sampled configuration.  The resulting local-energy tail makes the
*optimization*, not the estimator, the fragile part.

This script measures that fragility instead of asserting it: the same
systems are optimized all-electron and with ccECP over a small grid of
seeds and local-energy clipping widths, and every run is recorded with
its reference-consistent error.  Energies under the two Hamiltonians are
not comparable in absolute terms, so the comparable columns are the
error against the matching FCI/CBS reference and the fraction of the
correlation energy recovered.

Usage:
    PYTHONPATH=. python3 scripts/ecp_stability_study.py --out out/ecp_stability.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

SYSTEMS = {
    "LiH": (["Li", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]], 0, 0),
    "Be": (["Be"], [[0.0, 0.0, 0.0]], 0, 0),
}


def load_reference(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        return {row["system"]: row for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/ecp_stability.csv"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--pretrain-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--determinants", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="1234,7")
    parser.add_argument("--clip-widths", type=str, default="2,5")
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    from angel_src.NQS.fire.config import replace_config, small_config
    from angel_src.NQS.fire.solver import run_fire_vmc
    from angel_src.NQS.fire.system import system_from_arrays

    references = {
        "all-electron": load_reference(Path("out/fci_reference.csv")),
        "ccecp": load_reference(Path("out/fci_reference_ccecp.csv")),
    }
    seeds = [int(s) for s in args.seeds.split(",")]
    clips = [float(c) for c in args.clip_widths.split(",")]
    names = args.only.split(",") if args.only else list(SYSTEMS)

    base = small_config()
    rows: list[dict] = []
    for name in names:
        symbols, coords, charge, spin = SYSTEMS[name]
        for treatment in ("all-electron", "ccecp"):
            ecp = None if treatment == "all-electron" else treatment
            reference_row = references[treatment][name]
            exact = float(reference_row["fci_cbs_hartree"])
            hf_cbs = float(reference_row["hf_cbs_hartree"])
            system = system_from_arrays(
                symbols, np.asarray(coords, float), charge=charge, spin=spin,
                units="bohr", name=name, ecp=ecp,
            )
            for clip in clips:
                for seed in seeds:
                    config = replace_config(
                        base,
                        ecp=ecp,
                        seed=seed,
                        ansatz=replace(base.ansatz, n_determinants=args.determinants),
                        mcmc=replace(base.mcmc, batch_size=args.batch_size),
                        opt=replace(
                            base.opt,
                            steps=args.steps,
                            n_eval_steps=max(args.steps // 10, 10),
                            lr_decay_time=args.steps / 8.0,
                            clip_width=clip,
                        ),
                        pretrain=replace(base.pretrain, steps=args.pretrain_steps),
                    )
                    result = run_fire_vmc(system, config, pretrain=True)
                    row = {
                        "system": name,
                        "core_treatment": treatment,
                        "n_electrons": system.n_electrons,
                        "clip_width": clip,
                        "seed": seed,
                        "energy_hartree": result.energy_hartree,
                        "error_hartree": result.energy_error_hartree,
                        "variance_hartree2": result.energy_variance_hartree2,
                        "reference_hartree": exact,
                        "hf_cbs_hartree": hf_cbs,
                        "error_vs_reference_mhartree": (result.energy_hartree - exact) * 1e3,
                        "correlation_recovered_percent":
                            100.0 * (hf_cbs - result.energy_hartree) / (hf_cbs - exact),
                        "above_hartree_fock": bool(result.energy_hartree > hf_cbs),
                    }
                    rows.append(row)
                    print(
                        "%-4s %-12s clip=%.0f seed=%-5d E=%12.5f  dE=%+9.2f mEh  corr=%6.1f %%"
                        % (name, treatment, clip, seed, result.energy_hartree,
                           row["error_vs_reference_mhartree"],
                           row["correlation_recovered_percent"]),
                        flush=True,
                    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
