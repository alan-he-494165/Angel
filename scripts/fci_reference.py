#!/usr/bin/env python3
"""Full-CI ground-truth energies at the complete basis set limit.

Why not simply full CI in one basis
-----------------------------------
The FiRE solver works in real space and is not confined to any orbital basis,
so full CI in a *fixed* finite basis is not a valid ground truth for it: the
real-space ansatz can fall below FCI/cc-pVDZ without being wrong.  The
well-defined FCI reference is the complete-basis-set limit of full CI, which
is what this script builds.

Composite construction (all-electron, clamped nuclei, non-relativistic)
-----------------------------------------------------------------------
Direct all-electron FCI in cc-pVQZ is affordable only for the smallest cases,
and the valence-only cc-pVXZ series converges slowly for core correlation, so
E_FCI(CBS) is assembled from three separately converged pieces:

  E_HF(CBS)        three-point exponential extrapolation, E(X) = E_cbs +
                   A exp(-B X), over the cc-pVXZ Hartree-Fock energies
                   (Feller 1992).

  E_corr,val(CBS)  frozen-core full CI (CASCI with the complete virtual
                   space) in cc-pVTZ and cc-pVQZ, extrapolated with the
                   two-point X^-3 formula of Helgaker et al. 1997.

  dE_core          core and core-valence correlation, taken as the
                   difference between all-electron and frozen-core full CI
                   in a core-valence basis (cc-pCVTZ on the non-hydrogen
                   atoms, cc-pVTZ on hydrogen).  Zero for systems without a
                   core.

  E_FCI(CBS) = E_HF(CBS) + E_corr,val(CBS) + dE_core

Every piece is full CI within its orbital space -- no truncation of the
excitation level anywhere.  Literature non-relativistic energies are carried
in the output purely as an independent check on the extrapolation; the
comparison target for the solver is the FCI/CBS column.

Usage
-----
    python scripts/fci_reference.py --out out/fci_reference.csv
    python scripts/fci_reference.py --only H2
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np

SYSTEMS = {
    "H2": {
        "atom": [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))],
        "charge": 0,
        "spin": 0,
        "n_core": 0,
        "core_basis": None,
        "hf_bases": ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"],
        "valence_bases": ["cc-pvtz", "cc-pvqz"],
        "literature_hartree": -1.174476,
        "literature_source": "Kolos-Wolniewicz (clamped nuclei, R = 1.4 a0)",
    },
    "LiH": {
        "atom": [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 3.015))],
        "charge": 0,
        "spin": 0,
        "n_core": 1,
        "core_basis": {"Li": "cc-pcvtz", "H": "cc-pvtz"},
        "hf_bases": ["cc-pvdz", "cc-pvtz", "cc-pvqz"],
        "valence_bases": ["cc-pvtz", "cc-pvqz"],
        "literature_hartree": -8.070548,
        "literature_source": "Cencek-Rychlewski (ECG, R = 3.015 a0)",
    },
    "Be": {
        "atom": [("Be", (0.0, 0.0, 0.0))],
        "charge": 0,
        "spin": 0,
        "n_core": 1,
        "core_basis": {"Be": "cc-pcvtz"},
        "hf_bases": ["cc-pvdz", "cc-pvtz", "cc-pvqz"],
        "valence_bases": ["cc-pvtz", "cc-pvqz"],
        "literature_hartree": -14.667356,
        "literature_source": "Chakravorty et al. 1993 (estimated NR limit)",
    },
}

CARDINAL = {"cc-pvdz": 2, "cc-pvtz": 3, "cc-pvqz": 4, "cc-pv5z": 5}

# ECP basis names are the all-electron ones with a family prefix
# (``ccecp-cc-pvtz``), so the cardinal number is read off the tail.
_CARDINAL_TAIL = {"d": 2, "t": 3, "q": 4, "5": 5}


def cardinal(basis: str) -> int:
    """Cardinal number X of a correlation-consistent basis, ECP or not."""

    if basis in CARDINAL:
        return CARDINAL[basis]
    match = re.search(r"pv(d|t|q|5)z$", basis.lower())
    if match is None:
        raise KeyError(f"cannot read a cardinal number from basis {basis!r}")
    return _CARDINAL_TAIL[match.group(1)]


def ecp_variant(spec: dict, ecp: str) -> dict:
    """Recast an all-electron system spec onto an ECP Hamiltonian.

    The ECP removes the core explicitly, so nothing is frozen in the CI and
    the core-correlation correction is not applicable.  The resulting total
    energies are *not* comparable with all-electron literature values -- they
    belong to a different Hamiltonian -- so the literature column is dropped
    and the correlation energy is reported instead, which is the quantity
    that should agree with the all-electron valence result.
    """

    variant = dict(spec)
    variant["ecp"] = ecp
    variant["hf_bases"] = [f"{ecp}-{basis}" for basis in spec["hf_bases"]]
    variant["valence_bases"] = [f"{ecp}-{basis}" for basis in spec["valence_bases"]]
    variant["n_core"] = 0
    variant["core_basis"] = None
    variant["literature_hartree"] = None
    variant["literature_source"] = (
        f"not applicable: {ecp} defines a different Hamiltonian from the "
        "all-electron non-relativistic limit"
    )
    return variant


def _mol(spec: dict, basis):
    from pyscf import gto

    return gto.M(
        atom=[(s, tuple(c)) for s, c in spec["atom"]],
        unit="Bohr",
        basis=basis,
        charge=spec["charge"],
        spin=spec["spin"],
        ecp=spec.get("ecp"),
        max_memory=10000,
        verbose=0,
    )


def hartree_fock(spec: dict, basis) -> tuple[float, int, object]:
    from pyscf import scf

    mol = _mol(spec, basis)
    mf = scf.RHF(mol).run()
    return float(mf.e_tot), int(mol.nao_nr()), mf


def fci_energy(mf, n_core: int) -> float:
    """Full CI in the orbital space left after freezing ``n_core`` orbitals.

    ``n_core = 0`` is all-electron full CI.  Freezing is done with CASCI over
    the complete remaining virtual space, which is full CI for the correlated
    electrons, not a truncated active space.
    """
    from pyscf import fci, mcscf

    if n_core == 0:
        solver = fci.FCI(mf)
        solver.max_memory = 10000
        solver.conv_tol = 1e-10
        return float(solver.kernel()[0])
    n_orb = mf.mo_coeff.shape[1]
    n_elec = mf.mol.nelectron
    casci = mcscf.CASCI(mf, n_orb - n_core, n_elec - 2 * n_core)
    casci.fcisolver.conv_tol = 1e-10
    casci.max_memory = 10000
    return float(casci.kernel()[0])


def extrapolate_hf(cards: np.ndarray, energies: np.ndarray) -> float:
    """Three-point exponential extrapolation E(X) = E_cbs + A exp(-B X)."""
    if len(cards) < 3:
        return float(energies[-1])
    x1, x2, x3 = cards[-3:]
    e1, e2, e3 = energies[-3:]
    if not np.isclose(x2 - x1, x3 - x2):
        return float(e3)
    denominator = (e3 - e2) - (e2 - e1)
    if abs(denominator) < 1e-12:
        return float(e3)
    return float(e3 - (e3 - e2) ** 2 / denominator)


def extrapolate_corr(cards: np.ndarray, energies: np.ndarray) -> float:
    """Two-point X^-3 extrapolation of a correlation energy."""
    y, x = cards[-2], cards[-1]
    ey, ex = energies[-2], energies[-1]
    return float((x**3 * ex - y**3 * ey) / (x**3 - y**3))


def build_reference(name: str, spec: dict, log) -> tuple[dict, list[dict]]:
    detail: list[dict] = []
    hf_energies, hf_cards = [], []
    mfs: dict[str, object] = {}

    for basis in spec["hf_bases"]:
        t0 = time.time()
        energy, nao, mf = hartree_fock(spec, basis)
        mfs[basis] = mf
        hf_energies.append(energy)
        hf_cards.append(cardinal(basis))
        detail.append(
            {"system": name, "quantity": "HF", "basis": basis, "n_orbitals": nao,
             "energy_hartree": energy, "seconds": round(time.time() - t0, 1)}
        )
        log(f"{name:>4} HF   {basis:<9} nao={nao:>3}  {energy:.8f}")

    corr_val, corr_cards = [], []
    for basis in spec["valence_bases"]:
        t0 = time.time()
        mf = mfs[basis]
        energy = fci_energy(mf, spec["n_core"])
        corr = energy - float(mf.e_tot)
        corr_val.append(corr)
        corr_cards.append(cardinal(basis))
        detail.append(
            {"system": name, "quantity": "FCI_frozen_core", "basis": basis,
             "n_orbitals": int(mf.mo_coeff.shape[1]), "energy_hartree": energy,
             "seconds": round(time.time() - t0, 1)}
        )
        log(f"{name:>4} FCI  {basis:<9} (valence) {energy:.8f}  Ecorr={corr:.8f}"
            f"  [{time.time() - t0:.0f} s]")

    delta_core = 0.0
    if spec["n_core"]:
        t0 = time.time()
        _, nao, mf = hartree_fock(spec, spec["core_basis"])
        e_all = fci_energy(mf, 0)
        e_fc = fci_energy(mf, spec["n_core"])
        delta_core = e_all - e_fc
        label = "/".join(sorted(set(spec["core_basis"].values())))
        detail += [
            {"system": name, "quantity": "FCI_all_electron", "basis": label,
             "n_orbitals": nao, "energy_hartree": e_all,
             "seconds": round(time.time() - t0, 1)},
            {"system": name, "quantity": "FCI_frozen_core", "basis": label,
             "n_orbitals": nao, "energy_hartree": e_fc, "seconds": 0.0},
        ]
        log(f"{name:>4} core {label:<9} nao={nao:>3}  dE_core={delta_core:.8f}"
            f"  [{time.time() - t0:.0f} s]")

    hf_cbs = extrapolate_hf(np.array(hf_cards, float), np.array(hf_energies))
    corr_cbs = extrapolate_corr(np.array(corr_cards, float), np.array(corr_val))
    fci_cbs = hf_cbs + corr_cbs + delta_core
    summary = {
        "system": name,
        "hf_cbs_hartree": hf_cbs,
        "corr_valence_cbs_hartree": corr_cbs,
        "delta_core_hartree": delta_core,
        "fci_cbs_hartree": fci_cbs,
        "hf_bases": "/".join(spec["hf_bases"]),
        "valence_bases": "/".join(spec["valence_bases"]),
        "core_basis": "/".join(sorted(set(spec["core_basis"].values()))) if spec["core_basis"] else "",
        "literature_hartree": spec["literature_hartree"],
        "literature_source": spec["literature_source"],
        "cbs_minus_literature_mhartree": (
            (fci_cbs - spec["literature_hartree"]) * 1e3
            if spec["literature_hartree"] is not None
            else None
        ),
        "core_treatment": spec.get("ecp") or "all-electron",
    }
    if spec["literature_hartree"] is None:
        log(f"{name:>4} FCI/CBS = {fci_cbs:.6f}   "
            f"(E_corr = {corr_cbs:.6f}; no literature value for this Hamiltonian)")
    else:
        log(f"{name:>4} FCI/CBS = {fci_cbs:.6f}   "
            f"(literature {spec['literature_hartree']:.6f}, "
            f"{summary['cbs_minus_literature_mhartree']:+.2f} mEh)")
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--only", default=None, help="comma separated subset")
    parser.add_argument(
        "--ecp",
        default=None,
        help=(
            "Effective core potential family (e.g. ccecp).  Builds the "
            "reference for the ECP Hamiltonian in the matching basis series "
            "instead of the all-electron one; the two sets of total energies "
            "are not comparable."
        ),
    )
    args = parser.parse_args()
    if args.out is None:
        args.out = (
            "out/fci_reference.csv"
            if args.ecp is None
            else f"out/fci_reference_{args.ecp}.csv"
        )

    def log(message: str) -> None:
        print(message, flush=True)

    names = list(SYSTEMS) if args.only is None else args.only.split(",")
    summaries, details = [], []
    for name in names:
        spec = SYSTEMS[name]
        if args.ecp is not None:
            spec = ecp_variant(spec, args.ecp)
        summary, detail = build_reference(name, spec, log)
        summaries.append(summary)
        details.extend(detail)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    detail_path = out.with_name(out.stem + "_components.csv")
    with detail_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    log(f"written to {out.resolve()} and {detail_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
