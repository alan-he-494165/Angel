#!/usr/bin/env python3
"""Check the optional ECP metadata on :class:`MolecularSystem`.

Every assertion here is against PySCF built with the same effective core
potential, so the electron counts and reduced charges FiRE works with are the
ones the reference calculations will use.  The last test pins the default:
without ``ecp=`` nothing changes.

Usage::

    python tests/test_ecp_system.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from angel_src.NQS.fire.system import (  # noqa: E402
    atomic_number,
    ecp_from_name,
    system_from_arrays,
)

CASES = [
    {"name": "Li", "symbols": ["Li"], "coords": [[0.0, 0.0, 0.0]], "spin": 1},
    {"name": "Be", "symbols": ["Be"], "coords": [[0.0, 0.0, 0.0]], "spin": 0},
    {"name": "C", "symbols": ["C"], "coords": [[0.0, 0.0, 0.0]], "spin": 2},
    {
        "name": "LiH",
        "symbols": ["Li", "H"],
        "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]],
        "spin": 0,
    },
]


def _pyscf_reference(symbols, coords, spin, ecp):
    from pyscf import gto

    mol = gto.M(
        atom=[(s, tuple(float(x) for x in c)) for s, c in zip(symbols, coords)],
        unit="bohr",
        basis="sto-3g" if ecp is None else "ccecp-cc-pvdz",
        ecp=None if ecp is None else ecp,
        spin=spin,
        verbose=0,
    )
    return {
        "n_electrons": int(mol.nelectron),
        "charges": np.asarray(mol.atom_charges(), dtype=float),
        "nuclear_repulsion": float(mol.energy_nuc()),
    }


def test_ecp_matches_pyscf() -> None:
    for case in CASES:
        system = system_from_arrays(
            case["symbols"],
            np.array(case["coords"]),
            units="bohr",
            spin=case["spin"],
            name=case["name"],
            ecp="ccecp",
        )
        want = _pyscf_reference(case["symbols"], case["coords"], case["spin"], "ccecp")
        assert system.n_electrons == want["n_electrons"], (
            f"{case['name']}: {system.n_electrons} electrons vs PySCF "
            f"{want['n_electrons']}"
        )
        assert np.allclose(system.charges, want["charges"]), (
            f"{case['name']}: charges {system.charges} vs PySCF {want['charges']}"
        )
        assert np.isclose(system.nuclear_repulsion, want["nuclear_repulsion"]), (
            f"{case['name']}: E_nuc {system.nuclear_repulsion} vs PySCF "
            f"{want['nuclear_repulsion']}"
        )
        assert system.has_ecp and system.ecp_name == "ccecp"
        assert system.charge == 0 and system.spin == case["spin"]


def test_core_electrons_removed() -> None:
    """ccECP removes the 1s pair of Li/Be/C and nothing from H."""

    expected_core = {"Li": 2, "Be": 2, "C": 2, "H": 0}
    for symbol, n_core in expected_core.items():
        entries = ecp_from_name("ccecp", [symbol])
        assert entries is not None and entries[0] is not None, symbol
        entry = entries[0]
        assert entry.n_core == n_core, f"{symbol}: n_core {entry.n_core} != {n_core}"
        assert entry.z_eff == atomic_number(symbol) - n_core
        # A local channel is always present; hydrogen has no projector.
        assert entry.local_channel is not None, symbol
        if symbol == "H":
            assert entry.nonlocal_channels == ()
            assert entry.max_angular_momentum == -1
        else:
            assert entry.max_angular_momentum >= 0, symbol


def test_channel_radial_form() -> None:
    """The Be local channel carries the +Z_core/r term that softens the cusp."""

    entry = ecp_from_name("ccecp", ["Be"])[0]
    local = entry.local_channel
    inverse_r = local.r_powers == -1.0
    assert inverse_r.sum() == 1
    assert np.isclose(local.coefficients[inverse_r][0], float(entry.n_core))
    # r -> 0 limit: the tabulated potential diverges as +n_core/r, cancelling
    # part of the -Z_eff/r Coulomb attraction rather than adding to it.
    assert local.evaluate(np.array([1e-3]))[0] > 0.0
    # and it decays at chemical distances
    assert abs(local.evaluate(np.array([6.0]))[0]) < 1e-8


def test_default_is_all_electron() -> None:
    """No ecp= argument: full charges, full electron count, no metadata."""

    system = system_from_arrays(
        ["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]]), units="bohr"
    )
    assert system.ecp is None
    assert not system.has_ecp
    assert system.ecp_name is None
    assert system.n_core_electrons == 0
    assert system.n_electrons == 4
    assert np.allclose(system.charges, [3.0, 1.0])
    assert system.provenance()["core_treatment"] == "all-electron"
    assert system.pyscf_ecp_spec() == {}


def main() -> int:
    tests = [
        test_ecp_matches_pyscf,
        test_core_electrons_removed,
        test_channel_radial_form,
        test_default_is_all_electron,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
