#!/usr/bin/env python3
"""Validate the FiRE ECP operator against PySCF.

A single Slater determinant has the same energy in VMC as it does in
Hartree-Fock, so sampling |Psi_HF|^2 and averaging the local energy must
reproduce ``E_RHF`` for the same molecule, basis and effective core
potential.  Any error in the ECP term -- a wrong radial power, a missing
(2l+1), a mis-normalised quadrature -- shows up as a shift far larger than
the Monte Carlo error bar.  The ECP expectation value alone is also compared
against ``Tr(D V_ECP)`` from PySCF's ECP integrals, which isolates the new
operator from the rest of the Hamiltonian.

The trial wavefunction is the RHF determinant in an *uncontracted* ccECP
basis so that the orbitals are single Gaussians and can be evaluated in JAX.
The AO normalisation and ordering are calibrated numerically against
``mol.eval_gto`` rather than assumed, and the calibration is asserted to be
point-independent.

Usage::

    PYTHONPATH=. python3 scripts/validate_ecp_operator.py --out out/ecp_operator_check.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from angel_src.NQS.fire.ecp import ecp_potential_fn  # noqa: E402
from angel_src.NQS.fire.hamiltonian import local_energy_fn  # noqa: E402
from angel_src.NQS.fire.system import system_from_arrays  # noqa: E402

# Uncontracted ccECP-derived primitives.  Small on purpose: the reference is
# RHF in this same basis, so the basis only has to be rich enough to give the
# determinant real angular structure.
BASIS = {
    "H": [[0, [0.90, 1.0]], [0, [0.20, 1.0]], [1, [0.70, 1.0]]],
    "Li": [[0, [0.65, 1.0]], [0, [0.12, 1.0]], [1, [0.30, 1.0]]],
    "Be": [[0, [1.10, 1.0]], [0, [0.25, 1.0]], [1, [0.55, 1.0]]],
}

SYSTEMS = [
    {"name": "H2", "symbols": ["H", "H"], "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]},
    {"name": "Be", "symbols": ["Be"], "coords": [[0.0, 0.0, 0.0]]},
    {"name": "LiH", "symbols": ["Li", "H"], "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]]},
]


def build_scf(symbols, coords, ecp_name):
    """RHF in the uncontracted basis with the ECP; returns mol and mean field."""

    from pyscf import gto, scf

    mol = gto.M(
        atom=[(s, tuple(float(x) for x in c)) for s, c in zip(symbols, coords)],
        unit="bohr",
        basis={s: BASIS[s] for s in set(symbols)},
        ecp={s: ecp_name for s in set(symbols)},
        verbose=0,
    )
    mean_field = scf.RHF(mol).run()
    return mol, mean_field


def raw_ao_spec(mol):
    """Per-AO (centre index, angular momentum, component, exponent).

    Only uncontracted shells are supported, which is what ``BASIS`` provides.
    """

    spec = []
    for shell in range(mol.nbas):
        n_prim = mol.bas_nprim(shell)
        n_ctr = mol.bas_nctr(shell)
        if n_prim != 1 or n_ctr != 1:
            raise ValueError("validate_ecp_operator requires an uncontracted basis.")
        angular = int(mol.bas_angular(shell))
        if angular > 1:
            raise ValueError("only s and p shells are implemented here.")
        exponent = float(mol.bas_exp(shell)[0])
        centre = int(mol.bas_atom(shell))
        for component in range(2 * angular + 1):
            spec.append((centre, angular, component, exponent))
    return spec


def make_raw_ao(mol):
    """JAX evaluation of the unnormalised AOs, in PySCF's ordering."""

    spec = raw_ao_spec(mol)
    centres = jnp.asarray(np.array([mol.atom_coord(s[0]) for s in spec]))
    exponents = jnp.asarray(np.array([s[3] for s in spec]))
    angular = np.array([s[1] for s in spec])
    component = np.array([s[2] for s in spec])
    # PySCF's spherical p ordering is (x, y, z); the calibration below fails
    # loudly if that is ever untrue for this build.
    axis = jnp.asarray(np.where(angular == 1, component, 0))
    is_p = jnp.asarray(angular == 1)

    def raw_ao(position):
        displacement = position[None, :] - centres
        radial = jnp.exp(-exponents * jnp.sum(displacement**2, axis=-1))
        angular_factor = jnp.where(
            is_p, jnp.take_along_axis(displacement, axis[:, None], axis=1)[:, 0], 1.0
        )
        return radial * angular_factor

    return raw_ao


def calibrate(mol, raw_ao):
    """Normalisation N with AO_pyscf = N * AO_raw; asserted point-independent."""

    points = np.random.default_rng(0).normal(size=(6, 3)) * 1.3
    pyscf_values = mol.eval_gto("GTOval_sph", points)
    raw_values = np.array([np.asarray(raw_ao(jnp.asarray(p))) for p in points])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pyscf_values / raw_values
    scale = np.nanmedian(ratio, axis=0)
    spread = np.nanmax(np.abs(ratio - scale), axis=0) / np.abs(scale)
    if np.nanmax(spread) > 1e-10:
        raise AssertionError(
            "AO normalisation is not point-independent (max relative spread "
            f"{np.nanmax(spread):.2e}); the assumed ordering or radial form is wrong."
        )
    return scale


def make_trial(mol, mean_field):
    """Return ``(log_psi, sign_log_psi, n_up)`` for the RHF determinant."""

    raw_ao = make_raw_ao(mol)
    scale = calibrate(mol, raw_ao)
    n_occ = int(mean_field.mol.nelectron // 2)
    coefficients = jnp.asarray(
        np.asarray(mean_field.mo_coeff)[:, :n_occ] * scale[:, None]
    )

    def orbitals(r):
        return jax.vmap(raw_ao)(r) @ coefficients

    def sign_log_psi(_params, r):
        matrix = orbitals(r)
        sign_up, log_up = jnp.linalg.slogdet(matrix[:n_occ])
        sign_down, log_down = jnp.linalg.slogdet(matrix[n_occ:])
        return sign_up * sign_down, log_up + log_down

    def log_psi(params, r):
        return sign_log_psi(params, r)[1]

    return log_psi, sign_log_psi, n_occ


def metropolis(log_psi, key, positions, n_steps, step_size):
    """Simple all-electron Metropolis continuing from ``positions``."""

    batched = jax.vmap(log_psi, in_axes=(None, 0))

    @jax.jit
    def step(state, subkey):
        positions, logp = state
        move_key, accept_key = jax.random.split(subkey)
        proposal = positions + step_size * jax.random.normal(move_key, positions.shape)
        logp_new = batched(None, proposal)
        accept = jnp.log(jax.random.uniform(accept_key, (positions.shape[0],))) < 2.0 * (
            logp_new - logp
        )
        positions = jnp.where(accept[:, None, None], proposal, positions)
        logp = jnp.where(accept, logp_new, logp)
        return (positions, logp), jnp.mean(accept)

    state = (positions, batched(None, positions))
    rates = []
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        state, rate = step(state, subkey)
        rates.append(float(rate))
    return state[0], float(np.mean(rates))


def blocked_error(values: np.ndarray, n_blocks: int = 32) -> float:
    trimmed = values[: (len(values) // n_blocks) * n_blocks]
    blocks = trimmed.reshape(n_blocks, -1).mean(axis=1)
    return float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))


def run_case(spec, args) -> dict:
    mol, mean_field = build_scf(spec["symbols"], spec["coords"], args.ecp)
    system = system_from_arrays(
        spec["symbols"],
        np.array(spec["coords"]),
        units="bohr",
        name=spec["name"],
        ecp=args.ecp,
    )
    assert system.n_electrons == mol.nelectron, (system.n_electrons, mol.nelectron)

    log_psi, sign_log_psi, _ = make_trial(mol, mean_field)
    energy = local_energy_fn(
        log_psi,
        system,
        backend="auto",
        sign_log_psi=sign_log_psi,
        ecp_quadrature=args.quadrature,
    )
    v_ecp = ecp_potential_fn(sign_log_psi, system, n_quadrature=args.quadrature)
    batched_energy = jax.jit(jax.vmap(energy, in_axes=(None, 0, 0)))
    batched_v_ecp = jax.jit(jax.vmap(v_ecp, in_axes=(None, 0, 0)))

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (args.walkers, system.n_electrons, 3))
    key, subkey = jax.random.split(key)
    positions, _ = metropolis(log_psi, subkey, positions, args.burn_in, args.step_size)

    energies: list[np.ndarray] = []
    ecp_values: list[np.ndarray] = []
    acceptance = []
    for _ in range(args.blocks):
        key, subkey = jax.random.split(key)
        positions, rate = metropolis(
            log_psi, subkey, positions, args.decorrelate, args.step_size
        )
        acceptance.append(rate)
        key, energy_key = jax.random.split(key)
        keys = jax.random.split(energy_key, args.walkers)
        energies.append(np.asarray(batched_energy(None, positions, keys)))
        key, ecp_key = jax.random.split(key)
        keys = jax.random.split(ecp_key, args.walkers)
        ecp_values.append(np.asarray(batched_v_ecp(None, positions, keys)))

    local_energies = np.concatenate(energies)
    ecp_samples = np.concatenate(ecp_values)

    density = mean_field.make_rdm1()
    ecp_reference = float(np.einsum("ij,ji->", density, mol.intor("ECPscalar")))

    return {
        "system": spec["name"],
        "ecp": args.ecp,
        "n_electrons": int(system.n_electrons),
        "charges": "/".join(f"{z:g}" for z in system.charges),
        "e_rhf_hartree": float(mean_field.e_tot),
        "e_vmc_hartree": float(local_energies.mean()),
        "vmc_error_hartree": blocked_error(local_energies),
        "e_vmc_minus_rhf_mhartree": (float(local_energies.mean()) - float(mean_field.e_tot))
        * 1e3,
        "ecp_vmc_hartree": float(ecp_samples.mean()),
        "ecp_reference_hartree": ecp_reference,
        "ecp_error_hartree": blocked_error(ecp_samples),
        "ecp_diff_mhartree": (float(ecp_samples.mean()) - ecp_reference) * 1e3,
        "n_samples": int(local_energies.size),
        "quadrature": args.quadrature,
        "acceptance": round(float(np.mean(acceptance)), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/ecp_operator_check.csv")
    parser.add_argument("--ecp", default="ccecp")
    parser.add_argument("--quadrature", type=int, default=12)
    parser.add_argument("--walkers", type=int, default=2048)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--decorrelate", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--only", default=None, help="comma-separated system names")
    args = parser.parse_args()

    wanted = None if args.only is None else set(args.only.split(","))
    rows = []
    for spec in SYSTEMS:
        if wanted is not None and spec["name"] not in wanted:
            continue
        row = run_case(spec, args)
        rows.append(row)
        print(
            f"{row['system']:>4}  E_VMC = {row['e_vmc_hartree']:.6f} "
            f"+- {row['vmc_error_hartree']:.6f}  E_RHF = {row['e_rhf_hartree']:.6f}  "
            f"diff = {row['e_vmc_minus_rhf_mhartree']:+.3f} mEh  |  "
            f"V_ECP {row['ecp_vmc_hartree']:.6f} vs {row['ecp_reference_hartree']:.6f} "
            f"({row['ecp_diff_mhartree']:+.3f} mEh)"
        )

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
