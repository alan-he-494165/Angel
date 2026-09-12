#!/usr/bin/env python3
"""Where does Var(E_L) come from, and does the ansatz carry a nuclear cusp?

Two measurements, both at *fixed* parameters after a short optimization:

1. ``Var(E_L)`` split into kinetic / Coulomb / ECP-local / ECP-nonlocal, with
   the mean local energy binned by the closest electron-nucleus distance.  An
   all-electron wavefunction satisfying Kato's cusp condition keeps ``E_L``
   flat into ``r -> 0``; a cusped wavefunction under a pseudopotential does
   not, because the ECP local channel's ``+z_eff/r`` term has removed the
   Coulomb singularity that the cusp was cancelling.

2. The cusp itself, ``d log|Psi| / dr`` as an electron is walked radially into
   a nucleus.  Kato requires ``-z_eff`` all-electron; a pseudopotential
   requires ``0`` (the pseudo-orbital is smooth at the nucleus).

Usage::

    python scripts/diagnose_ecp_variance.py                 # H2, both Hamiltonians
    python scripts/diagnose_ecp_variance.py --steps 2000 --systems H2 LiH
    python scripts/diagnose_ecp_variance.py --cusp-only     # seconds, no VMC
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "out" / "ecp_variance"

GEOMETRIES: dict[str, tuple[list[str], np.ndarray, int, int]] = {
    "H2": (["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), 0, 0),
    "LiH": (["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]]), 0, 0),
    "Be": (["Be"], np.array([[0.0, 0.0, 0.0]]), 0, 0),
}


@dataclass(frozen=True)
class VarianceRow:
    """One system under one Hamiltonian."""

    system: str
    core_treatment: str
    n_samples: int
    energy_hartree: float
    var_total: float
    var_kinetic: float
    var_coulomb: float
    var_ecp_local: float
    var_ecp_nonlocal: float
    var_sum_diagonal: float
    mean_el_within_0p05: float
    cusp_measured: float
    cusp_expected: float
    log_ratio_max: float
    frac_log_ratio_clamped: float


def build(name: str, ecp: str | None, steps: int, pretrain_steps: int, batch: int):
    from angel_src.NQS.fire.config import small_config
    from angel_src.NQS.fire.system import system_from_arrays

    symbols, coords, charge, spin = GEOMETRIES[name]
    system = system_from_arrays(symbols, coords, charge=charge, spin=spin,
                                units="bohr", name=name, ecp=ecp)
    base = small_config()
    config = replace(
        base,
        mcmc=replace(base.mcmc, batch_size=batch),
        opt=replace(base.opt, steps=steps, learning_rate=0.05,
                    lr_decay_time=steps / 4.0, n_eval_steps=max(steps // 10, 10)),
        pretrain=replace(base.pretrain, steps=pretrain_steps),
        ecp=ecp,
        log_every=max(steps // 4, 1),
    )
    return system, config


def measure_cusp(sign_log_psi, params, system, nucleus: int = 0,
                 radii: tuple[float, ...] = (2e-3, 5e-3, 1e-2, 2e-2)) -> float:
    """Slope of log|Psi| along a radius into ``nucleus``, extrapolated to r=0.

    Averaged over random directions and random positions of the spectators, so
    a single number is comparable between runs.
    """
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(0)
    centre = jnp.asarray(system.nuclei)[nucleus]
    slopes = []
    for trial in range(16):
        key, k_pos, k_dir = jax.random.split(key, 3)
        r0 = jax.random.normal(k_pos, (system.n_electrons, 3)) * 1.2 + centre
        direction = jax.random.normal(k_dir, (3,))
        direction = direction / jnp.linalg.norm(direction)
        logs = []
        for radius in radii:
            r = r0.at[0].set(centre + radius * direction)
            logs.append(float(sign_log_psi(params, r)[1]))
        slope = np.polyfit(np.asarray(radii), np.asarray(logs), 1)[0]
        slopes.append(slope)
    return float(np.median(slopes))


def run_one(name: str, ecp: str | None, args: argparse.Namespace) -> VarianceRow:
    import jax
    import jax.numpy as jnp

    from angel_src.NQS.fire.ansatz import make_log_psi
    from angel_src.NQS.fire.ecp import ecp_potential_fn
    from angel_src.NQS.fire.hamiltonian import kinetic_energy_fn, potential_energy
    from angel_src.NQS.fire.mcmc import default_n_steps, make_sampler
    from angel_src.NQS.fire.pretrain import pretrain_to_hartree_fock
    from angel_src.NQS.fire.train import optimize

    label = ecp or "all_electron"
    system, config = build(name, ecp, args.steps, args.pretrain_steps, args.batch_size)
    _, log_psi, sign_log_psi = make_log_psi(system, config.ansatz)
    key = jax.random.PRNGKey(args.seed)
    key, pre_key, train_key, samp_key = jax.random.split(key, 4)

    params = pretrain_to_hartree_fock(system, config, key=pre_key)["params"]
    z_eff = float(system.charges[0])
    if args.cusp_only:
        cusp = measure_cusp(sign_log_psi, params, system)
        return VarianceRow(name, label, 0, float("nan"), *([float("nan")] * 7),
                           cusp, 0.0 if system.has_ecp else -z_eff, 0.0, 0.0)

    result = optimize(system, config, params=params, key=train_key)
    params = result["params"]
    cusp = measure_cusp(sign_log_psi, params, system)

    nuclei, charges = jnp.asarray(system.nuclei), jnp.asarray(system.charges)
    v_kin = jax.jit(jax.vmap(kinetic_energy_fn(log_psi, "loop"), in_axes=(None, 0)))
    v_coul = jax.jit(jax.vmap(
        lambda r: potential_energy(r, nuclei, charges, system.nuclear_repulsion)))
    if system.has_ecp:
        kw = dict(n_quadrature=config.ecp_quadrature, cutoff=config.ecp_cutoff,
                  max_log_ratio=config.ecp_max_log_ratio)
        stripped = replace(system, ecp=tuple(
            None if e is None else replace(
                e, channels=tuple(c for c in e.channels if c.is_local))
            for e in system.ecp))
        v_full = jax.jit(jax.vmap(ecp_potential_fn(sign_log_psi, system, **kw),
                                  in_axes=(None, 0, 0)))
        v_loc = jax.jit(jax.vmap(ecp_potential_fn(sign_log_psi, stripped, **kw),
                                 in_axes=(None, 0, 0)))

    mcmc = replace(config.mcmc, batch_size=args.eval_batch)
    init_sampler, sample_block = make_sampler(log_psi, system, mcmc)
    n_steps = default_n_steps(system, mcmc)
    state = init_sampler(samp_key, args.eval_batch)
    step_fn = jax.jit(lambda s: sample_block(params, s, n_steps))
    for _ in range(args.burn_in_blocks):
        state = step_fn(state)

    parts: dict[str, list[np.ndarray]] = {k: [] for k in
                                          ("kinetic", "coulomb", "ecp_local",
                                           "ecp_nonlocal", "total", "min_dist")}
    for block in range(args.eval_blocks):
        state = step_fn(state)
        r = state.positions
        kin, coul = np.asarray(v_kin(params, r)), np.asarray(v_coul(r))
        dist = np.asarray(jnp.linalg.norm(
            r[:, :, None, :] - nuclei[None, None, :, :], axis=-1))
        if system.has_ecp:
            key, k_block = jax.random.split(key)
            keys = jax.random.split(k_block, r.shape[0])
            full = np.asarray(v_full(params, r, keys))
            loc = np.asarray(v_loc(params, r, keys))
        else:
            full = loc = np.zeros_like(kin)
        parts["kinetic"].append(kin)
        parts["coulomb"].append(coul)
        parts["ecp_local"].append(loc)
        parts["ecp_nonlocal"].append(full - loc)
        parts["total"].append(kin + coul + full)
        parts["min_dist"].append(dist.min(axis=(1, 2)))
        if (block + 1) % 10 == 0:
            print(f"  [{name}/{label}] block {block + 1}/{args.eval_blocks}", flush=True)

    data = {k: np.concatenate(v) for k, v in parts.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"samples_{name}_{label}.npz", **data)
    close = data["min_dist"] < 0.05
    terms = ("kinetic", "coulomb", "ecp_local", "ecp_nonlocal")
    return VarianceRow(
        system=name, core_treatment=label, n_samples=int(data["total"].size),
        energy_hartree=float(data["total"].mean()),
        var_total=float(data["total"].var(ddof=1)),
        **{f"var_{t}": float(data[t].var(ddof=1)) for t in terms},
        var_sum_diagonal=float(sum(data[t].var(ddof=1) for t in terms)),
        mean_el_within_0p05=float(data["total"][close].mean()) if close.any() else float("nan"),
        cusp_measured=cusp, cusp_expected=0.0 if system.has_ecp else -z_eff,
        log_ratio_max=0.0, frac_log_ratio_clamped=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["H2"], choices=list(GEOMETRIES))
    parser.add_argument("--ecp", nargs="+", default=["ccecp", "none"],
                        help="Hamiltonians to compare; 'none' is all-electron.")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--pretrain-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--eval-blocks", type=int, default=40)
    parser.add_argument("--burn-in-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cusp-only", action="store_true",
                        help="Measure the cusp on the pretrained ansatz only (seconds).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = [run_one(name, None if choice == "none" else choice, args)
            for name in args.systems for choice in args.ecp]

    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(VarianceRow.__dataclass_fields__)
    with open(args.out / "ecp_variance.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    print(json.dumps([row.__dict__ for row in rows], indent=2))
    print(f"wrote {args.out / 'ecp_variance.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
