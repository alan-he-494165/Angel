#!/usr/bin/env python3
"""Check whether a converged FiRE run really sits above the exact energy.

A variational Monte Carlo energy cannot legitimately fall below the exact
ground state, so when it does the cause is one of

1. a wrong Hamiltonian (the reported quantity is not <H>),
2. a biased estimator (clipped local energies used for the reported mean),
3. Monte Carlo error, either the plain statistical error or a bias from
   sampling |Psi|^2 imperfectly - with an all-electron core the local energy
   has heavy tails and a finite sample under-weights them,
4. a reference that is itself too high.

(1) and (2) are ruled out by inspection elsewhere (``hamiltonian.py`` passes
the hydrogen zero-variance test; ``train.py`` reports the mean of *unclipped*
local energies).  This script quantifies (3): it retrains a system, then
evaluates the final wavefunction at **fixed parameters** with a long,
well-equilibrated chain, reporting a blocked error bar and the shape of the
local-energy distribution.

Writes ``<out>/<system>_diagnosis.json``, ``<out>/variational_diagnosis.csv``
and the raw eval-phase local energies as ``<out>/<system>_local_energies.npy``.

Example
-------
    python scripts/diagnose_variational.py --system LiH --steps 4000 \\
        --learning-rate 0.05 --eval-batch 1024 --eval-blocks 400 --out out/diag
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

# Same geometries and literature values as scripts/validate_fire.py.
SYSTEMS = {
    "H2": dict(symbols=["H", "H"], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
               charge=0, spin=0, literature=-1.174476,
               source="Kolos-Wolniewicz, R = 1.4 a0"),
    "LiH": dict(symbols=["Li", "H"], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]],
                charge=0, spin=0, literature=-8.070548,
                source="Cencek-Rychlewski, R = 3.015 a0"),
    "Be": dict(symbols=["Be"], coords=[[0.0, 0.0, 0.0]],
               charge=0, spin=0, literature=-14.667356,
               source="Chakravorty et al. 1993"),
}


def diagnose(name: str, args) -> dict:
    import jax

    from angel_src.NQS.fire.ansatz import init_params, make_log_psi
    from angel_src.NQS.fire.config import small_config
    from angel_src.NQS.fire.hamiltonian import local_energy_fn, resolve_backend
    from angel_src.NQS.fire.system import system_from_arrays
    from angel_src.NQS.fire.mcmc import default_n_steps, make_sampler
    from angel_src.NQS.fire.train import blocking_analysis, evaluate_frozen, optimize

    spec = SYSTEMS[name]
    system = system_from_arrays(
        spec["symbols"], np.asarray(spec["coords"], dtype=float),
        charge=spec["charge"], spin=spec["spin"], units="bohr", name=name,
    )
    base = small_config()
    config = replace(
        base,
        ansatz=replace(base.ansatz, n_determinants=args.determinants),
        mcmc=replace(base.mcmc, batch_size=args.batch_size),
        opt=replace(
            base.opt, steps=args.steps,
            n_eval_steps=max(args.steps // 10, 10),
            learning_rate=args.learning_rate,
            lr_decay_time=max(args.steps / 4.0, 1.0),
        ),
        pretrain=replace(base.pretrain, steps=args.pretrain_steps),
        seed=args.seed,
        log_every=max(args.steps // 10, 1),
    )
    if config.double_precision:
        jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(config.seed)
    key, pretrain_key, train_key, eval_key = jax.random.split(key, 4)

    params = None
    if args.pretrain_steps > 0:
        from angel_src.NQS.fire.pretrain import pretrain_to_hartree_fock
        params = pretrain_to_hartree_fock(system, config, key=pretrain_key)["params"]

    print(f"[{name}] training {args.steps} steps at lr_0 = {args.learning_rate}", flush=True)
    result = optimize(
        system, config, params=params, key=train_key,
        callback=lambda step, record: print(
            f"[{name}] {step:6d} E={record['energy']:.6f}", flush=True),
    )
    params = result["params"]
    train_energy = float(result["energy_hartree"])
    train_error = float(result["energy_error_hartree"])

    _, log_psi, _ = make_log_psi(system, config.ansatz)
    backend = resolve_backend(config.laplacian_backend)
    local_energy = local_energy_fn(
        log_psi, system, backend=backend, sign_log_psi=make_log_psi(
            system, config.ansatz
        )[2], ecp_quadrature=config.ecp_quadrature,
        ecp_cutoff=config.ecp_cutoff, ecp_max_log_ratio=config.ecp_max_log_ratio,
    )
    _, sample_block = make_sampler(log_psi, system, config.mcmc)
    if system.has_ecp:
        keyed = jax.vmap(local_energy, in_axes=(None, 0, 0))

        def batched_local_energy(p, positions, energy_key):
            keys = jax.random.split(energy_key, positions.shape[0])
            return keyed(p, positions, keys)
    else:
        unkeyed = jax.vmap(local_energy, in_axes=(None, 0))

        def batched_local_energy(p, positions, energy_key):
            del energy_key
            return unkeyed(p, positions)

    # Preserve --eval-batch without returning to the unequilibrated Gaussian
    # cloud: resize the terminal walker population, then burn in the duplicated
    # or selected equilibrated chains before recording.
    state = result["sampler_state"]
    positions = state.positions
    if args.eval_batch != positions.shape[0]:
        repeats = (args.eval_batch + positions.shape[0] - 1) // positions.shape[0]
        positions = jax.numpy.tile(positions, (repeats, 1, 1))[:args.eval_batch]
        state = state._replace(positions=positions)

    print(f"[{name}] fixed-parameter evaluation: {args.eval_blocks} blocks x "
          f"{args.eval_batch} walkers after {args.burn_in_blocks} burn-in blocks", flush=True)
    evaluated = evaluate_frozen(
        params, state, eval_key,
        sample_block=sample_block,
        batched_local_energy=batched_local_energy,
        n_steps=default_n_steps(system, config.mcmc),
        n_blocks=args.eval_blocks,
        block_steps=1,
        burn_in_blocks=args.burn_in_blocks,
        callback=lambda block, running_mean: print(
            f"    eval block {block}/{args.eval_blocks}  running mean "
            f"{running_mean:.6f}", flush=True
        ) if block % max(args.eval_blocks // 10, 1) == 0 else None,
    )
    block_means, samples = evaluated["block_means"], evaluated["samples"]

    mean = float(block_means.mean())
    analysis = blocking_analysis(block_means)
    sem, block = analysis.error, analysis.block_size
    median = float(np.median(samples))
    mad = float(np.mean(np.abs(samples - median)))
    outside = samples[np.abs(samples - median) > config.opt.clip_width * mad]
    literature = float(spec["literature"])

    record = {
        "system": name,
        "learning_rate": args.learning_rate,
        "train_window_energy_hartree": train_energy,
        "train_window_error_hartree": train_error,
        "eval_energy_hartree": mean,
        "eval_blocked_error_hartree": sem,
        "eval_blocking_block_size": block,
        "eval_n_samples": int(samples.size),
        "eval_variance_hartree2": float(samples.var()),
        "eval_skewness": float(((samples - samples.mean()) ** 3).mean() / samples.std() ** 3),
        "local_energy_min": float(samples.min()),
        "local_energy_max": float(samples.max()),
        "local_energy_q001": float(np.quantile(samples, 0.001)),
        "local_energy_q999": float(np.quantile(samples, 0.999)),
        "fraction_beyond_5_mad": float(outside.size / samples.size),
        "mean_shift_if_tail_dropped_mhartree": float(
            (samples[np.abs(samples - median) <= config.opt.clip_width * mad].mean()
             - samples.mean()) * 1e3
        ),
        "literature_hartree": literature,
        "literature_source": spec["source"],
        "eval_minus_literature_mhartree": (mean - literature) * 1e3,
        "train_minus_literature_mhartree": (train_energy - literature) * 1e3,
        "sigmas_below_literature": (literature - mean) / sem if sem > 0 else float("nan"),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}_local_energies.npy", samples.astype(np.float32))
    (out / f"{name}_diagnosis.json").write_text(json.dumps(record, indent=2))
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--system", type=str, default="LiH",
                        help="Comma-separated subset of " + ", ".join(SYSTEMS))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--determinants", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--pretrain-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-batch", type=int, default=1024)
    parser.add_argument("--eval-blocks", type=int, default=400)
    parser.add_argument("--burn-in-blocks", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("out/diag"))
    return parser.parse_args()


def main() -> int:
    import csv

    args = parse_args()
    records = [diagnose(name, args) for name in args.system.split(",")]
    path = Path(args.out) / "variational_diagnosis.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(
            f"{record['system']}: train-window {record['train_window_energy_hartree']:.6f}, "
            f"fixed-parameter eval {record['eval_energy_hartree']:.6f} "
            f"+/- {record['eval_blocked_error_hartree'] * 1e3:.3f} mEh "
            f"(block {record['eval_blocking_block_size']}), "
            f"literature {record['literature_hartree']:.6f}, "
            f"eval - lit {record['eval_minus_literature_mhartree']:+.3f} mEh "
            f"({record['sigmas_below_literature']:+.2f} sigma below), "
            f"skew {record['eval_skewness']:.2f}, "
            f"tail fraction {record['fraction_beyond_5_mad']:.2e}"
        )
    print(f"wrote {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
