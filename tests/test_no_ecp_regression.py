#!/usr/bin/env python3
"""Golden-value regression test for the default all-electron FiRE path.

The effective-core-potential support added on top of FiRE is *optional*: with
``ecp=None`` -- the default -- the solver must behave exactly as it did before
that work started.  This module pins that behaviour with recorded values.

Two checks:

``zero_variance``
    The exact hydrogen 1s function is an eigenstate, so ``local_energy_fn``
    must return E_L = -0.5 Eh at every configuration with zero variance.  This
    catches any change to the Hamiltonian itself (kinetic term, electron-nucleus
    Coulomb, nuclear repulsion).

``trajectories``
    A short fixed-seed optimization (300 steps, 128 walkers, seed 1234) of H2,
    LiH and Be through the full stack -- pretraining, MCMC, SPRING -- must
    reproduce its recorded energy trace.  This catches any change that perturbs
    the default code path even slightly, including a stray extra draw from the
    PRNG stream.

Usage::

    python tests/test_no_ecp_regression.py --write   # record the golden file
    python tests/test_no_ecp_regression.py           # check against it

``--write`` is only legitimate before the ECP work begins, or after a change
that is *intended* to move the all-electron path; in that case say so in the
commit message.  The file is also collected by pytest where available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "golden_all_electron.json"

# Geometries in bohr, matching scripts/validate_fire.py.
SYSTEMS = [
    {"name": "H2", "symbols": ["H", "H"], "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]},
    {"name": "LiH", "symbols": ["Li", "H"], "coords": [[0.0, 0.0, 0.0], [0.0, 0.0, 3.015]]},
    {"name": "Be", "symbols": ["Be"], "coords": [[0.0, 0.0, 0.0]]},
]

# Tolerances.  The trace is a chaotic trajectory, so an intended-identical code
# path must reproduce it to round-off, not merely to chemical accuracy; the
# looser floor on the energy absorbs float64 reduction-order jitter.
ATOL_TRACE = 1e-8
ATOL_ENERGY = 1e-9
ATOL_ZERO_VARIANCE = 1e-12


def golden_config():
    """The fixed configuration the golden values were recorded with."""

    from angel_src.NQS.fire.config import MCMCConfig, OptConfig, replace_config, small_config

    config = small_config(seed=1234)
    return replace_config(
        config,
        mcmc=MCMCConfig(
            batch_size=128,
            n_global_moves=config.mcmc.n_global_moves,
            n_burn_in=config.mcmc.n_burn_in,
        ),
        opt=OptConfig(
            steps=300,
            learning_rate=config.opt.learning_rate,
            lr_decay_time=config.opt.lr_decay_time,
            n_eval_steps=100,
        ),
        double_precision=True,
    )


def zero_variance_record() -> dict:
    """E_L of the exact hydrogen 1s function; must be -0.5 Eh with no spread."""

    import jax
    import jax.numpy as jnp

    from angel_src.NQS.fire.hamiltonian import local_energy_fn
    from angel_src.NQS.fire.system import system_from_arrays

    jax.config.update("jax_enable_x64", True)
    system = system_from_arrays(
        ["H"], np.zeros((1, 3)), units="bohr", spin=1, name="H_atom"
    )

    def log_psi(_params, r):
        return -jnp.linalg.norm(r[0])

    energy = local_energy_fn(log_psi, system, backend="loop")
    positions = jax.random.normal(jax.random.PRNGKey(0), (512, 1, 3))
    values = jax.vmap(energy, in_axes=(None, 0))(None, positions)
    return {
        "mean_hartree": float(np.mean(values)),
        "std_hartree": float(np.std(values)),
        "max_abs_deviation": float(np.max(np.abs(np.asarray(values) + 0.5))),
    }


def trajectory_record(spec: dict) -> dict:
    """Run one short fixed-seed optimization and return its energy trace."""

    from angel_src.NQS.fire.solver import run_fire_vmc
    from angel_src.NQS.fire.system import system_from_arrays

    system = system_from_arrays(
        spec["symbols"], np.array(spec["coords"]), units="bohr", name=spec["name"]
    )
    started = time.time()
    result = run_fire_vmc(system, golden_config(), pretrain=True)
    trace = list(np.asarray(result.history["energy_hartree"], dtype=float))
    return {
        "energy_hartree": float(result.energy_hartree),
        "energy_error_hartree": float(result.energy_error_hartree),
        "variance_hartree2": float(result.energy_variance_hartree2),
        "hf_energy_hartree": (
            None if result.hf_energy_hartree is None else float(result.hf_energy_hartree)
        ),
        "n_electrons": int(system.n_electrons),
        "charges": [float(z) for z in system.charges],
        # Every 10th step keeps the file small while still pinning the path.
        "trace_every_10": [float(v) for v in trace[::10]],
        "seconds": round(time.time() - started, 1),
    }


def build_record() -> dict:
    record = {"zero_variance": zero_variance_record(), "systems": {}}
    for spec in SYSTEMS:
        record["systems"][spec["name"]] = trajectory_record(spec)
    return record


def _compare(current: dict, golden: dict) -> list[str]:
    failures: list[str] = []

    zv = current["zero_variance"]
    if abs(zv["mean_hartree"] + 0.5) > ATOL_ZERO_VARIANCE:
        failures.append(f"hydrogen E_L mean {zv['mean_hartree']!r} != -0.5")
    if zv["std_hartree"] > ATOL_ZERO_VARIANCE:
        failures.append(f"hydrogen E_L std {zv['std_hartree']!r} is not zero")

    for name, want in golden["systems"].items():
        got = current["systems"].get(name)
        if got is None:
            failures.append(f"{name}: missing from current record")
            continue
        if got["n_electrons"] != want["n_electrons"]:
            failures.append(
                f"{name}: n_electrons {got['n_electrons']} != {want['n_electrons']} "
                "(the default path must stay all-electron)"
            )
        if not np.allclose(got["charges"], want["charges"], atol=0.0):
            failures.append(
                f"{name}: charges {got['charges']} != {want['charges']} "
                "(the default path must use full nuclear charges)"
            )
        de = abs(got["energy_hartree"] - want["energy_hartree"])
        if de > ATOL_ENERGY:
            failures.append(
                f"{name}: energy moved by {de:.3e} Eh "
                f"({got['energy_hartree']:.12f} vs {want['energy_hartree']:.12f})"
            )
        a = np.asarray(got["trace_every_10"])
        b = np.asarray(want["trace_every_10"])
        if a.shape != b.shape:
            failures.append(f"{name}: trace length {a.shape} != {b.shape}")
        else:
            dmax = float(np.max(np.abs(a - b)))
            if dmax > ATOL_TRACE:
                failures.append(
                    f"{name}: energy trace deviates by up to {dmax:.3e} Eh "
                    f"(first at step {int(np.argmax(np.abs(a - b))) * 10})"
                )
    return failures


def check() -> list[str]:
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"{GOLDEN_PATH} not found; record it first with "
            "`python tests/test_no_ecp_regression.py --write`."
        )
    golden = json.loads(GOLDEN_PATH.read_text())
    return _compare(build_record(), golden)


def test_all_electron_path_unchanged() -> None:
    """pytest entry point."""

    failures = check()
    assert not failures, "\n".join(failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="record the golden file instead of checking against it",
    )
    args = parser.parse_args()

    if args.write:
        record = build_record()
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(record, indent=2) + "\n")
        zv = record["zero_variance"]
        print(f"hydrogen E_L = {zv['mean_hartree']:.12f} +- {zv['std_hartree']:.2e}")
        for name, rec in record["systems"].items():
            print(
                f"{name:>4}  E = {rec['energy_hartree']:.9f} Eh  "
                f"n_el = {rec['n_electrons']}  ({rec['seconds']} s)"
            )
        print(f"wrote {GOLDEN_PATH.relative_to(REPO_ROOT)}")
        return 0

    failures = check()
    if failures:
        print("ALL-ELECTRON REGRESSION FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("all-electron path unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
