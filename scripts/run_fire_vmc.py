#!/usr/bin/env python3
"""Run the FiRE real-space variational NQS solver on one XYZ geometry.

Label-free: the driver minimizes the sampled expectation value of the
all-electron Hamiltonian and never reads a reference energy.  Any comparison
value passed with ``--reference-hartree`` is only printed alongside the
result, it does not enter the optimization.

Examples
--------
Smoke test on a small molecule with the reduced preset::

    python scripts/run_fire_vmc.py data/h2.xyz --preset small --steps 300

Production-scale settings for an aromatic dimer frame::

    python scripts/run_fire_vmc.py data/c6h6_dimer_scan.xyz --frame 4 \\
        --steps 50000 --batch-size 2048 --out runs/dimer_f4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("geometry", type=Path, help="XYZ file (angstrom).")
    parser.add_argument("--frame", type=int, default=0, help="Frame index in a multi-frame XYZ.")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0, help="2S, i.e. n_up - n_down.")
    parser.add_argument(
        "--preset",
        choices=("paper", "small"),
        default="paper",
        help="'paper' uses the Tab. S1 defaults, 'small' the reduced CPU preset.",
    )
    parser.add_argument("--steps", type=int, default=3000, help="Optimization steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Walkers per step.")
    parser.add_argument("--determinants", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.05, help="SPRING base learning rate lr_0.")
    parser.add_argument("--lr-decay-time", type=float, default=0.99)
    parser.add_argument("--cutoff", type=float, default=None, help="Embedding cutoff in bohr.")
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--pretrain-basis", type=str, default=None)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--laplacian",
        choices=("auto", "folx", "loop"),
        default=None,
        help="Laplacian backend; 'auto' prefers folx when installed.",
    )
    parser.add_argument("--single-precision", action="store_true", help="Disable float64.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None, help="Directory for summary.json/history.csv.")
    parser.add_argument(
        "--reference-hartree",
        type=float,
        default=None,
        help="Optional comparison energy, printed only (never used in training).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    from dataclasses import replace

    from angel_src.NQS.fire.config import FiREConfig, small_config

    config = small_config() if args.preset == "small" else FiREConfig()

    ansatz = config.ansatz
    if args.determinants is not None:
        ansatz = replace(ansatz, n_determinants=args.determinants)
    if args.cutoff is not None:
        ansatz = replace(ansatz, cutoff=args.cutoff)

    mcmc = config.mcmc
    if args.batch_size is not None:
        mcmc = replace(mcmc, batch_size=args.batch_size)

    opt = config.opt
    if args.steps is not None:
        opt = replace(opt, steps=args.steps, n_eval_steps=min(opt.n_eval_steps, args.steps))
    if args.learning_rate is not None:
        opt = replace(opt, learning_rate=args.learning_rate)
    if args.lr_decay_time is not None:
        opt = replace(opt, lr_decay_time=args.lr_decay_time)

    pretrain = config.pretrain
    if args.pretrain_steps is not None:
        pretrain = replace(pretrain, steps=args.pretrain_steps)
    if args.pretrain_basis is not None:
        pretrain = replace(pretrain, basis=args.pretrain_basis)

    updates = {"ansatz": ansatz, "mcmc": mcmc, "opt": opt, "pretrain": pretrain,
               "log_every": max(args.log_every, 1)}
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.laplacian is not None:
        updates["laplacian_backend"] = args.laplacian
    if args.single_precision:
        updates["double_precision"] = False
    return replace(config, **updates)


def main() -> int:
    args = parse_args()
    config = build_config(args)

    from angel_src.NQS.fire.solver import solve_geometry_fire

    start = time.perf_counter()

    def report(step: int, record: dict) -> None:
        if "loss" in record:
            print(f"  pretrain {step:6d}  loss {record['loss']:.4e}", flush=True)
        else:
            print(
                f"  step {step:6d}  E = {record['energy']:.6f} Eh"
                f"  var {record['variance']:.4f}"
                f"  acc {record['acceptance_local']:.2f}/{record['acceptance_global']:.2f}",
                flush=True,
            )

    result = solve_geometry_fire(
        args.geometry,
        frame=args.frame,
        charge=args.charge,
        spin=args.spin,
        config=config,
        pretrain=not args.no_pretrain,
        output_dir=args.out,
        callback=report,
    )
    elapsed = time.perf_counter() - start

    summary = result.to_dict()
    summary["wall_time_seconds"] = round(elapsed, 1)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.reference_hartree is not None:
        difference = (result.energy_hartree - args.reference_hartree) * 1000.0
        print(f"reference comparison: {difference:+.2f} mEh above/below the supplied value")
    if args.out is not None:
        print(f"written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
