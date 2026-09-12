"""FiRE: finite-range-embedding real-space variational NQS solver.

Reimplementation of the wave function, optimizer and sampler of

    M. Scherbela, N. Gao, P. Grohs, S. Guennemann,
    "Accurate Ab-initio Neural-network Solutions to Large-Scale Electronic
    Structure Problems", arXiv:2504.06087.

This is a second, independent solver for ANGEL; the second-quantized NetKet
solver in ``angel_src.NQS.netket_vmc`` is untouched.  Like that solver, FiRE
is label-free: it minimizes the sampled expectation value of the molecular
Hamiltonian and never consumes a correlated reference energy.  See
``README.md`` in this directory for the equation-by-module map and for the
documented deviations from the paper.

Only :mod:`config` and :mod:`system` are imported eagerly; everything that
needs JAX is resolved on first attribute access so that importing
``angel_src.NQS`` stays cheap and dependency-free.
"""

from __future__ import annotations

from typing import Any

from .config import (
    AnsatzConfig,
    FiREConfig,
    MCMCConfig,
    OptConfig,
    PretrainConfig,
    replace_config,
    small_config,
)
from .system import MolecularSystem, system_from_arrays, system_from_xyz

_LAZY: dict[str, str] = {
    "FiREWaveFunction": "angel_src.NQS.fire.ansatz",
    "make_log_psi": "angel_src.NQS.fire.ansatz",
    "local_energy_fn": "angel_src.NQS.fire.hamiltonian",
    "potential_energy": "angel_src.NQS.fire.hamiltonian",
    "init_walkers": "angel_src.NQS.fire.mcmc",
    "make_sampler": "angel_src.NQS.fire.mcmc",
    "spring_update": "angel_src.NQS.fire.optimizer",
    "SpringState": "angel_src.NQS.fire.optimizer",
    "pretrain_to_hartree_fock": "angel_src.NQS.fire.pretrain",
    "extrapolate_energy": "angel_src.NQS.fire.train",
    "extrapolate_shared_slope": "angel_src.NQS.fire.train",
    "optimize": "angel_src.NQS.fire.train",
    "FiREResult": "angel_src.NQS.fire.solver",
    "run_fire_vmc": "angel_src.NQS.fire.solver",
    "solve_geometry_fire": "angel_src.NQS.fire.solver",
}

__all__ = [
    "AnsatzConfig",
    "FiREConfig",
    "FiREResult",
    "FiREWaveFunction",
    "MCMCConfig",
    "MolecularSystem",
    "OptConfig",
    "PretrainConfig",
    "SpringState",
    "extrapolate_energy",
    "extrapolate_shared_slope",
    "init_walkers",
    "local_energy_fn",
    "make_log_psi",
    "make_sampler",
    "optimize",
    "potential_energy",
    "pretrain_to_hartree_fock",
    "replace_config",
    "run_fire_vmc",
    "small_config",
    "solve_geometry_fire",
    "spring_update",
    "system_from_arrays",
    "system_from_xyz",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
