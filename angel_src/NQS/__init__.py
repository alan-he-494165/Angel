"""Variational neural-quantum-state solvers for ANGEL.

Two independent, label-free solvers live here:

* ``netket_vmc`` - second-quantized NQS on the frozen-core active-space
  Hamiltonian built by ``angel_src.electronic.fci`` (NetKet, complex RBM).
* ``fire`` - real-space neural-network VMC with finite-range embeddings
  (arXiv:2504.06087), in JAX.

The FiRE symbols are resolved lazily so that importing this package does not
require JAX.
"""

from typing import Any

from .netket_vmc import (
    NQSConfig,
    VMCResult,
    build_netket_hamiltonian,
    run_vmc,
    solve_geometry,
)

_FIRE_EXPORTS = {
    "FiREConfig",
    "FiREResult",
    "MolecularSystem",
    "fire_small_config",
    "run_fire_vmc",
    "solve_geometry_fire",
    "system_from_xyz",
}

__all__ = [
    "NQSConfig",
    "VMCResult",
    "build_netket_hamiltonian",
    "run_vmc",
    "solve_geometry",
    *sorted(_FIRE_EXPORTS),
]


def __getattr__(name: str) -> Any:
    """Resolve FiRE exports on first access (keeps JAX out of import time)."""

    if name not in _FIRE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import fire

    alias = {"fire_small_config": "small_config"}.get(name, name)
    return getattr(fire, alias)


def __dir__() -> list[str]:
    return sorted(__all__)
