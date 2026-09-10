"""Label-free NetKet VMC for frozen-core electronic Hamiltonians.

The solver consumes the Hamiltonian dictionary produced by
``angel_src.electronic.fci.get_electronic_structure``.  It represents the
active space in a spin-orbital occupation basis and trains a complex RBM by
minimizing the sampled Hamiltonian expectation value.  No FCI energy or CI
vector is used as a training target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class NQSConfig:
    """Configuration for a NetKet VMC run."""

    n_iterations: int = 1_000
    n_samples: int = 4_096
    n_chains: int = 64
    n_discard_per_chain: int = 100
    rbm_alpha: float = 2.0
    learning_rate: float = 0.01
    sr_diag_shift: float = 0.01
    seed: int = 1234
    sampler_seed: int | None = None
    use_sr: bool = True
    log_path: Path | None = None
    chunk_size: int | None = None


@dataclass(frozen=True)
class VMCResult:
    """Summary of a variational NQS optimization."""

    energy_hartree: float
    energy_error_hartree: float | None
    energy_variance_hartree2: float | None
    n_iterations: int
    seed: int
    backend: str


def _require_netket():
    try:
        import netket as nk
    except ImportError as error:
        raise ImportError(
            "NetKet VMC requires netket, jax, flax, and optax. "
            "Install a NetKet version compatible with the active Python/JAX environment."
        ) from error
    return nk


def _spin_orbital_index(orbital: int, spin: int, n_orbitals: int) -> int:
    """Map spatial orbital/spin to alpha-then-beta spin-orbital ordering."""
    return orbital + spin * n_orbitals


def build_netket_hamiltonian(hamiltonian: Mapping[str, Any]):
    """Build a NetKet fermionic operator from a frozen-core Hamiltonian.

    The input uses PySCF's active-space convention:

    ``H = ecore + h[p,q] a†[p] a[q] +
    1/2 eri[p,q,r,s] a†[p] a†[r] a[s] a[q]``.

    ``nelec`` must contain the active alpha and beta electron counts.  The
    nuclear and frozen-core contribution in ``ecore`` is retained as a scalar
    shift in the variational energy.
    """
    import numpy as np

    nk = _require_netket()
    h1e = np.asarray(hamiltonian["h1e"])
    eri = np.asarray(hamiltonian["eri"])
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be a square active-space matrix.")
    n_orbitals = h1e.shape[0]
    if eri.shape != (n_orbitals, n_orbitals, n_orbitals, n_orbitals):
        raise ValueError("eri must have shape (norb, norb, norb, norb).")
    nelec = hamiltonian["nelec"]
    if not isinstance(nelec, (tuple, list)) or len(nelec) != 2:
        raise ValueError("nelec must be a (nalpha, nbeta) pair.")
    nalpha, nbeta = (int(nelec[0]), int(nelec[1]))

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=n_orbitals,
        s=0.5,
        n_fermions_per_spin=(nalpha, nbeta),
    )
    terms: list[tuple[tuple[int, int], ...]] = []
    weights: list[complex] = []

    for p in range(n_orbitals):
        for q in range(n_orbitals):
            coefficient = h1e[p, q]
            if abs(coefficient) == 0:
                continue
            for spin in (0, 1):
                terms.append(
                    (
                        (_spin_orbital_index(p, spin, n_orbitals), 1),
                        (_spin_orbital_index(q, spin, n_orbitals), 0),
                    )
                )
                weights.append(complex(coefficient))

    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    coefficient = 0.5 * eri[p, q, r, s]
                    if abs(coefficient) == 0:
                        continue
                    for spin_p in (0, 1):
                        for spin_r in (0, 1):
                            terms.append(
                                (
                                    (_spin_orbital_index(p, spin_p, n_orbitals), 1),
                                    (_spin_orbital_index(r, spin_r, n_orbitals), 1),
                                    (_spin_orbital_index(s, spin_r, n_orbitals), 0),
                                    (_spin_orbital_index(q, spin_p, n_orbitals), 0),
                                )
                            )
                            weights.append(complex(coefficient))

    return nk.operator.FermionOperator2nd(
        hilbert,
        terms=terms,
        weights=weights,
        constant=float(hamiltonian.get("ecore", 0.0)),
        dtype=complex,
    )


def _build_variational_state(hamiltonian, config: NQSConfig):
    nk = _require_netket()
    sampler = nk.sampler.MetropolisFermionHop(
        hamiltonian.hilbert,
        spin_symmetric=True,
        n_chains=config.n_chains,
        n_discard_per_chain=config.n_discard_per_chain,
    )
    model = nk.models.RBMMultiVal(
        n_classes=2,
        alpha=config.rbm_alpha,
        param_dtype=complex,
    )
    return nk.vqs.MCState(
        sampler,
        model,
        n_samples=config.n_samples,
        n_discard_per_chain=config.n_discard_per_chain,
        chunk_size=config.chunk_size,
        seed=config.seed,
        sampler_seed=config.sampler_seed or config.seed + 1,
    )


def _as_float(value: Any) -> float | None:
    import numpy as np

    if value is None:
        return None
    return float(np.asarray(value))


def run_vmc(hamiltonian: Mapping[str, Any], config: NQSConfig | None = None) -> VMCResult:
    """Train an NQS variationally on a frozen-core active-space Hamiltonian."""
    config = config or NQSConfig()
    if config.n_iterations < 1 or config.n_samples < 1:
        raise ValueError("n_iterations and n_samples must be positive.")
    nk = _require_netket()
    operator = build_netket_hamiltonian(hamiltonian)
    vstate = _build_variational_state(operator, config)
    optimizer = nk.optimizer.Sgd(learning_rate=config.learning_rate)
    preconditioner = nk.optimizer.SR(diag_shift=config.sr_diag_shift) if config.use_sr else None
    driver = nk.driver.VMC(
        operator,
        optimizer,
        variational_state=vstate,
        preconditioner=preconditioner,
    )

    out = None
    if config.log_path is not None:
        config.log_path.parent.mkdir(parents=True, exist_ok=True)
        out = nk.logging.JsonLog(str(config.log_path), save_params=False)
    driver.run(n_iter=config.n_iterations, out=out)

    stats = vstate.expect(operator)
    return VMCResult(
        energy_hartree=_as_float(stats.mean),
        energy_error_hartree=_as_float(getattr(stats, "error_of_mean", None)),
        energy_variance_hartree2=_as_float(getattr(stats, "variance", None)),
        n_iterations=config.n_iterations,
        seed=config.seed,
        backend=str(nk.config.gpu_backend if hasattr(nk.config, "gpu_backend") else "jax"),
    )


def solve_geometry(
    geometry: str | Path,
    basis: str,
    *,
    active_electrons: int,
    active_orbitals: int,
    config: NQSConfig | None = None,
    charge: int = 0,
    spin: int = 0,
    use_gpu: bool | None = None,
) -> VMCResult:
    """Build a frozen-core Hamiltonian and solve it with label-free VMC."""
    from angel_src.electronic.fci import get_electronic_structure

    electronic = get_electronic_structure(
        geometry,
        basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        charge=charge,
        spin=spin,
        hamiltonian_only=True,
        calculate_energy=False,
        use_gpu=use_gpu,
    )
    return run_vmc(electronic.hamiltonian, config=config)
