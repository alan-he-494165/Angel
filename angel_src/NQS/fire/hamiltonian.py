"""Local energy of the molecular Hamiltonian (Eq. 9-11 of arXiv:2504.06087).

    E_L(r) = -1/2 ( Laplacian log|Psi| + |grad log|Psi||^2 ) + V(r)

with the all-electron Coulomb potential

    V(r) = - sum_{i,m} Z_m / |r_i - R_m|
           + sum_{i<j} 1 / |r_i - r_j|
           + sum_{m<n} Z_m Z_n / |R_m - R_n|.

Two Laplacian backends are provided.  ``folx`` (the forward-Laplacian
framework the paper builds on) is used when importable; otherwise an exact
forward-mode fallback contracts 3 n_el Hessian-vector products.  Both are
numerically equivalent; the fallback is O(n_el) times more expensive.

Effective core potentials are *not* implemented.  The paper uses ccECP for
its large systems (Sec. G); this solver is all-electron, which is the
consistent choice for the first- and second-row aromatic dimers in ANGEL's
scope but limits the affordable system size.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .system import MolecularSystem

Array = jax.Array


def potential_energy(r: Array, nuclei: Array, charges: Array, nuclear_repulsion: float) -> Array:
    """Coulomb potential energy of one electron configuration, in hartree."""

    n_el = r.shape[0]
    displacement_en = r[:, None, :] - nuclei[None, :, :]
    distance_en = jnp.sqrt(jnp.sum(displacement_en**2, axis=-1) + 1e-24)
    electron_nucleus = -jnp.sum(charges[None, :] / distance_en)

    displacement_ee = r[:, None, :] - r[None, :, :]
    squared = jnp.sum(displacement_ee**2, axis=-1) + 1e-24
    distance_ee = jnp.sqrt(jnp.where(jnp.eye(n_el, dtype=bool), 1.0, squared))
    upper = jnp.triu(jnp.ones((n_el, n_el), dtype=bool), k=1)
    electron_electron = jnp.sum(jnp.where(upper, 1.0 / distance_ee, 0.0))

    return electron_nucleus + electron_electron + nuclear_repulsion


def _laplacian_loop(log_psi_r: Callable[[Array], Array], r: Array) -> tuple[Array, Array]:
    """Exact (gradient, Laplacian) via 3 n_el Hessian-vector products."""

    shape = r.shape
    flat = r.reshape(-1)
    n_coords = flat.size

    def flat_log_psi(x: Array) -> Array:
        return log_psi_r(x.reshape(shape))

    grad_fn = jax.grad(flat_log_psi)
    basis = jnp.eye(n_coords, dtype=flat.dtype)

    def diagonal_entry(unit: Array) -> Array:
        gradient, hessian_column = jax.jvp(grad_fn, (flat,), (unit,))
        return jnp.dot(unit, hessian_column)

    laplacian = jnp.sum(jax.vmap(diagonal_entry)(basis))
    return grad_fn(flat).reshape(shape), laplacian


def _laplacian_folx(log_psi_r: Callable[[Array], Array], r: Array) -> tuple[Array, Array]:
    """(gradient, Laplacian) through the folx forward-Laplacian framework."""

    import folx

    forward = folx.forward_laplacian(log_psi_r)
    result = forward(r)
    return result.jacobian.dense_array.reshape(r.shape), result.laplacian


def folx_available() -> bool:
    try:  # pragma: no cover - depends on the environment
        import folx  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(requested: str) -> str:
    """Resolve ``"auto"`` to an available Laplacian backend."""

    if requested not in ("auto", "folx", "loop"):
        raise ValueError(f"unknown laplacian_backend {requested!r}.")
    if requested == "auto":
        return "folx" if folx_available() else "loop"
    if requested == "folx" and not folx_available():
        raise ImportError("laplacian_backend='folx' requires the folx package.")
    return requested


def kinetic_energy_fn(
    log_psi: Callable[..., Array], backend: str = "loop"
) -> Callable[..., Array]:
    """Return ``(params, r) -> -1/2 (Laplacian + |grad|^2) log|Psi|``."""

    laplacian = _laplacian_folx if backend == "folx" else _laplacian_loop

    def kinetic(params, r: Array) -> Array:
        gradient, lap = laplacian(lambda x: log_psi(params, x), r)
        return -0.5 * (lap + jnp.sum(gradient**2))

    return kinetic


def local_energy_fn(
    log_psi: Callable[..., Array],
    system: MolecularSystem,
    backend: str = "auto",
    *,
    sign_log_psi: Callable[..., tuple] | None = None,
    ecp_quadrature: int = 12,
    ecp_cutoff: float = 10.0,
    ecp_max_log_ratio: float = 20.0,
) -> Callable[..., Array]:
    """Return the local energy of a single configuration.

    For an all-electron system -- the default -- the returned callable is
    ``(params, r) -> E_L(r)``; batch it with
    ``jax.vmap(local_energy, in_axes=(None, 0))``.

    When ``system`` carries an effective core potential the nonlocal channels
    need a randomly rotated quadrature grid, so the signature becomes
    ``(params, r, key) -> E_L(r)`` and ``sign_log_psi`` is required (the
    projector integral needs the sign of the wavefunction, not only its
    logarithm).  Batch that form with
    ``jax.vmap(local_energy, in_axes=(None, 0, 0))``.

    The branch is taken once, here at construction time, so the all-electron
    path traces to exactly the graph it did before ECP support existed.
    """

    backend = resolve_backend(backend)
    nuclei = jnp.asarray(system.nuclei)
    charges = jnp.asarray(system.charges)
    nuclear_repulsion = float(system.nuclear_repulsion)
    kinetic = kinetic_energy_fn(log_psi, backend)

    if not system.has_ecp:

        def local_energy(params, r: Array) -> Array:
            return kinetic(params, r) + potential_energy(
                r, nuclei, charges, nuclear_repulsion
            )

        return local_energy

    if sign_log_psi is None:
        raise ValueError(
            "an ECP system needs sign_log_psi; pass the third return value of "
            "make_log_psi()."
        )

    from .ecp import ecp_potential_fn

    v_ecp = ecp_potential_fn(
        sign_log_psi,
        system,
        n_quadrature=ecp_quadrature,
        cutoff=ecp_cutoff,
        max_log_ratio=ecp_max_log_ratio,
    )

    def local_energy_with_ecp(params, r: Array, key: Array) -> Array:
        return (
            kinetic(params, r)
            + potential_energy(r, nuclei, charges, nuclear_repulsion)
            + v_ecp(params, r, key)
        )

    return local_energy_with_ecp
