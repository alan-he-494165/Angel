"""Hartree-Fock orbital pretraining (Tab. S1: 2000 Adam steps).

The ansatz orbital matrices Phi_d (Eq. 25) are regressed onto the
block-diagonal matrix of occupied mean-field orbitals evaluated at the
walker positions.  This fixes the *shape* of the orbitals before the
variational optimization starts; it is a mean-field initialization, not a
correlated label, and it is discarded as soon as the energy minimization
begins.

PySCF evaluates the atomic orbitals on the host, so this loop alternates a
jitted Adam step with a NumPy call; the variational loop itself is fully
jitted.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .ansatz import init_params, make_log_psi
from .config import FiREConfig
from .mcmc import default_n_steps, init_walkers, make_sampler
from .system import MolecularSystem

Array = jax.Array


def ecp_consistent_basis(system: MolecularSystem, basis: str) -> str:
    """Basis name to pretrain against, given the system's core treatment.

    An all-electron basis has functions for orbitals the ECP has removed, so
    the mean-field reference must use the matching ECP basis set.  PySCF
    names those ``<family>-<basis>`` (``ccecp-cc-pvdz``), which is what is
    constructed here when the caller leaves the all-electron name in place.
    """

    if not system.has_ecp:
        return basis
    family = system.ecp_name or ""
    if basis.lower().startswith(family.lower()):
        return basis
    return f"{family}-{basis}"


def hartree_fock_orbitals(system: MolecularSystem, basis: str):
    """Run a mean-field calculation and return an MO evaluator.

    Returns ``(evaluate, hf_energy)`` where ``evaluate(coords)`` maps
    ``(n_points, 3)`` bohr coordinates to ``(n_points, n_up)`` and
    ``(n_points, n_down)`` occupied-orbital values for the two spin channels.

    If the system carries an effective core potential, the same potential and
    a matching basis are used here, so the orbitals being fitted describe the
    valence electrons the ansatz actually samples.
    """

    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = system.as_pyscf_atom()
    mol.unit = "Bohr"
    mol.basis = ecp_consistent_basis(system, basis)
    mol.charge = system.charge
    mol.spin = system.spin
    if system.has_ecp:
        mol.ecp = system.pyscf_ecp_spec()
    mol.build()

    mean_field = scf.RHF(mol) if system.spin == 0 else scf.ROHF(mol)
    mean_field.kernel()
    coefficients = np.asarray(mean_field.mo_coeff)
    if coefficients.ndim == 3:  # UHF-like output
        coeff_up, coeff_down = coefficients[0], coefficients[1]
    else:
        coeff_up = coeff_down = coefficients
    occupied_up = coeff_up[:, : system.n_up]
    occupied_down = coeff_down[:, : system.n_down]

    def evaluate(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ao = mol.eval_gto("GTOval_sph", np.asarray(coords, dtype=float))
        return ao @ occupied_up, ao @ occupied_down

    return evaluate, float(mean_field.e_tot)


def hartree_fock_targets(
    positions: np.ndarray, evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], n_up: int
) -> np.ndarray:
    """Block-diagonal HF orbital matrices for a batch of configurations."""

    batch, n_el, _ = positions.shape
    mo_up, mo_down = evaluate(positions.reshape(-1, 3))
    mo_up = mo_up.reshape(batch, n_el, n_up)
    mo_down = mo_down.reshape(batch, n_el, n_el - n_up)
    target = np.zeros((batch, n_el, n_el), dtype=float)
    target[:, :n_up, :n_up] = mo_up[:, :n_up, :]
    target[:, n_up:, n_up:] = mo_down[:, n_up:, :]
    return target


def pretrain_to_hartree_fock(
    system: MolecularSystem,
    config: FiREConfig | None = None,
    *,
    params=None,
    key: Array | None = None,
    callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Fit the ansatz orbitals to the mean-field ones; return params and loss."""

    config = config or FiREConfig()
    settings = config.pretrain
    key = jax.random.PRNGKey(config.seed) if key is None else key

    model, log_psi, _ = make_log_psi(system, config.ansatz)
    key, init_key = jax.random.split(key)
    if params is None:
        params = init_params(model, init_key, system.n_electrons)
    if settings.steps < 1:
        return {"params": params, "loss": [], "hf_energy_hartree": float("nan")}

    evaluate, hf_energy = hartree_fock_orbitals(system, settings.basis)

    batch_size = settings.batch_size or config.mcmc.batch_size
    init_sampler, sample_block = make_sampler(log_psi, system, config.mcmc)
    key, sampler_key = jax.random.split(key)
    sampler_state = init_sampler(sampler_key, batch_size)
    n_steps = default_n_steps(system, config.mcmc)

    # Adam with unit step size; the Tab. S1 schedule 1/(1 + t/1000) is
    # applied to the updates explicitly so it can be logged per step.
    optimizer = optax.adam(learning_rate=1.0)
    opt_state = optimizer.init(params)

    def orbital_loss(params, positions: Array, targets: Array) -> Array:
        def single(r: Array, target: Array) -> Array:
            _, state = model.apply(params, r, mutable=["intermediates"])
            orbitals = state["intermediates"]["orbital_matrices"][0]
            return jnp.mean((orbitals - target[None]) ** 2)

        return jnp.mean(jax.vmap(single)(positions, targets))

    @jax.jit
    def pretrain_step(params, opt_state, positions, targets, learning_rate):
        loss, gradients = jax.value_and_grad(orbital_loss)(params, positions, targets)
        updates, opt_state = optimizer.update(gradients, opt_state, params)
        updates = jax.tree_util.tree_map(lambda u: learning_rate * u, updates)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def draw(params, sampler_state):
        return sample_block(params, sampler_state, n_steps)

    n_refresh = int(round(settings.gaussian_refresh_fraction * batch_size))
    losses: list[float] = []
    for step in range(settings.steps):
        sampler_state = draw(params, sampler_state)
        positions = np.array(sampler_state.positions, dtype=float)
        if n_refresh > 0:
            key, refresh_key = jax.random.split(key)
            positions[:n_refresh] = np.asarray(
                init_walkers(refresh_key, system, n_refresh)
            )
        targets = hartree_fock_targets(positions, evaluate, system.n_up)
        learning_rate = settings.learning_rate / (1.0 + step / settings.lr_decay_time)
        params, opt_state, loss = pretrain_step(
            params,
            opt_state,
            jnp.asarray(positions),
            jnp.asarray(targets),
            learning_rate,
        )
        losses.append(float(loss))
        if callback is not None and step % max(config.log_every, 1) == 0:
            callback(step, {"loss": losses[-1], "learning_rate": learning_rate})

    return {"params": params, "loss": losses, "hf_energy_hartree": hf_energy}
