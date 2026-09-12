"""Metropolis-Hastings sampling of |Psi|^2 (Sec. 4.5 of arXiv:2504.06087).

Each block performs ``n_steps`` all-electron local Gaussian sweeps with an
on-the-fly adapted proposal width (target acceptance 50 %), followed by
``n_global_moves`` global single-electron jumps drawn from the charge-weighted
Gaussian mixture of Eq. 46,

    rho_global(r') = (1 / sum_m Z_m) sum_m Z_m N(r' | R_m, sigma_g^2 I),

whose asymmetry is corrected by the Hastings factor of Eq. 47.  The paper
uses ``sigma_g = 2 a0`` and ``n_steps = 2 n_el``.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import MCMCConfig
from .system import MolecularSystem

Array = jax.Array


class SamplerState(NamedTuple):
    """Walker positions plus the adaptive proposal width and diagnostics."""

    positions: Array  # (batch, n_el, 3)
    step_size: Array  # scalar
    key: Array
    acceptance_local: Array
    acceptance_global: Array


def _electrons_per_nucleus(system: MolecularSystem) -> tuple[np.ndarray, np.ndarray]:
    """Distribute spin-up / spin-down electrons over nuclei by charge.

    Deterministic largest-remainder assignment, used only to initialize the
    walkers; it has no effect on the sampled distribution.
    """

    charges = np.asarray(system.charges, dtype=float)
    total = charges.sum()
    counts = []
    for n_spin in (system.n_up, system.n_down):
        share = charges / total * n_spin
        base = np.floor(share).astype(int)
        remainder = n_spin - base.sum()
        if remainder > 0:
            order = np.argsort(-(share - base))
            base[order[:remainder]] += 1
        counts.append(base)
    return counts[0], counts[1]


def init_walkers(key: Array, system: MolecularSystem, batch_size: int, width: float = 1.0) -> Array:
    """Draw ``(batch_size, n_el, 3)`` starting positions around the nuclei."""

    up_counts, down_counts = _electrons_per_nucleus(system)
    centers = []
    for counts in (up_counts, down_counts):
        for nucleus, count in enumerate(counts):
            centers.extend([system.nuclei[nucleus]] * int(count))
    centers_array = jnp.asarray(np.stack(centers))  # (n_el, 3), up block first
    noise = jax.random.normal(key, (batch_size,) + centers_array.shape)
    return centers_array[None] + width * noise


def _gmm_log_density(x: Array, nuclei: Array, weights: Array, sigma: float) -> Array:
    """Log density of the charge-weighted Gaussian mixture at ``x`` (3-vector)."""

    squared = jnp.sum((x[None, :] - nuclei) ** 2, axis=-1)
    logits = jnp.log(weights) - 0.5 * squared / sigma**2
    normalization = -1.5 * jnp.log(2.0 * jnp.pi * sigma**2)
    return jax.scipy.special.logsumexp(logits) + normalization


def make_sampler(
    log_psi: Callable[..., Array],
    system: MolecularSystem,
    config: MCMCConfig,
) -> tuple[Callable[..., SamplerState], Callable[..., SamplerState]]:
    """Return ``(init_state, sample_block)`` for |Psi|^2 sampling.

    ``sample_block(params, state, n_steps)`` runs one block of ``n_steps``
    local sweeps plus ``config.n_global_moves`` global jumps and returns the
    updated :class:`SamplerState`.
    """

    nuclei = jnp.asarray(system.nuclei)
    charges = jnp.asarray(system.charges)
    weights = charges / jnp.sum(charges)
    sigma_g = float(config.global_sigma)
    n_el = system.n_electrons

    batched_log_psi = jax.vmap(log_psi, in_axes=(None, 0))

    def init_state(key: Array, batch_size: int | None = None) -> SamplerState:
        key, subkey = jax.random.split(key)
        positions = init_walkers(subkey, system, batch_size or config.batch_size)
        return SamplerState(
            positions=positions,
            step_size=jnp.asarray(config.step_size, dtype=positions.dtype),
            key=key,
            acceptance_local=jnp.asarray(0.0, dtype=positions.dtype),
            acceptance_global=jnp.asarray(0.0, dtype=positions.dtype),
        )

    def _local_sweep(carry, _):
        params, positions, log_amplitude, step_size, key, accepted = carry
        key, move_key, accept_key = jax.random.split(key, 3)
        proposal = positions + step_size * jax.random.normal(move_key, positions.shape)
        log_amplitude_new = batched_log_psi(params, proposal)
        log_ratio = 2.0 * (log_amplitude_new - log_amplitude)
        uniform = jax.random.uniform(accept_key, log_ratio.shape)
        accept = jnp.log(uniform) < log_ratio
        positions = jnp.where(accept[:, None, None], proposal, positions)
        log_amplitude = jnp.where(accept, log_amplitude_new, log_amplitude)
        carry = (params, positions, log_amplitude, step_size, key, accepted + jnp.mean(accept))
        return carry, None

    def _global_moves(carry, _):
        params, positions, log_amplitude, key, accepted = carry
        batch = positions.shape[0]
        key, index_key, center_key, move_key, accept_key = jax.random.split(key, 5)
        electron = jax.random.randint(index_key, (batch,), 0, n_el)
        center = jax.random.choice(center_key, nuclei.shape[0], (batch,), p=weights)
        new_position = nuclei[center] + sigma_g * jax.random.normal(move_key, (batch, 3))
        one_hot = jax.nn.one_hot(electron, n_el, dtype=positions.dtype)[:, :, None]
        old_position = jnp.sum(one_hot * positions, axis=1)
        proposal = positions * (1.0 - one_hot) + one_hot * new_position[:, None, :]

        log_amplitude_new = batched_log_psi(params, proposal)
        density = jax.vmap(_gmm_log_density, in_axes=(0, None, None, None))
        hastings = density(old_position, nuclei, weights, sigma_g) - density(
            new_position, nuclei, weights, sigma_g
        )
        log_ratio = 2.0 * (log_amplitude_new - log_amplitude) + hastings
        uniform = jax.random.uniform(accept_key, log_ratio.shape)
        accept = jnp.log(uniform) < log_ratio
        positions = jnp.where(accept[:, None, None], proposal, positions)
        log_amplitude = jnp.where(accept, log_amplitude_new, log_amplitude)
        return (params, positions, log_amplitude, key, accepted + jnp.mean(accept)), None

    def sample_block(params, state: SamplerState, n_steps: int) -> SamplerState:
        log_amplitude = batched_log_psi(params, state.positions)
        zero = jnp.zeros((), dtype=state.positions.dtype)
        carry = (params, state.positions, log_amplitude, state.step_size, state.key, zero)
        carry, _ = jax.lax.scan(_local_sweep, carry, None, length=n_steps)
        _, positions, log_amplitude, step_size, key, accepted_local = carry
        acceptance_local = accepted_local / n_steps

        # Adapt the local proposal width towards the target acceptance ratio.
        step_size = step_size * jnp.exp(
            config.step_size_adaptation * (acceptance_local - config.target_acceptance)
        )

        acceptance_global = jnp.asarray(0.0, dtype=positions.dtype)
        if config.n_global_moves > 0:
            carry = (params, positions, log_amplitude, key, zero)
            carry, _ = jax.lax.scan(_global_moves, carry, None, length=config.n_global_moves)
            _, positions, log_amplitude, key, accepted_global = carry
            acceptance_global = accepted_global / config.n_global_moves

        return SamplerState(
            positions=positions,
            step_size=step_size,
            key=key,
            acceptance_local=acceptance_local,
            acceptance_global=acceptance_global,
        )

    return init_state, sample_block


def default_n_steps(system: MolecularSystem, config: MCMCConfig) -> int:
    """The paper's rule: 2 n_el decorrelation sweeps between updates."""

    return int(config.n_steps) if config.n_steps is not None else 2 * system.n_electrons
