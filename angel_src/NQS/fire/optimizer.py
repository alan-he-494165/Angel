"""SPRING parameter updates (Eq. 12-14 of arXiv:2504.06087).

With the centred, N^{-1/2}-scaled log-derivative matrix

    O_ij = (d log|Psi(r_i)| / d theta_j - mean_i) / sqrt(N)

and the centred, scaled local energies eps_i = (E_L(r_i) - mean) / sqrt(N),
the SPRING update is

    delta_t = O^T (O O^T + lambda I)^{-1} (eps - O eta delta_{t-1}) + eta delta_{t-1}
    theta_{t+1} = theta_t - lr_t delta_t.

The constant factor 2 in the energy gradient (Eq. 13 is stated only up to
proportionality) is absorbed into the learning rate, as in the reference
SPRING formulation.  The linear system is solved in the sample space, so the
cost is O(N^2 P + N^3) and independent of the parameter count beyond the
matrix product.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

Array = jax.Array


class SpringState(NamedTuple):
    """Momentum term ``delta_{t-1}`` as a flat vector, plus the step index."""

    delta: Array
    step: Array


def init_spring_state(params) -> SpringState:
    flat, _ = ravel_pytree(params)
    return SpringState(delta=jnp.zeros_like(flat), step=jnp.zeros((), dtype=jnp.int32))


def spring_update(
    log_derivatives: Array,
    residuals: Array,
    state: SpringState,
    damping: float,
    decay: float,
    learning_rate: float | Array | None = None,
    norm_constraint: float | None = None,
) -> tuple[Array, SpringState]:
    """One SPRING step.

    Parameters
    ----------
    log_derivatives:
        ``(n_samples, n_params)`` matrix of per-sample gradients of
        ``log|Psi|``; centring and the ``1/sqrt(N)`` scaling are applied here.
    residuals:
        ``(n_samples,)`` local energies; centred and scaled here.
    learning_rate, norm_constraint:
        If both are given, the step is rescaled so that the Fisher norm of
        the applied update obeys ``lr^2 delta^T F delta <= norm_constraint``
        with ``F = O^T O``.  The paper does not list a norm constraint, but
        without one the momentum term (eta = 0.99) lets the parameters drift
        without bound on all-electron systems with heavy nuclei; see the
        README "Deviations" section.
    """

    n_samples = log_derivatives.shape[0]
    scale = 1.0 / jnp.sqrt(jnp.asarray(n_samples, dtype=log_derivatives.dtype))
    centred = (log_derivatives - jnp.mean(log_derivatives, axis=0, keepdims=True)) * scale
    epsilon = (residuals - jnp.mean(residuals)) * scale

    momentum = decay * state.delta
    rhs = epsilon - centred @ momentum
    gram = centred @ centred.T
    gram = gram + damping * jnp.eye(n_samples, dtype=gram.dtype)
    solution = jnp.linalg.solve(gram, rhs)
    delta = centred.T @ solution + momentum

    if norm_constraint is not None and learning_rate is not None:
        fisher_norm = jnp.sum((centred @ delta) ** 2)
        scale = jnp.minimum(
            1.0, jnp.sqrt(norm_constraint / (learning_rate**2 * fisher_norm + 1e-30))
        )
        delta = delta * scale
    return delta, SpringState(delta=delta, step=state.step + 1)


def per_sample_log_derivatives(
    log_psi: Callable[..., Array], params, positions: Array
) -> Array:
    """``(n_samples, n_params)`` gradients of ``log|Psi|`` w.r.t. the parameters."""

    _, unravel = ravel_pytree(params)
    del unravel
    grad_fn = jax.grad(log_psi)

    def flat_grad(r: Array) -> Array:
        return ravel_pytree(grad_fn(params, r))[0]

    return jax.vmap(flat_grad)(positions)


def apply_update(params, delta: Array, learning_rate: float | Array):
    """Return ``params - learning_rate * delta`` with ``delta`` flat."""

    flat, unravel = ravel_pytree(params)
    return unravel(flat - learning_rate * delta)


def learning_rate_at(step: Array | int, base: float, decay_time: float) -> Array:
    """Schedule ``lr_0 / (1 + t / decay_time)`` of Tab. S1."""

    return base / (1.0 + jnp.asarray(step, dtype=jnp.float32) / decay_time)


def clip_local_energies(
    energies: Array, width: float = 5.0, statistic: str = "median"
) -> Array:
    """Clip local energies at ``width`` mean absolute deviations (Tab. S1)."""

    if statistic == "median":
        center = jnp.median(energies)
    elif statistic == "mean":
        center = jnp.mean(energies)
    else:
        raise ValueError(f"unknown clip statistic {statistic!r}.")
    deviation = jnp.mean(jnp.abs(energies - center))
    return jnp.clip(energies, center - width * deviation, center + width * deviation)
