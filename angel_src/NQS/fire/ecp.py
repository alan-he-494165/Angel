"""Effective core potentials for the FiRE real-space solver (optional).

Frozen core has no literal analogue in a real-space wavefunction: there are no
orbitals to freeze.  The equivalent construction is an effective core
potential, which removes the core electrons from the sampled problem and
replaces them by a potential acting on the remaining valence electrons,

    V_ECP(R) = sum_I [ V_loc,I(r_iI) + sum_l V_l,I(r_iI) P_l,I ] ,

with ``P_l`` projecting onto angular momentum l about nucleus I.  The local
part is an ordinary multiplicative potential.  The projector part is nonlocal
and contributes to the local energy through an integral over the sphere of
radius ``r_iI``,

    (V_l P_l Psi)(R) / Psi(R)
        = (2l+1)/(4 pi) V_l(r_iI) Int dOmega' P_l(cos theta')
          Psi(..., R_I + r_iI Omega', ...) / Psi(R) ,

evaluated here with the icosahedral/octahedral quadratures of Mitas, Shirley
and Ceperley (J. Chem. Phys. 95, 3467 (1991)) on a randomly rotated grid, so
the residual quadrature error averages out over samples instead of biasing a
fixed direction.

In *variational* Monte Carlo this is exact up to that quadrature error: the
mean of E_L over |Psi|^2 is the expectation value of the full nonlocal
Hamiltonian.  The locality approximation and its determinant-only variant,
which are what the QMC literature usually attaches to pseudopotentials, are
needed only to define a local effective potential for a diffusion Monte Carlo
projector; they are not used and not needed here.

This module is imported only when a system carries an ECP.  The default
all-electron path never reaches it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .system import ECPChannel, MolecularSystem


def quadrature_grid(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, weights)`` of a spherical quadrature on the unit sphere.

    ``n_points = 6`` is the octahedral grid (exact through l = 3) and
    ``n_points = 12`` the icosahedral one (exact through l = 5).  Weights sum
    to one, i.e. they already carry the 1/(4 pi) of the surface average.
    """

    if n_points == 6:
        points = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
    elif n_points == 12:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        raw = []
        for sign_a in (1.0, -1.0):
            for sign_b in (1.0, -1.0):
                raw.extend(
                    [
                        [0.0, sign_a * 1.0, sign_b * phi],
                        [sign_a * 1.0, sign_b * phi, 0.0],
                        [sign_b * phi, 0.0, sign_a * 1.0],
                    ]
                )
        points = np.array(raw)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    else:
        raise ValueError(f"unsupported quadrature size {n_points}; use 6 or 12.")
    weights = np.full(len(points), 1.0 / len(points))
    return points, weights


def legendre(order: int, x):
    """Legendre polynomial P_l evaluated with an explicit low-order form."""

    if order == 0:
        return 1.0 + 0.0 * x
    if order == 1:
        return x
    if order == 2:
        return 0.5 * (3.0 * x**2 - 1.0)
    if order == 3:
        return 0.5 * (5.0 * x**3 - 3.0 * x)
    if order == 4:
        return 0.125 * (35.0 * x**4 - 30.0 * x**2 + 3.0)
    raise ValueError(
        f"projector l = {order} is beyond the implemented Legendre orders (l <= 4)."
    )


def _radial(channel: ECPChannel, distance, jnp):
    """V_l(r) for a channel, broadcast over ``distance``."""

    d = distance[..., None]
    coefficients = jnp.asarray(channel.coefficients)
    exponents = jnp.asarray(channel.exponents)
    powers = jnp.asarray(channel.r_powers)
    return jnp.sum(coefficients * d**powers * jnp.exp(-exponents * d**2), axis=-1)


def ecp_potential_fn(
    sign_log_psi: Callable[..., tuple],
    system: MolecularSystem,
    *,
    n_quadrature: int = 12,
    cutoff: float = 10.0,
    max_log_ratio: float = 60.0,
) -> Callable[..., object]:
    """Return ``(params, r, key) -> V_ECP(r)`` for a single configuration.

    ``key`` draws the random rotation of the quadrature grid and must differ
    between walkers and between steps.  Batch with
    ``jax.vmap(v_ecp, in_axes=(None, 0, 0))``.
    """

    import jax
    import jax.numpy as jnp

    if not system.has_ecp:
        raise ValueError("system carries no ECP; use the all-electron local energy.")

    entries = [
        (index, entry)
        for index, entry in enumerate(system.ecp or ())
        if entry is not None
    ]
    grid_np, weights_np = quadrature_grid(n_quadrature)
    grid = jnp.asarray(grid_np)
    weights = jnp.asarray(weights_np)
    nuclei = jnp.asarray(system.nuclei)
    n_electrons = system.n_electrons
    electron_index = jnp.arange(n_electrons)

    def v_ecp(params, r, key):
        base_sign, base_log = sign_log_psi(params, r)
        total = jnp.zeros((), dtype=jnp.asarray(r).dtype)

        for nucleus, entry in entries:
            centre = nuclei[nucleus]
            displacement = r - centre
            distance = jnp.linalg.norm(displacement, axis=-1)
            inside = (
                jnp.ones_like(distance, dtype=bool)
                if cutoff is None
                else distance < cutoff
            )

            local = entry.local_channel
            if local is not None:
                total = total + jnp.sum(jnp.where(inside, _radial(local, distance, jnp), 0.0))

            channels = entry.nonlocal_channels
            if not channels:
                continue

            key, subkey = jax.random.split(key)
            rotation = jax.random.orthogonal(subkey, 3)
            points = grid @ rotation.T
            direction = displacement / distance[:, None]
            cos_theta = direction @ points.T
            moved = centre + distance[:, None, None] * points[None, :, :]

            def ratio(index, position):
                sign, log_amplitude = sign_log_psi(params, r.at[index].set(position))
                delta = jnp.clip(log_amplitude - base_log, -max_log_ratio, max_log_ratio)
                return sign * base_sign * jnp.exp(delta)

            ratios = jax.vmap(
                lambda index, positions: jax.vmap(lambda p: ratio(index, p))(positions)
            )(electron_index, moved)

            for channel in channels:
                order = channel.angular_momentum
                radial = jnp.where(inside, _radial(channel, distance, jnp), 0.0)
                angular = jnp.sum(
                    weights * legendre(order, cos_theta) * ratios, axis=-1
                )
                total = total + (2 * order + 1) * jnp.sum(radial * angular)

        return total

    return v_ecp
