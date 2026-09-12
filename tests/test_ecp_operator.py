#!/usr/bin/env python3
"""Unit tests for the ECP operator, against cases with closed-form answers.

The projector integral

    (2l+1) Int dOmega/(4 pi) P_l(cos theta') Psi(...Omega'...) / Psi

collapses to 1 whenever the wavefunction's angular dependence about the
nucleus is itself the l-th spherical harmonic, so a pure s trial must return
exactly ``sum_i V_0(r_i)`` through the l = 0 channel and a pure p_z trial
exactly ``sum_i V_1(r_i)`` through l = 1.  Those two are checked to machine
precision; the quadrature exactness and the cutoff are checked directly.

Usage::

    python tests/test_ecp_operator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from angel_src.NQS.fire.ecp import (  # noqa: E402
    _radial,
    ecp_potential_fn,
    legendre,
    quadrature_grid,
)
from angel_src.NQS.fire.hamiltonian import local_energy_fn  # noqa: E402
from angel_src.NQS.fire.system import (  # noqa: E402
    ECPChannel,
    MolecularSystem,
    NuclearECP,
    system_from_arrays,
)

TOL = 1e-11


def _positions(seed: int, n_electrons: int, scale: float = 1.0) -> jnp.ndarray:
    return scale * jax.random.normal(jax.random.PRNGKey(seed), (n_electrons, 3))


def test_quadrature_exactness() -> None:
    """Weights normalised, and Int P_l dOmega = 0 within each grid's order."""

    for n_points, exact_through in ((6, 3), (12, 5)):
        points, weights = quadrature_grid(n_points)
        assert np.isclose(weights.sum(), 1.0)
        assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
        # Integrate against an arbitrary axis, under random rotations.
        for seed in range(3):
            rotation = np.asarray(jax.random.orthogonal(jax.random.PRNGKey(seed), 3))
            rotated = points @ rotation.T
            axis = rotation[0]
            cos_theta = rotated @ axis
            for order in range(1, min(exact_through, 4) + 1):
                value = float(np.sum(weights * legendre(order, cos_theta)))
                assert abs(value) < 1e-12, (n_points, order, value)


def test_s_wave_projector_exact() -> None:
    """A spherically symmetric trial: every sphere ratio is 1."""

    system = system_from_arrays(["Be"], np.zeros((1, 3)), units="bohr", ecp="ccecp")
    entry = system.ecp[0]
    assert entry.max_angular_momentum == 0

    def sign_log_psi(_params, r):
        return jnp.array(1.0), -jnp.sum(jnp.linalg.norm(r, axis=-1))

    operator = jax.jit(ecp_potential_fn(sign_log_psi, system, n_quadrature=12))
    r = _positions(3, system.n_electrons)
    distance = jnp.linalg.norm(r, axis=-1)
    expected = float(
        jnp.sum(_radial(entry.local_channel, distance, jnp))
        + jnp.sum(_radial(entry.nonlocal_channels[0], distance, jnp))
    )
    values = [float(operator(None, r, jax.random.PRNGKey(k))) for k in range(4)]
    for value in values:
        assert abs(value - expected) < TOL, (value, expected)
    # Rotation of the grid cannot matter for this trial.
    assert max(values) - min(values) < TOL


def _synthetic_p_system() -> MolecularSystem:
    """Be with an artificial l = 1 projector added to the ccECP data."""

    base = system_from_arrays(["Be"], np.zeros((1, 3)), units="bohr", ecp="ccecp")
    entry = base.ecp[0]
    p_channel = ECPChannel(
        angular_momentum=1,
        coefficients=np.array([3.7]),
        exponents=np.array([1.3]),
        r_powers=np.array([0.0]),
    )
    patched = NuclearECP(
        symbol=entry.symbol,
        name=entry.name,
        n_core=entry.n_core,
        z_eff=entry.z_eff,
        channels=entry.channels + (p_channel,),
    )
    return MolecularSystem(
        symbols=base.symbols,
        nuclei=base.nuclei,
        charges=base.charges,
        n_up=base.n_up,
        n_down=base.n_down,
        name="Be_synthetic_p",
        ecp=(patched,),
    )


def test_p_wave_projector_exact() -> None:
    """A pure p_z trial: the l = 1 channel integrates to sum_i V_1(r_i)."""

    system = _synthetic_p_system()
    entry = system.ecp[0]

    def sign_log_psi(_params, r):
        z = r[:, 2]
        radius = jnp.linalg.norm(r, axis=-1)
        return jnp.prod(jnp.sign(z)), jnp.sum(jnp.log(jnp.abs(z)) - radius)

    operator = jax.jit(ecp_potential_fn(sign_log_psi, system, n_quadrature=12))
    r = _positions(11, system.n_electrons)
    distance = jnp.linalg.norm(r, axis=-1)
    # The l = 0 projector averages Omega_z over the sphere and so vanishes for
    # a pure p trial; only the local term and the l = 1 projector survive.
    expected = float(
        jnp.sum(_radial(entry.local_channel, distance, jnp))
        + jnp.sum(_radial(entry.nonlocal_channels[1], distance, jnp))
    )
    for seed in range(4):
        value = float(operator(None, r, jax.random.PRNGKey(seed)))
        assert abs(value - expected) < 1e-9, (value, expected)


def test_local_only_potential() -> None:
    """ccECP hydrogen has no projector: the operator is the local term."""

    system = system_from_arrays(
        ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]), units="bohr", ecp="ccecp"
    )
    assert all(entry.nonlocal_channels == () for entry in system.ecp)

    def sign_log_psi(_params, r):
        return jnp.array(1.0), -jnp.sum(jnp.linalg.norm(r, axis=-1))

    operator = jax.jit(ecp_potential_fn(sign_log_psi, system))
    r = _positions(5, system.n_electrons)
    expected = 0.0
    for nucleus, entry in zip(system.nuclei, system.ecp):
        distance = jnp.linalg.norm(r - jnp.asarray(nucleus), axis=-1)
        expected += float(jnp.sum(_radial(entry.local_channel, distance, jnp)))
    assert abs(float(operator(None, r, jax.random.PRNGKey(0))) - expected) < TOL


def test_cutoff_suppresses_distant_electrons() -> None:
    """Electrons past the cutoff contribute nothing."""

    system = system_from_arrays(["Be"], np.zeros((1, 3)), units="bohr", ecp="ccecp")

    def sign_log_psi(_params, r):
        return jnp.array(1.0), -jnp.sum(jnp.linalg.norm(r, axis=-1))

    r = jnp.array([[0.8, 0.0, 0.0], [40.0, 0.0, 0.0]])
    key = jax.random.PRNGKey(0)

    default_cutoff = jax.jit(ecp_potential_fn(sign_log_psi, system, cutoff=10.0))
    no_cutoff = jax.jit(ecp_potential_fn(sign_log_psi, system, cutoff=1e4))
    # The tabulated potential is numerically zero at 40 bohr anyway, so the
    # cutoff must not change the value -- it only avoids the far-field ratios.
    assert abs(float(default_cutoff(None, r, key)) - float(no_cutoff(None, r, key))) < TOL
    assert abs(float(default_cutoff(None, r, key))) > 1e-3  # the near electron counts

    # Cutting inside every electron zeroes the operator exactly.
    everything_cut = jax.jit(ecp_potential_fn(sign_log_psi, system, cutoff=0.5))
    assert float(everything_cut(None, r, key)) == 0.0


def test_default_signature_unchanged() -> None:
    """No ECP: two-argument local energy, and no ECP machinery is touched."""

    system = system_from_arrays(["Be"], np.zeros((1, 3)), units="bohr")
    assert not system.has_ecp

    def log_psi(_params, r):
        return -jnp.sum(jnp.linalg.norm(r, axis=-1))

    energy = local_energy_fn(log_psi, system, backend="loop")
    value = float(energy(None, _positions(7, system.n_electrons)))
    assert np.isfinite(value)

    ecp_system = system_from_arrays(["Be"], np.zeros((1, 3)), units="bohr", ecp="ccecp")
    try:
        local_energy_fn(log_psi, ecp_system, backend="loop")
    except ValueError as error:
        assert "sign_log_psi" in str(error)
    else:  # pragma: no cover
        raise AssertionError("ECP system must require sign_log_psi")


def main() -> int:
    tests = [
        test_quadrature_exactness,
        test_s_wave_projector_exact,
        test_p_wave_projector_exact,
        test_local_only_potential,
        test_cutoff_suppresses_distant_electrons,
        test_default_signature_unchanged,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
