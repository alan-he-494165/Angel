#!/usr/bin/env python3
"""Check that FiRE envelopes are cusp-free exactly at ECP nuclei."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from angel_src.NQS.fire.ansatz import init_params, make_log_psi  # noqa: E402
from angel_src.NQS.fire.config import small_config  # noqa: E402
from angel_src.NQS.fire.system import system_from_arrays  # noqa: E402


def _models():
    """Equivalent one-electron ansatzes, differing only in core treatment."""

    coordinates = np.zeros((1, 3))
    all_electron = system_from_arrays(
        ["H"], coordinates, units="bohr", spin=1, name="H_all_electron"
    )
    ecp = system_from_arrays(
        ["H"], coordinates, units="bohr", spin=1, name="H_ecp", ecp="ccecp"
    )
    base = small_config().ansatz
    config = replace(
        base,
        n_determinants=1,
        hidden_dim=4,
        embedding_mlp_widths=(4,),
        jastrow_mlp_widths=(4,),
        n_registers=1,
        register_dim=2,
        n_envelopes=1,
    )
    return all_electron, ecp, config


def _controlled_envelope_params(params):
    """Remove other radial dependence while retaining the learned envelope."""

    controlled = jax.tree_util.tree_map(jnp.zeros_like, params)
    mutable = jax.tree_util.tree_map(np.asarray, controlled)
    # The last embedding bias makes h constant; the orbital readout then
    # supplies a nonzero constant prefactor.  The one-electron cusp Jastrow is
    # identically zero, and its two learned terms remain zero.
    mutable["params"]["embedding_mlp"]["Dense_1"]["bias"] = np.ones(4)
    mutable["params"]["orbitals"]["readout_up"] = np.ones((1, 4, 1))
    mutable["params"]["orbitals"]["envelope_weights"] = np.ones((1, 1, 1))
    return jax.tree_util.tree_map(jnp.asarray, mutable)


def _radial_intercept(sign_log_psi, params) -> tuple[float, np.ndarray]:
    """Extrapolate the symmetric outward radial slope to the nucleus."""

    radii = np.array([1e-3, 2e-3, 4e-3])

    def psi(x: float) -> float:
        sign, log_abs = sign_log_psi(params, jnp.array([[x, 0.0, 0.0]]))
        return float(sign * jnp.exp(log_abs))

    psi_zero = psi(0.0)
    # Averaging +r and -r cancels any smooth Cartesian odd part.  The
    # remaining one-sided radial slope detects the nonanalytic |r| term of an
    # all-electron exponential cusp, while a Gaussian starts at order r^2.
    slopes = np.array(
        [((psi(r) + psi(-r)) * 0.5 - psi_zero) / r for r in radii]
    )
    intercept = float(np.polyfit(radii, slopes, deg=1)[1])
    return intercept, slopes


def test_cusp_free_ecp_envelope() -> None:
    all_electron, ecp, config = _models()
    key = jax.random.PRNGKey(20260912)
    model_ae, _, sign_log_ae = make_log_psi(all_electron, config)
    model_ecp, _, sign_log_ecp = make_log_psi(ecp, config)
    params_ae = init_params(model_ae, key, all_electron.n_electrons)
    params_ecp = init_params(model_ecp, key, ecp.n_electrons)

    leaves_ae = jax.tree_util.tree_leaves(params_ae)
    leaves_ecp = jax.tree_util.tree_leaves(params_ecp)
    assert [leaf.shape for leaf in leaves_ae] == [leaf.shape for leaf in leaves_ecp]
    assert len(leaves_ae) == len(leaves_ecp)
    # H ccECP removes no electrons and leaves Z unchanged, so equal seeds also
    # demonstrate that the static mask adds no parameter leaf or PRNG draw.
    assert all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(leaves_ae, leaves_ecp)
    )

    controlled = _controlled_envelope_params(params_ae)
    ecp_intercept, ecp_slopes = _radial_intercept(sign_log_ecp, controlled)
    ae_intercept, ae_slopes = _radial_intercept(sign_log_ae, controlled)

    assert abs(ecp_intercept) < 1e-4, (ecp_intercept, ecp_slopes)
    assert abs(ae_intercept) > 0.5, (ae_intercept, ae_slopes)
    assert abs(ae_intercept) > 1e4 * abs(ecp_intercept), (
        ae_intercept,
        ecp_intercept,
    )
    print(f"ECP radial-slope intercept: {ecp_intercept:.6e}")
    print(f"all-electron radial-slope intercept: {ae_intercept:.6e}")
    print("parameter pytrees are structurally and numerically identical")


def main() -> int:
    test_cusp_free_ecp_envelope()
    print("ok  test_cusp_free_ecp_envelope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
