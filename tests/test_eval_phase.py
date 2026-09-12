#!/usr/bin/env python3
"""Regression checks for fixed-parameter evaluation and FiRE checkpoints."""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from angel_src.NQS.fire.ansatz import init_params, make_log_psi  # noqa: E402
from angel_src.NQS.fire.checkpoint import load_params, save_params  # noqa: E402
from angel_src.NQS.fire.config import small_config  # noqa: E402
from angel_src.NQS.fire.system import system_from_arrays  # noqa: E402
from angel_src.NQS.fire.train import blocking_analysis, optimize  # noqa: E402


def test_blocking_analysis() -> None:
    rng = np.random.default_rng(20260912)
    n = 65536
    for phi in (0.0, 0.9):
        values = np.empty(n)
        values[0] = rng.normal()
        innovation = np.sqrt(1.0 - phi**2)
        for index in range(1, n):
            values[index] = phi * values[index - 1] + innovation * rng.normal()
        analysis = blocking_analysis(values)
        ratio = analysis.error / analysis.naive_error
        expected = np.sqrt((1.0 + phi) / (1.0 - phi))
        assert analysis.error >= analysis.naive_error
        if phi == 0.0:
            assert abs(ratio - 1.0) < 0.2, ratio
        else:
            assert 0.5 * expected < ratio < 2.0 * expected, (ratio, expected)


def _tiny_problem():
    system = system_from_arrays(
        ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        units="bohr", name="H2",
    )
    base = small_config()
    config = replace(
        base,
        ansatz=replace(
            base.ansatz, n_determinants=1, hidden_dim=16,
            embedding_mlp_widths=(16,), jastrow_mlp_widths=(16, 16),
            n_registers=2, register_dim=4, n_envelopes=2,
        ),
        mcmc=replace(
            base.mcmc, batch_size=16, n_steps=1, n_global_moves=0, n_burn_in=2,
        ),
        opt=replace(base.opt, steps=3, n_eval_steps=3),
        pretrain=replace(base.pretrain, steps=0),
        seed=77,
    )
    model, log_psi, _ = make_log_psi(system, config.ansatz)
    params = init_params(model, jax.random.PRNGKey(991), system.n_electrons)
    return system, config, params, log_psi


def test_frozen_and_disabled() -> tuple[dict, object, object]:
    system, config, initial, log_psi = _tiny_problem()
    key = jax.random.PRNGKey(123)
    disabled = optimize(system, config, params=initial, key=key)
    enabled_config = replace(
        config, opt=replace(
            config.opt, eval_blocks=4, eval_block_steps=1, eval_burn_in_blocks=1,
        )
    )
    enabled = optimize(system, enabled_config, params=initial, key=key)
    for left, right in zip(
        jax.tree_util.tree_leaves(disabled["params"]),
        jax.tree_util.tree_leaves(enabled["params"]),
    ):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert enabled["eval_blocks_used"] == 4
    assert (enabled["energy_eval_error_hartree"]
            >= enabled["energy_eval_naive_error_hartree"])

    old_keys = {
        "params", "history", "energy_hartree", "energy_error_hartree",
        "variance_hartree2", "energy_extrapolated_hartree", "extrapolation_slope",
        "n_eval_steps", "laplacian_backend", "mcmc_steps_per_block", "sampler_state",
    }
    assert old_keys <= disabled.keys()
    assert disabled["eval_blocks_used"] == 0
    assert np.isnan(disabled["energy_eval_hartree"])
    assert np.isnan(disabled["energy_eval_error_hartree"])
    assert np.isnan(disabled["energy_eval_variance_hartree2"])
    assert np.isnan(disabled["eval_autocorr_time"])
    # A second disabled run pins the fixed-seed path and catches any hidden
    # evaluation draw or compilation side effect in the default branch.
    repeated = optimize(system, config, params=initial, key=key)
    assert disabled["energy_hartree"] == repeated["energy_hartree"]
    return disabled, initial, (system, config, log_psi)


def test_checkpoint(params, context) -> None:
    system, config, log_psi = context
    positions = np.array([[[0.2, -0.3, 0.4], [0.1, 0.5, 1.1]]])
    before = np.asarray(jax.vmap(log_psi, in_axes=(None, 0))(params, positions))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "h2_params"
        save_params(path, params, config=config, system=system)
        loaded = load_params(path)
        after = np.asarray(jax.vmap(log_psi, in_axes=(None, 0))(loaded, positions))
    assert np.array_equal(before, after), (before, after)


def main() -> int:
    test_blocking_analysis()
    disabled, _, context = test_frozen_and_disabled()
    test_checkpoint(disabled["params"], context)
    print("blocking estimator correct")
    print("frozen evaluation leaves parameters unchanged")
    print("checkpoint round-trip is bitwise exact")
    print("evaluation disabled path preserves existing results and sentinels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
