"""Variational optimization loop for the FiRE ansatz.

One optimization step is: draw a fresh block of walkers from |Psi|^2, evaluate
the local energies, clip them, form the centred log-derivative matrix and
take a SPRING step.  The energy expectation is never compared against a
reference value - the objective is the sampled Hamiltonian expectation value
alone, so the procedure is label-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .ansatz import init_params, make_log_psi
from .config import FiREConfig
from .hamiltonian import local_energy_fn, resolve_backend
from .mcmc import default_n_steps, make_sampler
from .optimizer import (
    SpringState,
    apply_update,
    clip_local_energies,
    init_spring_state,
    learning_rate_at,
    per_sample_log_derivatives,
    spring_update,
)
from .system import MolecularSystem

Array = jax.Array


@dataclass
class TrainingHistory:
    """Per-step optimization diagnostics."""

    step: list[int] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    energy_std: list[float] = field(default_factory=list)
    variance: list[float] = field(default_factory=list)
    grad_norm_sq: list[float] = field(default_factory=list)
    acceptance_local: list[float] = field(default_factory=list)
    acceptance_global: list[float] = field(default_factory=list)
    step_size: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "step": list(self.step),
            "energy_hartree": list(self.energy),
            "energy_std_hartree": list(self.energy_std),
            "variance_hartree2": list(self.variance),
            "grad_norm_sq": list(self.grad_norm_sq),
            "acceptance_local": list(self.acceptance_local),
            "acceptance_global": list(self.acceptance_global),
            "mcmc_step_size": list(self.step_size),
            "learning_rate": list(self.learning_rate),
        }


@dataclass(frozen=True)
class BlockingAnalysis:
    """Flyvbjerg--Petersen error estimate for a scalar correlated series."""

    error: float
    naive_error: float
    block_size: int
    tau_int: float


def blocking_analysis(values: "np.ndarray | list[float]") -> BlockingAnalysis:
    """Find the first stable standard-error plateau under pairwise blocking."""

    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size < 2:
        return BlockingAnalysis(float("nan"), float("nan"), 1, float("nan"))

    errors: list[float] = []
    counts: list[int] = []
    current = series
    while current.size >= 2:
        errors.append(float(current.std(ddof=1) / np.sqrt(current.size)))
        counts.append(int(current.size))
        if current.size < 4:
            break
        current = 0.5 * (current[: current.size // 2 * 2 : 2]
                         + current[1 : current.size // 2 * 2 : 2])

    selected = len(errors) - 1
    for level in range(len(errors) - 1):
        # FP's sampling uncertainty of a standard-error estimate is
        # sigma/sqrt(2(n-1)); the first statistically unchanged level is the
        # autocorrelation plateau rather than a noisy maximum over levels.
        uncertainty = errors[level] / np.sqrt(2.0 * (counts[level] - 1))
        if abs(errors[level] - errors[level + 1]) <= uncertainty:
            selected = level
            break
    naive = errors[0]
    blocked = max(errors[selected], naive)
    tau = 0.5 * (blocked / naive) ** 2 if naive > 0.0 else float("nan")
    return BlockingAnalysis(blocked, naive, 2**selected, tau)


def evaluate_frozen(
    params,
    sampler_state,
    key: Array,
    *,
    sample_block: Callable,
    batched_local_energy: Callable,
    n_steps: int,
    n_blocks: int,
    block_steps: int,
    burn_in_blocks: int,
    callback: Callable[[int, float], None] | None = None,
) -> dict:
    """Continue an equilibrated chain and evaluate unmodified fixed-theta energies."""

    if n_blocks < 1:
        raise ValueError("n_blocks must be positive.")
    if block_steps < 1:
        raise ValueError("block_steps must be positive.")
    if burn_in_blocks < 0:
        raise ValueError("burn_in_blocks must be non-negative.")

    step = jax.jit(lambda state: sample_block(params, state, n_steps))
    energy = jax.jit(lambda positions, energy_key: batched_local_energy(
        params, positions, energy_key
    ))
    for _ in range(burn_in_blocks * block_steps):
        sampler_state = step(sampler_state)

    means: list[float] = []
    samples: list[np.ndarray] = []
    for block in range(n_blocks):
        block_samples: list[np.ndarray] = []
        for _ in range(block_steps):
            sampler_state = step(sampler_state)
            key, energy_key = jax.random.split(key)
            values = np.asarray(energy(sampler_state.positions, energy_key))
            block_samples.append(values)
        values = np.concatenate(block_samples)
        means.append(float(values.mean()))
        samples.append(values)
        if callback is not None:
            callback(block + 1, float(np.mean(means)))

    analysis = blocking_analysis(means)
    all_samples = np.concatenate(samples)
    return {
        "block_means": np.asarray(means),
        "samples": all_samples,
        "energy_hartree": float(np.mean(means)),
        "energy_error_hartree": analysis.error,
        "variance_hartree2": float(all_samples.var()),
        "blocks_used": len(means),
        "autocorr_time": analysis.tau_int,
        "blocking_block_size": analysis.block_size,
        "naive_error_hartree": analysis.naive_error,
        "sampler_state": sampler_state,
        "key": key,
    }


def extrapolate_energy(
    energies: "np.ndarray | list[float]",
    grad_norm_sq: "np.ndarray | list[float]",
    *,
    fit_fraction: float = 0.5,
    n_bins: int = 20,
) -> tuple[float, float]:
    """Fit ``E_t = E_inf + k |g_t|^2`` (Eq. S16) and return ``(E_inf, k)``.

    The last ``fit_fraction`` of the optimization trace is binned into
    ``n_bins`` averages before the linear fit, which suppresses the MC noise
    in both coordinates.  Extrapolating the two geometries of an interaction
    energy with a *shared* slope is what the paper does; use
    :func:`extrapolate_shared_slope` for that.
    """

    energies = np.asarray(energies, dtype=float)
    gradients = np.asarray(grad_norm_sq, dtype=float)
    start = int(len(energies) * (1.0 - fit_fraction))
    energies, gradients = energies[start:], gradients[start:]
    if len(energies) < 4:
        raise ValueError("not enough optimization steps to extrapolate.")
    bins = np.array_split(np.arange(len(energies)), min(n_bins, len(energies)))
    x = np.array([gradients[index].mean() for index in bins])
    y = np.array([energies[index].mean() for index in bins])
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def extrapolate_shared_slope(
    traces: "list[tuple[np.ndarray, np.ndarray]]",
    *,
    fit_fraction: float = 0.5,
    n_bins: int = 20,
) -> tuple[list[float], float]:
    """Joint fit of several geometries with one common slope ``k`` (Sec. H).

    ``traces`` holds one ``(energies, grad_norm_sq)`` pair per geometry.
    Returns the list of extrapolated energies and the shared slope.
    """

    columns: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    n_traces = len(traces)
    for index, (energies, gradients) in enumerate(traces):
        energies = np.asarray(energies, dtype=float)
        gradients = np.asarray(gradients, dtype=float)
        start = int(len(energies) * (1.0 - fit_fraction))
        energies, gradients = energies[start:], gradients[start:]
        bins = np.array_split(np.arange(len(energies)), min(n_bins, len(energies)))
        x = np.array([gradients[b].mean() for b in bins])
        y = np.array([energies[b].mean() for b in bins])
        block = np.zeros((len(x), n_traces + 1))
        block[:, index] = 1.0
        block[:, -1] = x
        columns.append(block)
        targets.append(y)
    design = np.concatenate(columns, axis=0)
    observed = np.concatenate(targets, axis=0)
    solution, *_ = np.linalg.lstsq(design, observed, rcond=None)
    return [float(value) for value in solution[:-1]], float(solution[-1])


def optimize(
    system: MolecularSystem,
    config: FiREConfig | None = None,
    *,
    params=None,
    key: Array | None = None,
    callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """Run the variational optimization and return parameters plus history.

    Returns a dict with ``params``, ``history`` (:class:`TrainingHistory`),
    ``energy_hartree``, ``energy_error_hartree``, ``variance_hartree2``,
    ``energy_extrapolated_hartree`` and the resolved backend/step settings.
    """

    config = config or FiREConfig()
    if config.opt.steps < 1:
        raise ValueError("opt.steps must be positive.")
    key = jax.random.PRNGKey(config.seed) if key is None else key

    model, log_psi, sign_log_psi = make_log_psi(system, config.ansatz)
    key, init_key = jax.random.split(key)
    if params is None:
        params = init_params(model, init_key, system.n_electrons)

    backend = resolve_backend(config.laplacian_backend)
    local_energy = local_energy_fn(
        log_psi,
        system,
        backend=backend,
        sign_log_psi=sign_log_psi,
        ecp_quadrature=config.ecp_quadrature,
        ecp_cutoff=config.ecp_cutoff,
        ecp_max_log_ratio=config.ecp_max_log_ratio,
    )
    if system.has_ecp:
        # The nonlocal channels integrate over a randomly rotated quadrature
        # grid, so every walker needs its own key at every step.
        _batched = jax.vmap(local_energy, in_axes=(None, 0, 0))

        def batched_local_energy(params, positions, key):
            return _batched(params, positions, jax.random.split(key, positions.shape[0]))

    else:
        _batched = jax.vmap(local_energy, in_axes=(None, 0))

        def batched_local_energy(params, positions, key):
            del key
            return _batched(params, positions)
    init_sampler, sample_block = make_sampler(log_psi, system, config.mcmc)
    n_steps = default_n_steps(system, config.mcmc)

    key, sampler_key = jax.random.split(key)
    sampler_state = init_sampler(sampler_key, config.mcmc.batch_size)
    spring_state = init_spring_state(params)

    @jax.jit
    def burn_in(params, sampler_state):
        return sample_block(params, sampler_state, config.mcmc.n_burn_in)

    @jax.jit
    def training_step(
        params, spring_state: SpringState, sampler_state, learning_rate, step_key
    ):
        sampler_state = sample_block(params, sampler_state, n_steps)
        positions = sampler_state.positions
        energies = batched_local_energy(params, positions, step_key)
        clipped = clip_local_energies(
            energies, config.opt.clip_width, config.opt.clip_statistic
        )
        log_derivatives = per_sample_log_derivatives(log_psi, params, positions)
        delta, spring_state = spring_update(
            log_derivatives,
            clipped,
            spring_state,
            config.opt.damping,
            config.opt.decay,
            learning_rate=learning_rate,
            norm_constraint=config.opt.norm_constraint,
        )
        params = apply_update(params, delta, learning_rate)
        n_samples = energies.shape[0]
        statistics = {
            "energy": jnp.mean(energies),
            "energy_std": jnp.std(energies) / jnp.sqrt(n_samples),
            "variance": jnp.var(energies),
            "grad_norm_sq": jnp.sum(delta**2),
            "acceptance_local": sampler_state.acceptance_local,
            "acceptance_global": sampler_state.acceptance_global,
            "step_size": sampler_state.step_size,
        }
        return params, spring_state, sampler_state, statistics

    if config.mcmc.n_burn_in > 0:
        sampler_state = burn_in(params, sampler_state)

    history = TrainingHistory()
    for step in range(config.opt.steps):
        learning_rate = float(
            learning_rate_at(step, config.opt.learning_rate, config.opt.lr_decay_time)
        )
        # Only the ECP path consumes this stream; leaving it untouched
        # otherwise keeps the all-electron trajectory bit-for-bit identical.
        if system.has_ecp:
            key, step_key = jax.random.split(key)
        else:
            step_key = key
        params, spring_state, sampler_state, statistics = training_step(
            params, spring_state, sampler_state, learning_rate, step_key
        )
        record = {name: float(value) for name, value in statistics.items()}
        history.step.append(step)
        history.energy.append(record["energy"])
        history.energy_std.append(record["energy_std"])
        history.variance.append(record["variance"])
        history.grad_norm_sq.append(record["grad_norm_sq"])
        history.acceptance_local.append(record["acceptance_local"])
        history.acceptance_global.append(record["acceptance_global"])
        history.step_size.append(record["step_size"])
        history.learning_rate.append(learning_rate)
        if callback is not None and (step % max(config.log_every, 1) == 0):
            callback(step, record | {"learning_rate": learning_rate})

    window = min(config.opt.n_eval_steps, len(history.energy))
    tail = np.asarray(history.energy[-window:], dtype=float)
    energy = float(tail.mean())
    error = float(tail.std(ddof=1) / np.sqrt(len(tail))) if len(tail) > 1 else float("nan")
    variance = float(np.mean(history.variance[-window:]))

    try:
        extrapolated, slope = extrapolate_energy(history.energy, history.grad_norm_sq)
    except ValueError:
        extrapolated, slope = float("nan"), float("nan")

    if config.opt.eval_blocks > 0:
        evaluated = evaluate_frozen(
            params,
            sampler_state,
            key,
            sample_block=sample_block,
            batched_local_energy=batched_local_energy,
            n_steps=n_steps,
            n_blocks=config.opt.eval_blocks,
            block_steps=config.opt.eval_block_steps,
            burn_in_blocks=config.opt.eval_burn_in_blocks,
        )
        eval_energy = evaluated["energy_hartree"]
        eval_error = evaluated["energy_error_hartree"]
        eval_variance = evaluated["variance_hartree2"]
        eval_blocks = evaluated["blocks_used"]
        eval_tau = evaluated["autocorr_time"]
        eval_naive_error = evaluated["naive_error_hartree"]
        sampler_state = evaluated["sampler_state"]
    else:
        eval_energy = float("nan")
        eval_error = float("nan")
        eval_variance = float("nan")
        eval_blocks = 0
        eval_tau = float("nan")
        eval_naive_error = float("nan")

    return {
        "params": params,
        "history": history,
        "energy_hartree": energy,
        "energy_error_hartree": error,
        "variance_hartree2": variance,
        "energy_eval_hartree": eval_energy,
        "energy_eval_error_hartree": eval_error,
        "energy_eval_variance_hartree2": eval_variance,
        "eval_blocks_used": eval_blocks,
        "eval_autocorr_time": eval_tau,
        "energy_eval_naive_error_hartree": eval_naive_error,
        "energy_extrapolated_hartree": extrapolated,
        "extrapolation_slope": slope,
        "n_eval_steps": window,
        "laplacian_backend": backend,
        "mcmc_steps_per_block": n_steps,
        "sampler_state": sampler_state,
    }
