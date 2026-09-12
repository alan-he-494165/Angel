"""Top-level entry points for the FiRE real-space variational NQS solver.

``run_fire_vmc`` takes a :class:`~angel_src.NQS.fire.system.MolecularSystem`
and returns a :class:`FiREResult`; ``solve_geometry_fire`` is the
geometry-file convenience wrapper that mirrors
:func:`angel_src.NQS.netket_vmc.solve_geometry` for the real-space solver.

Both are label-free: the only objective is the sampled expectation value of
the all-electron molecular Hamiltonian.  No FCI, CCSD(T) or experimental
energy enters the optimization; reference values are used only when the
caller compares results afterwards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

from .config import FiREConfig
from .system import MolecularSystem, system_from_xyz

__all__ = ["FiREResult", "run_fire_vmc", "solve_geometry_fire"]


@dataclass(frozen=True)
class FiREResult:
    """Summary of one FiRE optimization at a fixed nuclear geometry."""

    energy_hartree: float
    energy_error_hartree: float | None
    energy_variance_hartree2: float | None
    energy_extrapolated_hartree: float | None
    extrapolation_slope: float | None
    hf_energy_hartree: float | None
    nuclear_repulsion_hartree: float
    n_steps: int
    n_eval_steps: int
    seed: int
    laplacian_backend: str
    mcmc_steps_per_block: int
    acceptance_local: float
    acceptance_global: float
    system: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict, repr=False)
    energy_eval_hartree: float = float("nan")
    energy_eval_error_hartree: float = float("nan")
    energy_eval_variance_hartree2: float = float("nan")
    eval_blocks_used: int = 0
    eval_autocorr_time: float = float("nan")
    params: Any = field(default=None, repr=False, compare=False)

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        # Raw parameters have their own lossless NPZ checkpoint and must not
        # be deep-copied or coerced through the JSON summary path.
        record = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "params"
        }
        if not include_history:
            record.pop("history")
        return record

    def save(self, directory: str | Path) -> dict[str, Path]:
        """Write ``summary.json`` and ``history.csv`` into ``directory``."""

        target = Path(directory).expanduser()
        target.mkdir(parents=True, exist_ok=True)
        summary_path = target / "summary.json"
        summary_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

        history_path = target / "history.csv"
        if self.history:
            columns = list(self.history)
            rows = zip(*(self.history[name] for name in columns))
            lines = [",".join(columns)]
            lines += [",".join(f"{value!r}" for value in row) for row in rows]
            history_path.write_text("\n".join(lines) + "\n")
        return {"summary": summary_path, "history": history_path}


def _config_record(config: FiREConfig) -> dict[str, Any]:
    record = asdict(config)
    return json.loads(json.dumps(record, default=str))


def apply_config_ecp(system: MolecularSystem, config: FiREConfig) -> MolecularSystem:
    """Return the system with ``config.ecp`` applied, if it asks for one.

    ``config.ecp`` defaults to ``None``, in which case the system is returned
    untouched and the run is all-electron.  A system that already carries an
    ECP is accepted only if it is the same family, so a mismatch between the
    geometry definition and the CLI flag fails loudly instead of silently
    solving a different Hamiltonian.
    """

    if config.ecp is None:
        return system
    if system.has_ecp:
        if system.ecp_name != config.ecp:
            raise ValueError(
                f"system carries ECP {system.ecp_name!r} but config asks for "
                f"{config.ecp!r}."
            )
        return system

    from .system import system_from_arrays

    return system_from_arrays(
        list(system.symbols),
        system.nuclei,
        charge=system.charge,
        spin=system.spin,
        units="bohr",
        name=system.name,
        ecp=config.ecp,
    )


def run_fire_vmc(
    system: MolecularSystem,
    config: FiREConfig | None = None,
    *,
    pretrain: bool = True,
    output_dir: str | Path | None = None,
    callback: Callable[[int, dict], None] | None = None,
) -> FiREResult:
    """Pretrain (optionally) and variationally optimize the FiRE ansatz."""

    import jax

    config = config or FiREConfig()
    system = apply_config_ecp(system, config)
    if config.double_precision:
        jax.config.update("jax_enable_x64", True)

    from .pretrain import pretrain_to_hartree_fock
    from .train import optimize

    key = jax.random.PRNGKey(config.seed)
    key, pretrain_key, train_key = jax.random.split(key, 3)

    params = None
    hf_energy: float | None = None
    if pretrain and config.pretrain.steps > 0:
        pretrained = pretrain_to_hartree_fock(
            system, config, key=pretrain_key, callback=callback
        )
        params = pretrained["params"]
        hf_energy = pretrained["hf_energy_hartree"]

    result = optimize(system, config, params=params, key=train_key, callback=callback)
    history = result["history"].as_dict()

    fire_result = FiREResult(
        energy_hartree=result["energy_hartree"],
        energy_error_hartree=result["energy_error_hartree"],
        energy_variance_hartree2=result["variance_hartree2"],
        energy_extrapolated_hartree=result["energy_extrapolated_hartree"],
        extrapolation_slope=result["extrapolation_slope"],
        hf_energy_hartree=hf_energy,
        nuclear_repulsion_hartree=system.nuclear_repulsion,
        n_steps=config.opt.steps,
        n_eval_steps=result["n_eval_steps"],
        seed=config.seed,
        laplacian_backend=result["laplacian_backend"],
        mcmc_steps_per_block=result["mcmc_steps_per_block"],
        acceptance_local=float(history["acceptance_local"][-1]),
        acceptance_global=float(history["acceptance_global"][-1]),
        system=system.provenance(),
        config=_config_record(config),
        history=history,
        energy_eval_hartree=result["energy_eval_hartree"],
        energy_eval_error_hartree=result["energy_eval_error_hartree"],
        energy_eval_variance_hartree2=result["energy_eval_variance_hartree2"],
        eval_blocks_used=result["eval_blocks_used"],
        eval_autocorr_time=result["eval_autocorr_time"],
        params=result["params"],
    )
    if output_dir is not None:
        fire_result.save(output_dir)
    return fire_result


def solve_geometry_fire(
    geometry: str | Path,
    *,
    frame: int = 0,
    charge: int = 0,
    spin: int = 0,
    config: FiREConfig | None = None,
    pretrain: bool = True,
    output_dir: str | Path | None = None,
    callback: Callable[[int, dict], None] | None = None,
) -> FiREResult:
    """Solve one frame of an XYZ file with the real-space variational NQS.

    Unlike :func:`angel_src.NQS.netket_vmc.solve_geometry`, no active space or
    orbital basis is selected: the ansatz acts on all electrons in real space,
    so ``active_electrons``/``active_orbitals`` have no counterpart here.  The
    basis argument of the mean-field pretraining step affects only the
    orbital initialization, not the Hamiltonian being minimized.
    """

    system = system_from_xyz(geometry, frame=frame, charge=charge, spin=spin)
    return run_fire_vmc(
        system,
        config,
        pretrain=pretrain,
        output_dir=output_dir,
        callback=callback,
    )
