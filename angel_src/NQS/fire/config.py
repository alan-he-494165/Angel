"""Configuration for the FiRE real-space variational NQS solver.

Defaults follow Tab. S1 of arXiv:2504.06087 ("Accurate Ab-initio
Neural-network Solutions to Large-Scale Electronic Structure Problems").
Values that the paper reports for its GPU-scale production runs are kept as
defaults here; reduce ``batch_size`` and ``steps`` for CPU work.

All lengths are in bohr, all energies in hartree.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnsatzConfig:
    """Wave-function hyperparameters (Sec. 4.2 of arXiv:2504.06087)."""

    # Eq. 15: number of determinants in the linear combination.
    n_determinants: int = 4
    # Eq. 6: electron-electron embedding cutoff c.  The paper uses 5 a0 for
    # (non-)covalent interaction energies and 3 a0 for ionization and
    # singlet-triplet gaps.
    cutoff: float = 5.0
    # Eq. 20: electron-nucleus cutoff c_nuc.
    cutoff_nuclei: float = 20.0
    # Embedding width d (Eq. 18-24).
    hidden_dim: int = 256
    # Widths of the edge MLP inside the spatial filter Gamma (Eq. 20).
    edge_mlp_widths: tuple[int, ...] = (16, 8)
    # d_0, number of Gaussians in chi (Eq. 21).
    n_gaussians: int = 32
    # Order of the polynomial cutoff f_cut (Gasteiger et al. envelope).
    cutoff_poly_order: int = 6
    # Widths of the MLP applied to h^1 (Eq. 24).
    embedding_mlp_widths: tuple[int, ...] = (256,)
    # Widths of the Jastrow MLPs (Eq. 29 and Eq. 31).
    jastrow_mlp_widths: tuple[int, ...] = (256, 256)
    # Eq. 30: number of attention registers and their dimension.
    n_registers: int = 16
    register_dim: int = 16
    # Eq. 26: exponential envelopes per nucleus.
    n_envelopes: int = 8


@dataclass(frozen=True)
class MCMCConfig:
    """Metropolis-Hastings settings (Sec. 4.5 of arXiv:2504.06087)."""

    # Number of walkers, i.e. the batch size N_walker.
    batch_size: int = 4096
    # Decorrelation sweeps between optimization steps; the paper uses 2 n_el.
    # ``None`` selects that rule at run time.
    n_steps: int | None = None
    # Number of global single-electron jumps per sweep block (Eq. 46).
    n_global_moves: int = 20
    # Width of the nucleus-centred Gaussian mixture used for global moves.
    global_sigma: float = 2.0
    # Local proposal width, adapted on the fly towards ``target_acceptance``.
    step_size: float = 0.3
    target_acceptance: float = 0.5
    # Multiplicative adaptation rate of the local proposal width.
    step_size_adaptation: float = 0.05
    # Burn-in sweeps after initialization / before the first optimization step.
    n_burn_in: int = 100


@dataclass(frozen=True)
class OptConfig:
    """SPRING optimization settings (Eq. 12-14, Tab. S1)."""

    steps: int = 50_000
    # Learning rate schedule lr_0 / (1 + t / lr_decay_time).  Tab. S1 gives
    # lr_0 = 0.1; the sweep in README "Base learning rate" found 0.05 better
    # on the validation molecules (H2 +0.31 vs +0.57 mEh against FCI/CBS, no
    # multi-hartree transients on LiH) and worse on all-electron Be, where
    # neither rate is converged at the CPU step budget.  0.05 is the default.
    learning_rate: float = 0.05
    lr_decay_time: float = 10_000.0
    # SPRING damping lambda and momentum/decay eta.
    damping: float = 1e-3
    decay: float = 0.99
    # Fisher-norm constraint on the applied update, lr^2 d^T F d <= c.  Not in
    # Tab. S1; without it the eta = 0.99 momentum lets the parameters drift
    # without bound on all-electron systems (see README "Deviations").
    # Set to ``None`` to reproduce the paper's update exactly.
    norm_constraint: float | None = 1e-3
    # Local-energy clipping: clip at ``clip_width`` times the mean absolute
    # deviation around the median.
    clip_width: float = 5.0
    clip_statistic: str = "median"
    # Number of trailing steps averaged for the reported energy estimate.
    n_eval_steps: int = 500
    # Fixed-parameter evaluation is opt-in so the established training PRNG
    # stream and compilation path remain unchanged when reproducing old runs.
    eval_blocks: int = 0
    # Each recorded block advances this many optimization-block equivalents;
    # block means, rather than clipped gradient samples, enter the estimator.
    eval_block_steps: int = 10
    # Continuing the terminal chain preserves equilibration; these extra
    # blocks only let it settle after theta is frozen (diagnostic prior bug).
    eval_burn_in_blocks: int = 2


@dataclass(frozen=True)
class PretrainConfig:
    """Hartree-Fock orbital pretraining (Tab. S1).

    This supervises orbital *shapes* against a mean-field reference; no
    correlated energy label enters the objective at any point.
    """

    steps: int = 2_000
    # Tab. S1 quotes the schedule shape ``1/(1 + t/1000)`` without a base
    # value; taken literally (base 1.0) the Adam updates diverge on LiH, so
    # the base is set to 0.1 here.  See README "Deviations".
    learning_rate: float = 0.1
    lr_decay_time: float = 1_000.0
    # ``ccecp-ccpvdz`` in the paper (ECP calculations).  All-electron runs use
    # a conventional basis.
    basis: str = "cc-pvdz"
    batch_size: int | None = None
    # Fraction of pretraining walkers redrawn each step from nucleus-centred
    # Gaussians instead of the ansatz, so that orbital shapes are fitted over
    # a broad region while the ansatz density is still poor (see README for
    # why this stands in for the paper's HF-density sampling).
    gaussian_refresh_fraction: float = 0.5


@dataclass(frozen=True)
class FiREConfig:
    """Top-level configuration of a FiRE run."""

    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    opt: OptConfig = field(default_factory=OptConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    # Optional effective core potential, e.g. "ccecp".  ``None`` -- the
    # default -- is the all-electron treatment that the validation numbers in
    # README.md were produced with.  Setting this changes the Hamiltonian, so
    # energies are then only comparable against ECP-consistent references.
    ecp: str | None = None
    # Spherical quadrature for the nonlocal ECP channels: 6 (octahedral,
    # exact through l = 3) or 12 (icosahedral, exact through l = 5).  Ignored
    # without an ECP.
    ecp_quadrature: int = 12
    # Electron-nucleus distance beyond which the ECP radial functions are
    # treated as zero; they decay as exp(-a r^2) with a of order 1-20.
    ecp_cutoff: float = 10.0
    # Cap on |log Psi(r') - log Psi(r)| inside the nonlocal projector.  The
    # quadrature moves one electron onto a sphere, and when the starting
    # configuration is close to a node of Psi the ratio diverges, which is
    # what destabilises long optimizations.  The cap bounds a single
    # contribution at exp(20) ~ 5e8; it is a controllable bias, so report the
    # value used and check sensitivity before quoting a number near it.
    ecp_max_log_ratio: float = 20.0
    seed: int = 1234
    # Laplacian backend: "auto" uses folx when importable, else the exact
    # forward-mode fallback.  "folx" and "loop" force one of the two.
    laplacian_backend: str = "auto"
    # float64 throughout; float32 loses the cancellation in the local energy
    # of larger systems and is only useful for quick experiments.
    double_precision: bool = True
    log_every: int = 1
    log_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the configuration."""

        record = asdict(self)
        record["log_path"] = None if self.log_path is None else str(self.log_path)
        return record


def replace_config(config: FiREConfig, **overrides: Any) -> FiREConfig:
    """Return a copy of ``config`` with top-level fields replaced."""

    return dataclasses.replace(config, **overrides)


def small_config(**overrides: Any) -> FiREConfig:
    """A reduced configuration for CPU smoke tests and unit validation.

    This is not a converged production setting: widths, batch size and step
    count are all far below the paper's defaults.
    """

    config = FiREConfig(
        ansatz=AnsatzConfig(
            n_determinants=2,
            hidden_dim=32,
            edge_mlp_widths=(8, 8),
            n_gaussians=8,
            embedding_mlp_widths=(32,),
            jastrow_mlp_widths=(32, 32),
            n_registers=4,
            register_dim=8,
            n_envelopes=4,
        ),
        mcmc=MCMCConfig(batch_size=128, n_global_moves=2, n_burn_in=20),
        opt=OptConfig(steps=500, lr_decay_time=125.0, n_eval_steps=100),
        pretrain=PretrainConfig(steps=200),
    )
    if overrides:
        config = replace_config(config, **overrides)
    return config
