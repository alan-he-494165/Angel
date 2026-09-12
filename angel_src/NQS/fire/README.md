# FiRE — real-space variational NQS solver

Reimplementation of the wave function, optimizer and sampler of

> M. Scherbela, N. Gao, P. Grohs, S. Günnemann,
> *Accurate Ab-initio Neural-network Solutions to Large-Scale Electronic
> Structure Problems*, [arXiv:2504.06087](https://arxiv.org/abs/2504.06087).

This is a **second, independent solver** inside `angel_src.NQS`. The
second-quantized NetKet solver (`angel_src/NQS/netket_vmc.py`) is untouched and
remains the default; nothing here imports it.

Like the NetKet path, FiRE is a **label-free variational NQS solver**: the only
objective is the sampled expectation value of the molecular Hamiltonian,

    E[θ] = ⟨Ψ_θ| H |Ψ_θ⟩ / ⟨Ψ_θ|Ψ_θ⟩ ,

estimated by Monte Carlo. No FCI, CCSD(T) or experimental energy is consumed at
any point of the optimization. Reference values appear only in validation
tables produced *after* a run. The mean-field pretraining step fits orbital
*shapes* to Hartree–Fock and is discarded once energy minimization starts; it
is an initialization, not a correlated label.

## Why a second solver

| | `netket_vmc` | `fire` |
|---|---|---|
| Representation | second quantization, active-space CI vector | first quantization, electrons in ℝ³ |
| Hamiltonian | frozen-core FCI Hamiltonian from PySCF integrals | all-electron Coulomb Hamiltonian, no basis set |
| Truncation error | active space + orbital basis | none (fixed-node-free VMC bias only) |
| Cost driver | number of active orbitals | number of electrons |
| Suitable size | ≲ 20 active orbitals | tens to ~100+ electrons (paper: 180 e) |
| Forces / geometries | not exposed | energies at arbitrary nuclear geometries |

For π-stacked and aromatic dimers the active-space route has to truncate
exactly the dispersion-carrying orbitals; the real-space route does not, which
is why it is worth having both.

## Module map

| Module | Paper reference | Contents |
|---|---|---|
| `config.py` | Tab. S1 | `AnsatzConfig`, `MCMCConfig`, `OptConfig`, `PretrainConfig`, `FiREConfig`; `small_config()` CPU preset |
| `system.py` | Eq. 9 | `MolecularSystem` (symbols, nuclei in bohr, charges, n_up/n_down), XYZ and array constructors, nuclear repulsion, provenance record |
| `ansatz.py` | Eq. 15–31 | pair features (16/17), `log(1+r)/r` rescaling (19), initial embedding (18), spatial filter Γ, χ, f_cut (20/21), message passing (22/23), readout MLP (24), orbitals + exponential envelopes (25/26), three-term Jastrow: analytic cusps (28), per-electron MLP (29), register cross-attention (30/31) |
| `hamiltonian.py` | Eq. 9–11 | local energy `E_L = -½ Σ(∇²lnΨ + |∇lnΨ|²) + V`; Laplacian via `folx` when installed, else an exact forward-mode loop |
| `mcmc.py` | Sec. 4.5, Eq. 46/47 | walker initialization by nuclear charge, adaptive all-electron Gaussian sweeps, global single-electron jumps from the charge-weighted Gaussian mixture with the Hastings correction |
| `optimizer.py` | Eq. 12–14 | SPRING update solved in sample space, learning-rate schedule, MAD clipping of local energies |
| `train.py` | Sec. 4.4, Eq. S16 | optimization loop, per-step diagnostics, energy estimate, gradient-norm energy extrapolation (single trace and shared-slope multi-geometry fit) |
| `pretrain.py` | Tab. S1 | Adam regression of the ansatz orbital matrices onto occupied HF orbitals evaluated by PySCF |
| `solver.py` | — | `run_fire_vmc`, `solve_geometry_fire`, `FiREResult` (`summary.json` + `history.csv`) |

Importing `angel_src.NQS.fire` pulls in only `config` and `system`; everything
that needs JAX is resolved lazily on attribute access.

## Usage

Python API:

```python
from angel_src.NQS.fire import FiREConfig, run_fire_vmc, system_from_xyz

system = system_from_xyz("data/c6h6_dimer_scan.xyz", frame=4)
result = run_fire_vmc(system, FiREConfig(), output_dir="runs/dimer_f4")
print(result.energy_hartree, result.energy_error_hartree)
```

Command line:

```bash
# CPU smoke test
python scripts/run_fire_vmc.py /tmp/h2.xyz --preset small --steps 300 \
    --pretrain-steps 50 --pretrain-basis sto-3g

# production-scale settings for one dimer frame (GPU strongly recommended)
python scripts/run_fire_vmc.py data/c6h6_dimer_scan.xyz --frame 4 \
    --steps 50000 --batch-size 2048 --out runs/dimer_f4
```

Validation against small all-electron systems:

```bash
python scripts/validate_fire.py --out out/fire_validation --steps 2000
```

`FiREResult.save()` writes `summary.json` (energies, config, provenance) and
`history.csv` (per-step energy, variance, preconditioned gradient norm,
acceptance ratios, step size, learning rate). Neither is committed — write run
output under `runs/` or `out/`.

## Hyperparameters

`FiREConfig()` reproduces the paper's Tab. S1 production values: 4
determinants, embedding width 256, e–e cutoff 5 a₀ (3 a₀ for ionization and
singlet–triplet gaps), e–n cutoff 20 a₀, 32 Gaussians in χ, Jastrow MLPs
256×256, 16 attention registers of width 16, 8 envelopes per nucleus, 4096
walkers, 2·n_el decorrelation sweeps plus 20 global moves per block, SPRING
with λ = 10⁻³ and η = 0.99, learning rate lr₀/(1 + t/10⁴), 50 000 optimization
steps, 2000 pretraining steps. These are GPU-scale settings. One value is
deliberately not the paper's: `OptConfig.learning_rate` ships as lr₀ = 0.05
rather than Tab. S1's 0.1, for the reason given under "Base learning rate"
below; pass `--learning-rate 0.1` to get the paper value back.

`small_config()` is the reduced CPU preset used by the tests and the validation
script (width 32, 2 determinants, 128 walkers, 500 steps). It is *not*
converged for anything beyond a few electrons.

## Ground truth: full CI at the basis-set limit

Full CI in a *fixed* finite basis is not a valid ground truth for this solver.
The ansatz is real-space and unconstrained by any orbital basis, so it can and
does fall below FCI/cc-pVDZ without being wrong.  The comparison target is
therefore full CI extrapolated to the complete basis set limit, built by
`scripts/fci_reference.py` as a composite of three separately converged pieces
(all-electron, clamped nuclei, non-relativistic):

| piece | method | bases |
|---|---|---|
| `E_HF(CBS)` | three-point exponential extrapolation (Feller 1992) | cc-pVDZ/TZ/QZ (+5Z for H2) |
| `E_corr,val(CBS)` | frozen-core full CI (CASCI over the complete virtual space), two-point `X^-3` extrapolation (Helgaker 1997) | cc-pVTZ, cc-pVQZ |
| `dE_core` | all-electron minus frozen-core full CI | cc-pCVTZ (cc-pVTZ on H) |

Every piece is full CI inside its orbital space — no truncated excitation
level anywhere.  The valence-only cc-pVXZ series alone misses roughly 40 mEh
of core correlation in LiH and Be, which is why `dE_core` is computed
separately in a core-valence basis.

| system | HF/CBS | valence corr. CBS | dE_core | **FCI/CBS** | literature NR | CBS - lit. |
|---|---|---|---|---|---|---|
| H2  | -1.133672  | -0.041039 | 0        | **-1.174711**  | -1.174476  | -0.24 mEh |
| LiH | -7.987298  | -0.037506 | -0.041822 | **-8.066625**  | -8.070548  | +3.92 mEh |
| Be  | -14.572988 | -0.046296 | -0.043712 | **-14.662996** | -14.667356 | +4.36 mEh |

The last column is the reference's own uncertainty: literature values
(Kolos-Wolniewicz for H2, Cencek-Rychlewski for LiH, Chakravorty et al. 1993
for Be) are *not* the comparison target, they are an independent check on the
extrapolation.  The composite is good to ~0.2 mEh for H2 and ~4 mEh for the
two systems that need a core correction, the residual being the cc-pCVTZ-level
`dE_core` and the two-point valence extrapolation.  Any solver error below
~4 mEh on LiH or Be is inside the noise of the reference itself.

## Validation (CPU preset, `small_config`)

`scripts/validate_fire.py`, 4 determinants, batch 256, 4000 optimization
steps, lr₀ = 0.1 (the table predates the default change to 0.05; the run
commands below pass it explicitly), single CPU device, double precision.  Nothing in the table enters
training: the solver minimizes the sampled Hamiltonian expectation value, and
references are read only in the summary step.

| system | e- | FiRE (Ha) | stat. err | E - FCI/CBS | ref. uncert. | corr. recovered | pretrain | wall (s) |
|---|---|---|---|---|---|---|---|---|
| H2 (1.4 a0)    | 2 | -1.174145  | 0.000047 | +0.57 mEh  | 0.24 mEh | 98.6 %  | none | 233 |
| LiH (3.015 a0) | 4 | -8.068573  | 0.000405 | -1.95 mEh  | 3.92 mEh | 102.5 % | none | 564 |
| Be atom        | 4 | -14.637434 | 0.001964 | +25.56 mEh | 4.36 mEh | 71.6 %  | 500 steps | 763 |

Read the first two rows as "converged to the FCI limit within the uncertainty
of the FCI limit itself": H2 sits 0.57 mEh above it, LiH 1.95 mEh below it,
both smaller than or comparable to the reference uncertainty.  The
correlation-recovered percentages are measured between HF/CBS and FCI/CBS, so
LiH exceeding 100 % says the composite reference is slightly too high, not
that the solver beat full CI.  For context, full CI in cc-pVDZ - the largest
basis in which all-electron FCI is routine for these systems - sits +11.3,
+51.9 and +45.6 mEh above the same limit.

All-electron Be is *not* converged at this preset: 25.6 mEh above the FCI
limit, far outside the reference uncertainty, with local-energy variance
0.39 Ha^2 against 2.4e-4 for H2.  The 1s cusp is only approximately
represented by the exponential envelopes, and Be also needs pretraining -
from a random start the run diverges.  Both are consequences of running
all-electron (deviation 7); the paper uses effective core potentials from Li
upwards.

### Base learning rate

`scripts/validate_fire.py --learning-rate` (and `scripts/run_fire_vmc.py
--learning-rate` / `--lr-decay-time`) override `OptConfig.learning_rate`.  The
same sweep at three values of `lr_0`, everything else identical (4000 steps,
decay time 1000, batch 256, 4 determinants):

| system | ref. unc. | lr_0 = 0.1 | lr_0 = 0.05 | lr_0 = 0.005 |
|---|---|---|---|---|
| H2  | +/- 0.24 mEh | +0.57 mEh  | **+0.31 mEh** | +3.34 mEh  |
| LiH | +/- 3.92 mEh | -1.95 mEh  | -4.47 mEh     | +6.11 mEh  |
| Be  | +/- 4.36 mEh | **+25.56 mEh** | +33.10 mEh | +89.50 mEh |

Correlation energy recovered: 98.6 / 99.2 / 91.9 % (H2), 102.5 / 105.6 /
92.3 % (LiH), 71.6 / 63.2 / 0.6 % (Be) at 0.1 / 0.05 / 0.005.

There is no single best value across the three systems.  `lr_0 = 0.05` is the
best setting for the two molecules - it halves the H2 error and gives the
smoothest traces, with none of the multi-hartree transients that `lr_0 = 0.1`
shows on LiH around step 600 - but it is worse on the all-electron atom
(+33.1 vs +25.6 mEh, sampling variance 0.52 vs 0.39 Eh^2), where the larger
rate is still descending at the final step (-6.7 mEh per 1000 steps, against
+3.6 for 0.05).  `lr_0 = 0.005` is uniformly worse, and not because it is
unstable: its traces are monotone, but H2 and LiH are still falling at -2.2
and -3.9 mEh per 1000 steps at step 4000 and Be has barely left its pretrained
mean-field start, i.e. the run is cut off mid-descent.

Both LiH numbers below zero sit inside the reference's own +/- 3.9 mEh
extrapolation uncertainty, so they are not evidence of a variational
violation.  `OptConfig.learning_rate` therefore defaults to **0.05**, the
better setting on the molecules; on all-electron systems at a short step
budget 0.1 converges further and is one flag away (`--learning-rate 0.1`).

### Is the reported energy really variational?

The `lr_0 = 0.05` LiH run reports -8.071093 E_h, which is 0.55 mE_h *below* the
essentially exact Cencek-Rychlewski value (-8.070548 E_h at the same geometry,
R = 3.015 a0) and 4.5 mE_h below the FCI/CBS composite.  A variational energy
cannot legitimately fall below the exact ground state, so this was checked
(`scripts/diagnose_variational.py`, results in `out/fire_variational_diagnosis.csv`
and `out/fire_lih_variational_check.png`):

| check | result |
| --- | --- |
| Hamiltonian | `local_energy_fn` on the exact H 1s wavefunction gives E_L = -0.500000000 with std 1.5e-16, i.e. the zero-variance test passes |
| estimator | `train.py` reports the mean of **unclipped** local energies; clipping enters the gradient only |
| geometry | LiH at R = 3.015 a0, the same geometry as the literature value |
| fixed-parameter evaluation | the *same* converged wavefunction, resampled with 1024 walkers x 400 blocks after 100 burn-in blocks: **-8.070404 +/- 0.000113 E_h**, i.e. +0.14 mE_h **above** the exact value |

So the wavefunction is above the exact energy and nothing is wrong with the
Hamiltonian; what is wrong is the *estimator used for reporting*.  The trailing
training-window mean is contaminated by the tails of the local-energy
distribution: for all-electron LiH the sampled E_L has skew -137, a minimum of
-29.1 E_h against a mean of -8.07, and the most negative 0.01 % of samples alone
move the mean by -0.24 mE_h.  Averaging 400 optimization steps of 256 walkers is
not enough to average that tail out, and the window standard error (0.195 mE_h)
understates the true uncertainty because the distribution is nowhere near
Gaussian.  H2, whose nuclei are both Z = 1, shows no such gap: window mean
-1.174400 vs fixed-parameter evaluation -1.174401 E_h.

Practical consequences:

* Sub-mE_h energies quoted from the validation table are only trustworthy for
  the two-electron system.  For LiH and Be, treat the tabulated numbers as
  accurate to roughly +/-1 mE_h, not to the quoted window standard error.
* Rankings across learning rates in the sweep above are affected at the same
  level; the 0.05-vs-0.1 gap on LiH (1.1 mE_h) is at the edge of what the
  training-window estimator can resolve, while the Be gap (7.5 mE_h) is not.
* The heavy tail is the expected signature of an electron-nucleus cusp that the
  ansatz satisfies only approximately at the Z = 3 Li core; this was not
  verified directly (it needs E_L resolved against electron-nucleus distance).

To get a defensible number, run the fixed-parameter evaluation:

```bash
python scripts/diagnose_variational.py --system LiH --steps 4000 \
    --learning-rate 0.05 --pretrain-steps 0 \
    --eval-batch 1024 --eval-blocks 400 --burn-in-blocks 100 --out out/diag
```

It retrains the system (the training scripts do not checkpoint parameters), then
evaluates at frozen parameters and reports a blocked error bar together with the
skewness and tail diagnostics of the local-energy distribution.  On the 10-core
laptop LiH + H2 together take about 25 min.

## How to run

Three stages, each a standalone script.  Stages 1 and 2 need the project
environment (JAX, PySCF); stage 3 needs only pandas and matplotlib.

**1. Build the ground truth** (once per geometry set; ~9 min for the three
validation systems):

```bash
python scripts/fci_reference.py --out out/fci_reference.csv
```

Writes `out/fci_reference.csv` (composite FCI/CBS per system) and
`out/fci_reference_components.csv` (the HF, valence-correlation and
core-correction pieces, with the basis series each was extrapolated over).

**2. Train.**  For the validation set, two invocations -- the molecules from a
random start, the all-electron atom from a pretrained start, since Be diverges
without pretraining (see "Hartree-Fock pretraining" below):

```bash
python scripts/validate_fire.py --out runs/main/a --only H2,LiH \
    --steps 4000 --batch-size 256 --determinants 4 --pretrain-steps 0 \
    --learning-rate 0.1 --reference-csv out/fci_reference.csv
python scripts/validate_fire.py --out runs/main/b --only Be \
    --steps 4000 --batch-size 256 --determinants 4 --pretrain-steps 500 \
    --learning-rate 0.1 --reference-csv out/fci_reference.csv
```

Each directory gets `fire_validation.csv` (one row per system, energy scored
against the FCI/CBS column and carrying the reference's own uncertainty) and
`fire_validation_traces.csv` (per-step energy, variance, gradient norm).
About 40 min per invocation on 10 CPU cores.

For a molecule of your own rather than the validation set:

```bash
python scripts/run_fire_vmc.py my_molecule.xyz --out runs/my_molecule \
    --steps 4000 --batch-size 256 --determinants 4 --learning-rate 0.1
```

**3. Plot.**  `scripts/plot_fire_validation.py` reads only the CSVs, so it
runs without JAX and re-renders from saved results at any time.  It accepts a
run directory holding `fire_validation.csv` directly, or one holding
sub-directories that each do -- so point it at the parent of the two
invocations above:

```bash
# convergence traces + final energies vs HF/CBS and FCI/cc-pVDZ
python scripts/plot_fire_validation.py --run runs/main --out out/fire_validation.png

# two settings overlaid (thick = --run, faint = --compare)
python scripts/plot_fire_validation.py --run runs/lr005 --compare runs/main \
    --labels '$lr_0=0.005$' '$lr_0=0.1$' --out out/fire_lr_comparison.png
```

## Effective core potentials (opt-in)

All-electron is the default and is unchanged: with `FiREConfig.ecp = None`
every code path, RNG draw and reported number is bit-for-bit what it was
before ECP support existed.  `tests/test_no_ecp_regression.py` pins that
against golden values (H2 -1.156094830, LiH -7.971630217, Be -14.534265878
E_h, plus the hydrogen zero-variance check) to 1e-8 E_h.

Turn the option on with a family name:

```python
config = replace_config(small_config(), ecp="ccecp")
system = system_from_arrays(["Li", "H"], coords, units="bohr", ecp="ccecp")
```

`run_fire_vmc` reconciles the two: if the config names a family and the
system carries none, the system is rebuilt with it; if both are set and
disagree, the run raises rather than silently solving a different
Hamiltonian.

What the option changes:

| piece | all-electron | with `ecp="ccecp"` |
| --- | --- | --- |
| nuclear charge in `V_ne` and the envelope init | `Z` | `Z_eff = Z - n_core` |
| electrons sampled | all | valence only (LiH: 4 -> 2) |
| extra Hamiltonian term | none | local radial channel + nonlocal projector on a randomly rotated Lebedev-style grid (`ecp_quadrature`, default 12 points) |
| pretraining orbitals | `pretrain.basis` | the ECP-consistent basis (`cc-pvdz` -> `ccecp-cc-pvdz`) on a PySCF molecule carrying the same ECP |
| reference table | `out/fci_reference.csv` | `out/fci_reference_ccecp.csv` |

### Validation

1. **Operator vs PySCF.** `scripts/validate_ecp_operator.py` evaluates the
   VMC energy of a *single Slater determinant* built from the ECP basis and
   compares it to RHF, and the ECP expectation value to a density-matrix
   trace over PySCF's own `ECPscalar` integrals.  Both agree to <=0.3 mE_h
   on H2, LiH and Be (`out/ecp_operator_check.csv`).
2. **References.** Under an ECP nothing is frozen in the CI (the core is
   already gone) and the core-correlation correction does not apply.  Total
   energies belong to a different Hamiltonian, so the literature column is
   left empty on purpose.  The cross-check that does carry weight is that
   valence correlation energy at the basis-set limit agrees between the two
   Hamiltonians: 0.03 (H2), 0.24 (LiH), 1.97 mE_h (Be).
3. **End to end.** `scripts/validate_fire.py --ecp ccecp` keeps the context
   basis, pretraining basis and reference table on the same Hamiltonian and
   records `core_treatment` in the output rows.

### Known limitation: optimization robustness, not the estimator

The estimator is correct (point 1 above), but the nonlocal projector
evaluates `Psi(r')/Psi(r)` on the quadrature sphere, and a neural ansatz
early in training can place a spurious node next to the sampled point.  The
resulting heavy local-energy tail makes the *optimizer* the fragile part.

`scripts/ecp_stability_study.py` measures this over systems x seed x
clipping width (`out/ecp_stability.csv`, 1200 steps, 4 determinants,
`lr_decay_time = steps/8`):

| treatment | `clip_width` | diverged | finite-run error vs FCI/CBS |
| --- | --- | --- | --- |
| all-electron | 2 | 0 / 4 | 14-70 mE_h |
| all-electron | 5 | 1 / 4 | 37-92 mE_h |
| ccECP | 2 | 1 / 4 | 31-94 mE_h |
| ccECP | 5 | 1 / 4 | 25-257 mE_h |

Two things follow, and the first is the one that matters:

* **Divergence is not ECP-specific.**  All-electron LiH at `clip_width = 5`,
  seed 7 blew up to -1e21 E_h.  It is a clipping/step-size failure of this
  aggressive CPU preset (`steps/8` decay) that the ECP path is *more* prone
  to, not a defect the ECP path introduces.
* **Use `clip_width = 2` for ECP runs** (the `OptConfig` default of 5 is
  tuned for all-electron).  `clip_width = 1` over-clips and biases badly
  (LiH: -0.083 vs -0.788 E_h).

Two further knobs were characterised and are not the problem:

* `ecp_max_log_ratio` (default 20, was 60) caps the log wavefunction ratio
  inside the projector.  20 and 60 give identical energies -- the cap never
  binds -- while 10 biases visibly.  It is a controllable bias: report it.
* Stripping the nonlocal channels collapses Be to -3.60 E_h against a
  reference of -1.010, so the projector is essential and functioning.

Practical recipe for an ECP production run: `clip_width = 2`, at least ~1000
pretraining steps (shorter pretraining leaves a polluted start -- step-0
variance 86 vs 0.056), and `lr_decay_time` no slower than `steps/8` (slower
decay is stable but still converging when the step budget runs out).

## Deviations from the paper

1. **Dense masked tensors instead of sparse kernels.** Sec. I of the paper
   keeps the finite-range structure sparse by computing the required tensor
   shapes on the fly, which is what makes the cost scale as O(N_el) rather than
   O(N_el²). Here the cutoffs are applied as masks on dense pair tensors: the
   *values* are identical, the *asymptotic cost* is not. Expect the paper's
   accuracy but quadratic scaling; this is the main thing to replace before
   running systems with many tens of electrons.
2. **Pretraining learning rate.** Tab. S1 gives the schedule shape
   `1/(1 + t/1000)` without a base value. Taken literally (base 1.0) the Adam
   updates diverge on LiH; the base is set to 0.1 here.
3. **Base learning rate.** Tab. S1 gives lr₀ = 0.1. The sweep above found
   0.05 better on both validation molecules and no worse anywhere the run is
   converged, so `OptConfig.learning_rate` defaults to 0.05; the paper value
   is recovered with `--learning-rate 0.1`.
4. **Pretraining walkers.** The paper draws pretraining samples from the HF
   density. Evaluating Gaussian basis functions inside JAX is not implemented,
   so walkers come from the current ansatz via the same MCMC kernel, with
   `pretrain.gaussian_refresh_fraction` (default 0.5) of them redrawn each step
   from nucleus-centred Gaussians so that orbital shapes are fitted over a
   broad region while the ansatz density is still poor. PySCF evaluates the
   target orbitals on the host each step. 500 pretraining steps at batch 256
   bring the H2 orbital regression loss to ~1e-5 and the pretrained ansatz to
   -1.140 Ha (below HF/cc-pVDZ, -1.1287 Ha); 100 steps is not enough and
   leaves energy minimization in a worse basin than a random start.

   **Open issue.** Fitting every determinant to the same HF reference
   collapses the expansion: the relative spread of the orbital matrices
   across determinants drops from 1.06 (random init) to 0.007 after
   pretraining. Energy minimization does break the symmetry again (back to
   0.86 after 1200 steps), but on H2 a pretrained start plateaus near
   -1.16 Ha with variance ~0.05 while a random start reaches -1.1744 Ha with
   variance 5e-4 in 400 steps. Until this is understood, pretraining is off
   by default in `scripts/validate_fire.py`; it is still the right choice at
   the scales where random initialization does not find the correct node
   structure, but budget thousands of pretraining and optimization steps.
5. **SPRING normalization.** Eq. 13 states the energy gradient only up to
   proportionality; the constant factor 2 is absorbed into the learning rate,
   as in the reference SPRING formulation.
6. **Fisher-norm constraint on the SPRING step.** Tab. S1 lists no norm
   constraint. With the tabulated momentum (eta = 0.99) the update is
   geometrically amplified along a persistent direction, and on all-electron
   Be the parameter norm grew without bound until the energy went non-finite.
   `OptConfig.norm_constraint` (default 1e-3) rescales the applied update so
   that `lr^2 delta^T (O^T O) delta <= c`, the constraint used in the SPRING
   and KFAC reference implementations. Set it to `None` to recover the
   paper's update exactly.
7. **Envelope initialization.** `envelope_decays` are initialized at the
   nuclear charge times a geometric spread over `[0.5 Z, 2 Z]`, not at 1.
   With ECPs (the paper's setting) the core is removed and the initial scale
   hardly matters; all-electron Li and Be start with a badly wrong core
   otherwise, which shows up as heavy-tailed local energies and
   non-variational energy estimates.
8. **All-electron only.** The paper uses ccECP effective core potentials for
   heavier elements. No ECP support here — every electron is sampled
   explicitly, which is affordable for H/C/N/O systems and expensive beyond.
9. **Single device.** No multi-GPU or multi-host distribution, no gradient
   accumulation over walker chunks. Peak memory is set by
   `batch_size × n_params` in the SPRING log-derivative matrix.
10. **Error bars.** The reported `energy_error_hartree` is the standard error of
   the trailing step-mean energies; it does not correct for autocorrelation
   between consecutive optimization steps and will be optimistic.
11. **Laplacian.** `folx` (forward Laplacian) is used when importable and is
   what the paper relies on for speed; the fallback is an exact forward-mode
   loop over the 3·N_el directions, correct but O(N_el) times more expensive.

## Using it inside the ANGEL workflow

For a distance scan of an aromatic dimer, the quantity that matters is the
*relative* energy along the scan, not the absolute total energy:

- keep `seed`, `steps`, `batch_size` and the ansatz size fixed across the whole
  scan, so that the residual variational bias is smooth in the scan coordinate;
- extrapolate with `extrapolate_shared_slope`, which is the paper's Sec. H
  procedure (one common slope `k` in `E_t = E_∞ + k|g_t|²` across geometries),
  rather than comparing energies at a fixed step count;
- feed relative energies within a consistent series to the MLIP fine-tune,
  never absolute totals from runs with different settings.

`FiREResult.history` carries the traces needed for that fit
(`energy_hartree`, `grad_norm_sq`).
