# AGENTS.md

## Project Identity

This repository is for:

> Sparse NQS-Supervised Fine-Tuning of Machine-Learned Interatomic Potentials Using Frozen-Core FCI Hamiltonians

The project is a research protocol for adapting pretrained machine-learned interatomic potentials (MLIPs) to noncovalent interaction (NCI) landscapes, especially pi-pi stacking and related weak interactions.

The intended contribution is a fine-tuning protocol, not a claim that the project introduces a universally superior MLIP.

## Core Technical Framing

The key method chain is:

```text
molecular geometry
-> frozen-core FCI Hamiltonian construction
-> label-free variational NQS solving
-> NQS-derived correction signal
-> sparse fine-tuning of a pretrained MLIP
-> NCI landscape evaluation
```

The NQS is not trained from FCI reference labels. In this project, "FCI" refers to the Hamiltonian representation and configuration space used to define the electronic problem. The NQS is trained variationally by minimizing the energy expectation value of the frozen-core FCI Hamiltonian.

Use wording such as:

- variational NQS solver
- frozen-core FCI Hamiltonian
- label-free NQS training
- NQS-derived correction signal
- sparse NQS-supervised MLIP fine-tuning
- neural-wavefunction teacher for MLIP adaptation

Avoid wording such as:

- FCI-supervised NQS
- NQS trained on FCI labels
- exact FCI labels for NQS
- FCI-calibrated NQS, unless an explicit calibration experiment is added
- NQS is always more accurate than CCSD(T)
- NQS is always cheaper than conventional quantum chemistry
- this is a universally better MLIP

## Scientific Scope

Primary focus:

- pi-pi stacking
- aromatic dimers
- substituted benzene dimers
- heteroaromatic stacking
- DNA base-pair stacking motifs

Secondary focus:

- pi-H interactions
- mixed dispersion/electrostatic NCIs
- S66x8 dispersion and mixed subsets

Do not expand the first version into all NCIs, full biomolecular simulation, periodic organic crystals, or a general-purpose force field unless the user explicitly requests a scope change.

## Baseline Assumptions

The base MLIP should be treated as an existing pretrained model to be adapted, not replaced.

Preferred first baseline:

- MACE-OFF

Optional comparisons:

- ANI-2x
- ANI-1ccx
- AIMNet2

The first fine-tuning implementation should favor a frozen-base delta-correction head:

```text
E_final = E_base + Delta_E_NQS
```

This is preferred because it reduces catastrophic forgetting and makes the protocol easier to evaluate under sparse supervision.

## Dataset and Benchmark Priorities

Use these resources as first choices:

- S66 / S66x8 for small balanced NCI curves
- DES370K / DES15K for dimer interaction subsets
- SPICE for general organic pretraining or baseline context
- L7 for larger pi-stacking transfer tests
- S12L / S30L only as later large-system transfer tests
- NCIAtlas for broader NCI category evaluation

Evaluation should emphasize NCI landscape properties, not only pointwise MAE:

- interaction energy MAE/RMSE
- binding energy error
- equilibrium distance error
- motif ranking accuracy
- curve-shape error
- sliding/rotation landscape error
- long-range tail behavior
- compressed-region repulsion behavior
- catastrophic forgetting on general organic chemistry

## NQS Solver Guidance

Candidate tools or families:

- DeepQMC
- FermiNet-style real-space neural wavefunctions
- PauliNet-style neural wavefunctions
- NAQS-style second-quantized neural wavefunctions

For each selected geometry, keep the following consistent:

- frozen-core convention
- orbital basis
- active space, if used
- dimer and monomer energy definitions
- counterpoise or basis-set-superposition treatment, if applicable
- stochastic optimization settings
- reported uncertainty or variance

For weak interaction energies, be careful with dimer-minus-monomer cancellation. Prefer reporting interaction curves and relative landscape features before making broad claims about absolute binding energies.

## Experimental Controls

Always preserve the distinction between sampling and supervision.

Useful comparisons:

- base MLIP without fine-tuning
- MD trajectory fine-tuning with low-level labels
- DFT-level fine-tuning on the same configurations
- NQS-supervised fine-tuning on the same configurations
- random NQS selection vs active NQS selection
- frozen-base delta head vs fuller fine-tuning

The strongest protocol test is:

> same configurations, different supervision signals.

This isolates whether improvement comes from NQS-derived variational wavefunction information rather than from better sampling alone.

## Claim Discipline

Preferred final claim:

> Sparse variational NQS supervision provides an affordable route for adapting pretrained MLIPs to weak-interaction landscapes by solving frozen-core FCI Hamiltonians only on configurations where DFT-trained potentials are most likely to inherit chemically meaningful noncovalent-interaction errors.

Keep claims narrow and testable:

- "more affordable than exhaustive high-level relabeling" is acceptable if a solving-budget comparison is shown.
- "more reliable than low-level trajectory fine-tuning" is acceptable only if the same configurations are compared under different supervision signals.
- "NQS improves accuracy" must be tied to specific benchmarks and metrics.

Do not claim:

- universal superiority over all MLIPs
- universal superiority over CCSD(T), SAPT, or DLPNO-CCSD(T)
- universal generalization from pi-stacking to all NCIs
- that MD trajectory fine-tuning is obsolete

## Repository Work Style

Before editing files:

1. Read `project.md`.
2. Check whether an `AGENTS.md` instruction already covers the change.
3. Keep terminology aligned with the variational frozen-core FCI Hamiltonian framing.

When adding code later, prefer a clear modular layout:

```text
data/          benchmark metadata or download scripts
configs/       experiment configs
src/           reusable source code
scripts/       runnable experiment entry points
notebooks/     exploratory analysis only
results/       generated summaries, not raw massive outputs
docs/          paper notes and figures
```

Do not commit large datasets, model checkpoints, raw trajectories, or generated heavy outputs unless the user explicitly asks for that repository policy.

## Immediate Research Tasks

Recommended next steps:

1. Assemble a small pi-stacking benchmark subset.
2. Run base MACE-OFF evaluation on NCI curves.
3. Select 25-100 NCI-critical configurations for the first variational NQS solving test.
4. Construct frozen-core FCI Hamiltonians for the smallest selected systems.
5. Validate NQS-derived interaction curves against trusted references.
6. Implement a frozen-base delta-correction head for sparse MLIP fine-tuning.

