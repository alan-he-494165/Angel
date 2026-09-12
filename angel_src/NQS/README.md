# NetKet VMC NQS

This module provides a first second-quantized neural quantum state solver for
ANGEL. It consumes the active-space Hamiltonian produced by
`angel_src.electronic.fci.get_electronic_structure`, preserves the frozen-core
and active-space conventions, and trains a complex RBM with NetKet VMC.

The NQS is optimized without FCI energies or CI vectors. The objective is the
sampled expectation value of the frozen-core Hamiltonian. The default sampler
uses particle-number-preserving same-spin fermion hops, and stochastic
reconfiguration is enabled by default.

Example:

```python
from angel_src.NQS import NQSConfig, solve_geometry

result = solve_geometry(
    "Structure/C6H6_dimer/benzene_dimer_parallel_scan.xyz",
    "def2-svp",
    active_electrons=12,
    active_orbitals=12,
    config=NQSConfig(n_iterations=5000, seed=1234),
)
print(result.energy_hartree, result.energy_error_hartree)
```

The current implementation is intended for compact active spaces. For
benzene-scale 12e/12o calculations, increase samples and iterations only
after validating the Hamiltonian/operator convention on a smaller system.

## Real-space solver (`fire/`)

A second, independent NQS solver lives in `angel_src/NQS/fire/`: a
reimplementation of the FiRE ansatz of arXiv:2504.06087 (finite-range
embeddings, SPRING optimization, Metropolis sampling in real space). It is
also label-free — it minimizes the sampled expectation value of the
all-electron Coulomb Hamiltonian — but works in first quantization, so it
needs no active space and no orbital basis, at the price of a much more
expensive optimization. Use it where active-space truncation is the limiting
error, e.g. dispersion in π-stacked dimers; use the NetKet solver above for
compact active spaces. Nothing in `fire/` imports or modifies `netket_vmc.py`.

```python
from angel_src.NQS import solve_geometry_fire, fire_small_config

result = solve_geometry_fire("Structure/C6H6_dimer/benzene_dimer_parallel_scan.xyz", frame=0)
```

See `angel_src/NQS/fire/README.md` for the ansatz/module map, hyperparameters,
documented deviations from the paper and the CLI (`scripts/run_fire_vmc.py`,
`scripts/validate_fire.py`).
