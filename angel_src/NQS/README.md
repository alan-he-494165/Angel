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
