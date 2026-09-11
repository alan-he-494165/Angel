# DLPNO-CCSD(T)/aug-cc-pVTZ benzene-dimer jobs

Each subdirectory contains one ORCA input and its matching PBS submission script for one generated benzene-dimer geometry. The input reads the corresponding XYZ file from `Structure/C6H6_dimer/` and runs a closed-shell singlet `DLPNO-CCSD(T)` calculation with `aug-cc-pVTZ`, `TightSCF`, and `TightPNO`.

Submit from the repository root or from an individual job directory, for example:

```bash
cd ORCA_scripts/benzene_dimer_parallel_3p60A
qsub benzene_dimer_parallel_3p60A.pbs
```

The PBS scripts load `OpenMPI/4.1.6-GCC-13.2.0` and `ORCA/6.1.0-gompi-2023b-avx2`, then invoke `/sw-eb/software/ORCA/6.1.0-gompi-2023b-avx2/bin/orca`. The PBS resource requests are conservative defaults and may need adjustment for the local cluster.
