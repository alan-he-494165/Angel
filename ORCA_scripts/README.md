# DLPNO-CCSD(T) benzene-dimer jobs

The original `benzene_dimer_parallel_*` subdirectories contain the
`DLPNO-CCSD(T)/aug-cc-pVTZ` calculations and their generated outputs. The
`cc-pVDZ/`, `cc-pVTZ/`, and `cc-pVQZ/` directories contain separate
cardinal-basis experiments for the same ten parallel C6H6 dimer geometries.
Each basis directory has one geometry subdirectory containing a matching ORCA
input and PBS submission script.

All inputs read the corresponding XYZ file from `Structure/C6H6_dimer/` and
run a closed-shell singlet `DLPNO-CCSD(T)` calculation with `TightSCF` and
`TightPNO`. The cardinal-basis jobs use the matching `cc-pVXZ/C` auxiliary
basis and request a 25-hour walltime.

Submit from the repository root or from an individual job directory, for example:

```bash
cd ORCA_scripts/benzene_dimer_parallel_3p60A
qsub benzene_dimer_parallel_3p60A.pbs
```

For a cardinal-basis job:

```bash
cd ORCA_scripts/cc-pVTZ/benzene_dimer_parallel_3p60A
qsub benzene_dimer_parallel_3p60A_tz.pbs
```

To inspect or submit all 30 cardinal-basis jobs:

```bash
cd ORCA_scripts
bash submit_all_cc_pvxz.sh --dry-run
bash submit_all_cc_pvxz.sh
```

The PBS scripts load `OpenMPI/4.1.6-GCC-13.2.0` and `ORCA/6.1.0-gompi-2023b-avx2`, then invoke `/sw-eb/software/ORCA/6.1.0-gompi-2023b-avx2/bin/orca`. The PBS resource requests are conservative defaults and may need adjustment for the local cluster.
