# DLPNO-CCSD(T)/cc-pVTZ benzene-dimer jobs

This directory contains one ORCA input and one PBS submission script for each parallel C6H6 dimer separation. All jobs use ORCA 6.1.0, 32 MPI processes, TightSCF, TightPNO, the matching cc-pVTZ/C auxiliary basis, and a 25-hour PBS walltime.

Each job reads its geometry from Structure/C6H6_dimer/. Submit a job from its geometry directory with qsub and the corresponding PBS filename.

