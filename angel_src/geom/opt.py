"""B3LYP-D3BJ geometry optimization and vibrational analysis with PySCF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .get_geom import GeometryFrame, get_geom


@dataclass(frozen=True)
class OptimizationResult:
    """Optimized geometry, energy, frequencies, and analysis metadata."""

    optimized_geometry: GeometryFrame
    energy_hartree: float
    frequencies_cm1: np.ndarray
    imaginary_modes: np.ndarray
    hessian: np.ndarray
    basis: str
    xc: str
    dispersion: str
    charge: int
    spin: int
    gpu_used: bool


def _as_numpy(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _build_method(mol, xc: str, dispersion: str, charge: int, spin: int, use_gpu: bool):
    from pyscf import dft

    method = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    method.xc = xc
    if use_gpu:
        method = method.to_gpu()
    if dispersion.lower() not in ("none", "off", ""):
        try:
            from pyscf.dftd3 import dftd3
        except ImportError as error:
            raise ImportError(
                "B3LYP-D3BJ requires the optional pyscf-dftd3 package. "
                "Install it or call optimize_and_analyze(..., dispersion='none')."
            ) from error
        method = dftd3(method, xc=xc, version=dispersion.lower())
    return method


def _gpu_available() -> bool:
    try:
        import cupy
        import gpu4pyscf  # noqa: F401

        return cupy.cuda.runtime.getDeviceCount() > 0
    except (ImportError, ModuleNotFoundError):
        return False


def _frame_to_mol(frame: GeometryFrame, basis: str, charge: int, spin: int, verbose: int):
    from pyscf import gto

    return gto.M(
        atom=frame.as_pyscf_atom(),
        basis=basis,
        charge=charge,
        spin=spin,
        unit="Angstrom",
        symmetry=False,
        verbose=verbose,
    )


def geomopt_vib(
    geometry: GeometryFrame | str | Path,
    *,
    frame: int | Sequence[int] | None = 0,
    basis: str = "def2-TZVP",
    xc: str = "B3LYP",
    dispersion: str = "d3bj",
    charge: int = 0,
    spin: int = 0,
    maxsteps: int = 100,
    use_gpu: bool | None = None,
    allow_cpu_fallback: bool = True,
    verbose: int = 0,
) -> OptimizationResult | list[OptimizationResult]:
    """Optimize one XYZ frame and perform harmonic vibrational analysis.

    The default is B3LYP-D3BJ/def2-TZVP.  A trajectory path selects ``frame``
    (default: the first frame).  Pass ``frame=None`` to optimize every frame,
    or pass a sequence of indices to optimize only selected frames. Geometry
    optimization uses PySCF's geometric interface and therefore requires the
    optional ``geometric`` package.
    """
    try:
        from pyscf.geomopt.geometric_solver import optimize
    except ImportError as error:
        raise ImportError(
            "Geometry optimization requires the optional 'geometric' package. "
            "Install geometric before calling geomopt_vib()."
        ) from error
    from pyscf.hessian import thermo

    if isinstance(geometry, GeometryFrame):
        selected = [geometry]
    elif frame is None:
        selected = get_geom(geometry)
    elif isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        selected = get_geom(geometry, frames=frame)
    else:
        selected = [get_geom(geometry, frame=frame)]
    results = [
        _optimize_one(
            item,
            basis=basis,
            xc=xc,
            dispersion=dispersion,
            charge=charge,
            spin=spin,
            maxsteps=maxsteps,
            use_gpu=use_gpu,
            allow_cpu_fallback=allow_cpu_fallback,
            verbose=verbose,
        )
        for item in selected
    ]
    return results if frame is None or isinstance(frame, Sequence) else results[0]

def _optimize_one(
    selected: GeometryFrame,
    *,
    basis: str,
    xc: str,
    dispersion: str,
    charge: int,
    spin: int,
    maxsteps: int,
    use_gpu: bool | None,
    allow_cpu_fallback: bool,
    verbose: int,
) -> OptimizationResult:
    if spin < 0:
        raise ValueError("spin must be non-negative in PySCF's 2S convention.")
    mol = _frame_to_mol(selected, basis, charge, spin, verbose)
    gpu_available = _gpu_available()
    requested_gpu = gpu_available if use_gpu is None else bool(use_gpu) and gpu_available
    gpu_used = requested_gpu
    try:
        method = _build_method(mol, xc, dispersion, charge, spin, requested_gpu)
        optimized_mol = optimize(method, maxsteps=maxsteps)
    except Exception:
        if not (requested_gpu and allow_cpu_fallback):
            raise
        method = _build_method(mol, xc, dispersion, charge, spin, False)
        optimized_mol = optimize(method, maxsteps=maxsteps)
        gpu_used = False

    energy_method = _build_method(optimized_mol, xc, dispersion, charge, spin, gpu_used)
    energy = energy_method.kernel()
    if not energy_method.converged:
        raise RuntimeError("Post-optimization DFT calculation did not converge.")
    hessian = _as_numpy(energy_method.Hessian().kernel())
    analysis = thermo.harmonic_analysis(optimized_mol, hessian)
    frequencies = np.asarray(analysis["freq_wavenumber"]).real.astype(float)
    imaginary = frequencies < 0
    coordinates = np.asarray(optimized_mol.atom_coords(unit="Angstrom"), dtype=float)
    optimized = GeometryFrame(
        tuple(optimized_mol.atom_symbol(index) for index in range(optimized_mol.natm)),
        coordinates,
        comment=f"{xc}-{dispersion}/{basis} E={float(energy):.12f} Eh",
    )
    return OptimizationResult(
        optimized_geometry=optimized,
        energy_hartree=float(energy),
        frequencies_cm1=frequencies,
        imaginary_modes=imaginary,
        hessian=hessian,
        basis=basis,
        xc=xc,
        dispersion=dispersion,
        charge=charge,
        spin=spin,
        gpu_used=gpu_used,
    )
