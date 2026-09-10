"""PySCF FCI-space construction and energy evaluation.

The public functions in this module are Python APIs; they do not invoke shell
commands or subprocesses.  The mean-field calculation can use GPU4PySCF when
both GPU4PySCF and a CUDA device are available.  PySCF's conventional FCI
solver remains the reference solver and is run from Python after the SCF
orbitals and integrals have been prepared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import pickle
import re
import warnings

import numpy as np


SUPPORTED_BASIS = ("sto-3g", "def2-svp", "def2-tzvp")
_GPU4PYSCF_AVAILABLE: bool | None = None


@dataclass(frozen=True)
class FCISpace:
    """The alpha/beta determinant space for a spatial-orbital FCI problem."""

    norb: int
    nalpha: int
    nbeta: int
    alpha_strings: np.ndarray
    beta_strings: np.ndarray

    @property
    def n_configurations(self) -> int:
        """Number of alpha/beta determinant pairs in the CI vector."""
        return int(self.alpha_strings.size * self.beta_strings.size)

    def iter_configurations(self) -> Iterator[tuple[int, int]]:
        """Yield ``(alpha_bitstring, beta_bitstring)`` determinant pairs."""
        for alpha in self.alpha_strings:
            for beta in self.beta_strings:
                yield int(alpha), int(beta)

    def occupied_orbitals(self, determinant: int) -> tuple[int, ...]:
        """Return spatial-orbital indices occupied by a determinant bitstring."""
        return tuple(index for index in range(self.norb) if determinant & (1 << index))


@dataclass(frozen=True)
class FCIResult:
    """Results and reproducibility metadata from an FCI calculation."""

    total_energy_hartree: float
    electronic_energy_hartree: float
    nuclear_repulsion_hartree: float
    ci_vector: np.ndarray
    space: FCISpace
    basis: str
    charge: int
    spin: int
    frozen_core: int
    gpu_available: bool
    gpu_used: bool
    converged: bool


@dataclass(frozen=True)
class ElectronicStructure:
    """Cached active-space Hamiltonian and optional FCI wavefunction."""

    hamiltonian: dict
    wavefunction: dict | None
    energy_hartree: float | None
    hamiltonian_path: Path
    wavefunction_path: Path | None


def _normalise_basis(basis: str) -> str:
    value = basis.strip().lower()
    aliases = {"sto3g": "sto-3g", "def2svp": "def2-svp", "def2tzvp": "def2-tzvp"}
    value = aliases.get(value, value)
    if value not in SUPPORTED_BASIS:
        choices = ", ".join(SUPPORTED_BASIS)
        raise ValueError(f"Unsupported basis {basis!r}; choose one of: {choices}.")
    return value


def _read_xyz(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path} is not a valid XYZ file.")
    try:
        natoms = int(lines[0].strip())
    except ValueError as error:
        raise ValueError(f"{path}: first line must be the atom count.") from error
    atom_lines = [line.strip() for line in lines[2 : 2 + natoms] if line.strip()]
    if len(atom_lines) != natoms:
        raise ValueError(f"{path}: expected {natoms} atom lines, found {len(atom_lines)}.")
    return "\n".join(atom_lines)


def build_molecule(
    geometry: str | Path | Sequence[Sequence[object]],
    basis: str,
    *,
    charge: int = 0,
    spin: int = 0,
    unit: str = "Angstrom",
    verbose: int = 0,
):
    """Build and return a PySCF molecule from geometry and a supported basis.

    ``geometry`` may be a PySCF atom string, an XYZ path, or a sequence such
    as ``[("H", (0, 0, 0)), ("H", (0, 0, 0.74))]``.  ``spin`` is PySCF's
    ``2S`` convention.
    """
    from pyscf import gto

    if isinstance(geometry, Path):
        geometry = _read_xyz(geometry)
    elif isinstance(geometry, str) and "\n" not in geometry and Path(geometry).is_file():
        geometry = _read_xyz(Path(geometry))
    if spin < 0:
        raise ValueError("spin must be a non-negative integer in PySCF's 2S convention.")
    return gto.M(
        atom=geometry,
        basis=_normalise_basis(basis),
        charge=charge,
        spin=spin,
        unit=unit,
        symmetry=False,
        verbose=verbose,
    )


def gpu4pyscf_available() -> bool:
    """Return whether GPU4PySCF and at least one CUDA device are available."""
    global _GPU4PYSCF_AVAILABLE
    if _GPU4PYSCF_AVAILABLE is not None:
        return _GPU4PYSCF_AVAILABLE

    try:
        import cupy
        import gpu4pyscf  # noqa: F401  (registers GPU PySCF methods)

        _GPU4PYSCF_AVAILABLE = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        # GPU4PySCF is optional. CUDA ABI or driver failures degrade to the
        # CPU PySCF path without interrupting a geometry scan.
        _GPU4PYSCF_AVAILABLE = False
    return _GPU4PYSCF_AVAILABLE


def _as_numpy(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


def _closed_shell_check(spin: int, active_electrons: int | None = None) -> None:
    if spin != 0:
        raise NotImplementedError("Only closed-shell systems (spin=0) are supported.")
    if active_electrons is not None and active_electrons % 2:
        raise ValueError("active_electrons must be even for a closed-shell active space.")


def _safe_molecule_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
    return value.strip(".-") or "molecule"


def _default_molecule_name(geometry: str | Path | Sequence[Sequence[object]]) -> str:
    if isinstance(geometry, Path):
        return geometry.stem
    if isinstance(geometry, str) and "\n" not in geometry and Path(geometry).is_file():
        return Path(geometry).stem
    return "molecule"


def _active_space_parameters(
    mol,
    nmo: int,
    active_electrons: int | None,
    active_orbitals: int | None,
    frozen_core: int,
) -> tuple[int, int, int]:
    """Return ``(ncore, ncas, nelecas)`` for a closed-shell CASCI space."""
    _closed_shell_check(mol.spin, active_electrons)
    if frozen_core < 0:
        raise ValueError("frozen_core must be non-negative.")
    if active_orbitals is not None and active_orbitals < 1:
        raise ValueError("active_orbitals must be positive.")
    if active_electrons is None:
        active_electrons = mol.nelectron - 2 * frozen_core
    if active_electrons < 0 or active_electrons > mol.nelectron:
        raise ValueError("active_electrons must be between zero and the molecular electron count.")
    if active_electrons % 2:
        raise ValueError("active_electrons must be even for a closed-shell active space.")
    ncore = (mol.nelectron - active_electrons) // 2
    if active_orbitals is None:
        active_orbitals = nmo - ncore
    if active_orbitals > nmo - ncore:
        raise ValueError("active_orbitals plus frozen core exceeds the number of molecular orbitals.")
    if active_electrons > 2 * active_orbitals:
        raise ValueError("active_electrons cannot exceed two electrons per active orbital.")
    if ncore < 0:
        raise ValueError("The active space contains more electrons than the molecule.")
    if frozen_core and frozen_core != ncore:
        raise ValueError(
            "frozen_core conflicts with active_electrons; specify either frozen_core "
            "or the corresponding active electron count."
        )
    return ncore, active_orbitals, active_electrons


def _cache_paths(
    cache_dir: Path,
    molecule_name: str,
    basis: str,
    active_electrons: int,
    active_orbitals: int,
) -> tuple[Path, Path]:
    stem = f"{_safe_molecule_name(molecule_name)}_{basis}_{active_electrons}_{active_orbitals}"
    return cache_dir / f"{stem}_ham.pkl", cache_dir / f"{stem}_wfn.pkl"


def _dump_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _energy_from_hamiltonian(hamiltonian: dict) -> tuple[float, np.ndarray]:
    from pyscf import fci

    solver = fci.direct_spin1.FCI()
    energy, ci_vector = solver.kernel(
        hamiltonian["h1e"],
        hamiltonian["eri"],
        hamiltonian["norb"],
        hamiltonian["nelec"],
        ecore=hamiltonian["ecore"],
    )
    return float(energy), np.asarray(ci_vector)


def get_electronic_structure(
    geometry: str | Path | Sequence[Sequence[object]],
    basis: str,
    *,
    molecule_name: str | None = None,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    frozen_core: int = 0,
    charge: int = 0,
    spin: int = 0,
    hamiltonian_only: bool = False,
    calculate_energy: bool = False,
    use_cache: bool = True,
    cache_dir: str | Path = "~/Angel/data/electronic_fci",
    use_gpu: bool | None = None,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
    verbose: int = 0,
) -> ElectronicStructure:
    """Generate or load a closed-shell active-space electronic structure.

    The active space is specified by the number of active electrons and
    spatial orbitals.  Any remaining molecular electrons are represented as a
    frozen core.  Two cache files are written under ``cache_dir``:
    ``..._ham.pkl`` for the Hamiltonian and ``..._wfn.pkl`` for the CI
    wavefunction.  Set ``hamiltonian_only=True`` to avoid generating or
    loading the wavefunction file.  Set ``calculate_energy=True`` to include
    the FCI energy in the returned object.
    """
    from pyscf import ao2mo, mcscf

    _closed_shell_check(spin, active_electrons)
    basis_name = _normalise_basis(basis)
    mol = build_molecule(geometry, basis_name, charge=charge, spin=spin, verbose=verbose)
    gpu_available = gpu4pyscf_available()
    requested_gpu = gpu_available if use_gpu is None else (bool(use_gpu) and gpu_available)
    gpu_used = False
    try:
        mf = _mean_field(mol, requested_gpu, conv_tol, max_cycle, verbose)
        gpu_used = requested_gpu
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Mean-field calculation did not converge.")
    except Exception as error:
        if not requested_gpu:
            raise
        warnings.warn(f"GPU mean-field calculation failed; retrying on CPU: {error}", RuntimeWarning)
        mf = _mean_field(mol, False, conv_tol, max_cycle, verbose)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Mean-field calculation did not converge on CPU.")
        gpu_used = False

    mo_coeff = _as_numpy(mf.mo_coeff)
    ncore, ncas, nelecas = _active_space_parameters(
        mol, mo_coeff.shape[1], active_electrons, active_orbitals, frozen_core
    )
    active_electrons = nelecas
    molecule_name = molecule_name or _default_molecule_name(geometry)
    cache_root = Path(cache_dir).expanduser()
    ham_path, wfn_path = _cache_paths(cache_root, molecule_name, basis_name, active_electrons, ncas)

    hamiltonian = None
    if use_cache and ham_path.exists():
        hamiltonian = _load_pickle(ham_path)
    if hamiltonian is None:
        cas = mcscf.CASCI(mf, ncas, (nelecas // 2, nelecas // 2))
        h1e, ecore = cas.get_h1eff()
        active_mo = mo_coeff[:, ncore : ncore + ncas]
        eri = _as_numpy(ao2mo.restore(1, ao2mo.kernel(mol, active_mo), ncas))
        space = generate_fci_space(ncas, nelecas)
        hamiltonian = {
            "h1e": _as_numpy(h1e),
            "eri": eri,
            "ecore": float(ecore),
            "norb": ncas,
            "nelec": (nelecas // 2, nelecas // 2),
            "space": space,
            "basis": basis_name,
            "charge": charge,
            "spin": spin,
            "molecule_name": molecule_name,
            "gpu_available": gpu_available,
            "gpu_used": gpu_used,
        }
        _dump_pickle(ham_path, hamiltonian)

    wavefunction = None
    energy = None
    if not hamiltonian_only:
        if use_cache and wfn_path.exists():
            wavefunction = _load_pickle(wfn_path)
        if wavefunction is None:
            energy, ci_vector = _energy_from_hamiltonian(hamiltonian)
            wavefunction = {
                "ci_vector": ci_vector,
                "space": hamiltonian["space"],
                "norb": hamiltonian["norb"],
                "nelec": hamiltonian["nelec"],
            }
            _dump_pickle(wfn_path, wavefunction)

    if calculate_energy:
        if energy is None:
            energy, _ = _energy_from_hamiltonian(hamiltonian)
    return ElectronicStructure(
        hamiltonian=hamiltonian,
        wavefunction=wavefunction,
        energy_hartree=energy,
        hamiltonian_path=ham_path,
        wavefunction_path=None if hamiltonian_only else wfn_path,
    )


def generate_fci_space(norb: int, nelec: int | tuple[int, int], spin: int = 0) -> FCISpace:
    """Generate PySCF's alpha/beta determinant space without solving it."""
    from pyscf.fci import cistring

    if norb < 1:
        raise ValueError("norb must be positive.")
    if isinstance(nelec, tuple):
        nalpha, nbeta = nelec
    else:
        nalpha = (nelec + spin) // 2
        nbeta = (nelec - spin) // 2
    if nalpha < 0 or nbeta < 0 or nalpha > norb or nbeta > norb:
        raise ValueError(f"Invalid electron count ({nalpha}, {nbeta}) for {norb} orbitals.")
    orbitals = range(norb)
    alpha = np.asarray(cistring.gen_strings4orblist(orbitals, nalpha), dtype=np.int64)
    beta = np.asarray(cistring.gen_strings4orblist(orbitals, nbeta), dtype=np.int64)
    return FCISpace(norb, nalpha, nbeta, alpha, beta)


def _mean_field(mol, use_gpu: bool, conv_tol: float, max_cycle: int, verbose: int):
    from pyscf import scf

    method = scf.RHF(mol) if mol.spin == 0 else scf.ROHF(mol)
    method.conv_tol = conv_tol
    method.max_cycle = max_cycle
    method.verbose = verbose
    if use_gpu:
        method = method.to_gpu()
    return method


def calculate_fci_energy(
    geometry: str | Path | Sequence[Sequence[object]],
    basis: str,
    *,
    charge: int = 0,
    spin: int = 0,
    frozen_core: int = 0,
    use_gpu: bool | None = None,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
    verbose: int = 0,
) -> FCIResult:
    """Run SCF followed by PySCF FCI and return total energy plus CI data.

    ``frozen_core`` is the number of lowest occupied *spatial* orbitals to
    freeze.  It is implemented with PySCF CASCI, so the returned CI vector and
    :class:`FCISpace` describe only the active orbitals.  With zero frozen
    orbitals, the direct PySCF FCI solver is used.
    """
    from pyscf import ao2mo, fci, mcscf

    _closed_shell_check(spin)
    if frozen_core < 0:
        raise ValueError("frozen_core must be non-negative.")
    mol = build_molecule(geometry, basis, charge=charge, spin=spin, verbose=verbose)
    gpu_available = gpu4pyscf_available()
    # An explicit request still degrades cleanly to CPU when CUDA support is
    # not installed; GPU4PySCF is only entered when it was detected above.
    requested_gpu = gpu_available if use_gpu is None else (bool(use_gpu) and gpu_available)
    gpu_used = False
    try:
        mf = _mean_field(mol, requested_gpu, conv_tol, max_cycle, verbose)
        gpu_used = requested_gpu
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Mean-field calculation did not converge.")
    except Exception as error:
        if not requested_gpu:
            raise
        warnings.warn(f"GPU mean-field calculation failed; retrying on CPU: {error}", RuntimeWarning)
        mf = _mean_field(mol, False, conv_tol, max_cycle, verbose)
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("Mean-field calculation did not converge on CPU.")
        gpu_used = False

    mo_coeff = _as_numpy(mf.mo_coeff)
    nmo = mo_coeff.shape[1]
    nalpha = (mol.nelectron + mol.spin) // 2
    nbeta = (mol.nelectron - mol.spin) // 2
    if frozen_core > min(nalpha, nbeta):
        raise ValueError("frozen_core cannot exceed the number of doubly occupied orbitals.")

    if frozen_core == 0:
        hcore = _as_numpy(mf.get_hcore())
        h1e = mo_coeff.T @ hcore @ mo_coeff
        eri = _as_numpy(ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), nmo))
        solver = fci.direct_spin1.FCI(mol)
        electronic, ci_vector = solver.kernel(
            h1e, eri, nmo, (nalpha, nbeta), ecore=0.0
        )
        active_norb, active_nelec = nmo, (nalpha, nbeta)
        nuclear = float(mol.energy_nuc())
        total = float(electronic + nuclear)
    else:
        ncas = nmo - frozen_core
        nelecas = (nalpha - frozen_core, nbeta - frozen_core)
        cas = mcscf.CASCI(mf, ncas, nelecas)
        total, _, ci_vector, _, _ = cas.kernel()
        total = float(total)
        # CASCI's total energy includes nuclear repulsion and core energy.
        nuclear = float(mol.energy_nuc())
        electronic = total - nuclear
        active_norb, active_nelec = ncas, nelecas

    space = generate_fci_space(active_norb, active_nelec)
    return FCIResult(
        total_energy_hartree=total,
        electronic_energy_hartree=float(electronic),
        nuclear_repulsion_hartree=nuclear,
        ci_vector=np.asarray(ci_vector),
        space=space,
        basis=_normalise_basis(basis),
        charge=charge,
        spin=spin,
        frozen_core=frozen_core,
        gpu_available=gpu_available,
        gpu_used=gpu_used,
        converged=True,
    )
