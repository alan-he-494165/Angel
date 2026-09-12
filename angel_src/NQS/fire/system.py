"""Molecular system definition for the FiRE solver.

Real-space neural-network VMC works directly with nuclear coordinates and
charges rather than with a second-quantized Hamiltonian, so this module
converts an XYZ geometry into the arrays the ansatz and the local energy
need.  Coordinates are stored in bohr; the repository's XYZ files are in
Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANGSTROM_TO_BOHR = 1.8897261254578281

_PERIODIC_TABLE: dict[str, int] = {
    symbol: index
    for index, symbol in enumerate(
        (
            "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr "
            "Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr"
        ).split()
    )
}


def atomic_number(symbol: str) -> int:
    """Return the nuclear charge of an element symbol."""

    key = symbol.strip().capitalize()
    if key in _PERIODIC_TABLE:
        return _PERIODIC_TABLE[key]
    try:  # elements beyond Kr are rare here, but do not fail silently
        from pyscf.data.elements import charge as _pyscf_charge

        return int(_pyscf_charge(key))
    except Exception as error:  # pragma: no cover - depends on pyscf
        raise ValueError(f"Unknown element symbol {symbol!r}.") from error


@dataclass(frozen=True)
class ECPChannel:
    """One angular-momentum channel of an effective core potential.

    The radial function follows PySCF's convention,

        V_l(r) = sum_i c_i r^(p_i) exp(-a_i r^2),

    with ``p_i`` in ``r_powers`` running from -2 upwards.  ``angular_momentum``
    is ``-1`` for the local channel and ``l >= 0`` for a nonlocal projector.
    """

    angular_momentum: int
    coefficients: np.ndarray
    exponents: np.ndarray
    r_powers: np.ndarray

    def __post_init__(self) -> None:
        shapes = {self.coefficients.shape, self.exponents.shape, self.r_powers.shape}
        if len(shapes) != 1 or len(self.coefficients.shape) != 1:
            raise ValueError("ECP channel arrays must be 1-D and the same length.")

    @property
    def is_local(self) -> bool:
        return self.angular_momentum < 0

    def evaluate(self, distance: np.ndarray) -> np.ndarray:
        """V_l at the given electron-nucleus distances (NumPy, for tests)."""

        d = np.asarray(distance)[..., None]
        return np.sum(
            self.coefficients * d**self.r_powers * np.exp(-self.exponents * d**2),
            axis=-1,
        )


@dataclass(frozen=True)
class NuclearECP:
    """The effective core potential carried by one nucleus.

    ``n_core`` electrons are removed from the system and the remaining
    electrons see the reduced charge ``z_eff = Z - n_core`` plus the channels
    below.  Some families (ccECP among them) define a potential for hydrogen
    with ``n_core = 0``: nothing is removed, but the bare Coulomb singularity
    is still softened, so ``channels`` must be applied regardless of
    ``n_core``.
    """

    symbol: str
    name: str
    n_core: int
    z_eff: float
    channels: tuple[ECPChannel, ...]

    @property
    def local_channel(self) -> ECPChannel | None:
        for channel in self.channels:
            if channel.is_local:
                return channel
        return None

    @property
    def nonlocal_channels(self) -> tuple[ECPChannel, ...]:
        return tuple(channel for channel in self.channels if not channel.is_local)

    @property
    def max_angular_momentum(self) -> int:
        """Largest l with a projector, or -1 if the potential is local only."""

        nonlocal_channels = self.nonlocal_channels
        if not nonlocal_channels:
            return -1
        return max(channel.angular_momentum for channel in nonlocal_channels)


def _channels_from_pyscf(blocks) -> tuple[ECPChannel, ...]:
    """Convert PySCF's nested ECP block format into :class:`ECPChannel`."""

    channels: list[ECPChannel] = []
    for angular_momentum, buckets in blocks:
        coefficients: list[float] = []
        exponents: list[float] = []
        powers: list[float] = []
        # ``buckets`` is indexed by the power of r offset by two.
        for index, terms in enumerate(buckets):
            for exponent, coefficient in terms:
                coefficients.append(float(coefficient))
                exponents.append(float(exponent))
                powers.append(float(index - 2))
        if not coefficients:
            continue
        channels.append(
            ECPChannel(
                angular_momentum=int(angular_momentum),
                coefficients=np.array(coefficients, dtype=float),
                exponents=np.array(exponents, dtype=float),
                r_powers=np.array(powers, dtype=float),
            )
        )
    return tuple(channels)


def ecp_from_name(
    name: "str | dict[str, str | None] | None",
    symbols: "list[str] | tuple[str, ...]",
) -> "tuple[NuclearECP | None, ...] | None":
    """Load per-nucleus ECP data from PySCF's tables.

    ``name`` is either a family name applied to every element that has an
    entry (e.g. ``"ccecp"``), a per-element mapping, or ``None`` for the
    all-electron treatment.  Elements without an entry in the family keep
    their full nuclear charge, so mixed systems are handled transparently.
    """

    if name is None:
        return None
    from pyscf.gto.basis import load_ecp as _pyscf_load_ecp

    mapping = name if isinstance(name, dict) else {symbol: name for symbol in symbols}
    out: list[NuclearECP | None] = []
    for symbol in symbols:
        family = mapping.get(symbol)
        if not family:
            out.append(None)
            continue
        data = _pyscf_load_ecp(family, symbol)
        if not data:
            out.append(None)
            continue
        n_core = int(data[0])
        channels = _channels_from_pyscf(data[1])
        if not channels:
            out.append(None)
            continue
        out.append(
            NuclearECP(
                symbol=symbol,
                name=str(family),
                n_core=n_core,
                z_eff=float(atomic_number(symbol) - n_core),
                channels=channels,
            )
        )
    if all(entry is None for entry in out):
        raise ValueError(
            f"effective core potential {name!r} has no entry for any of {list(symbols)}."
        )
    return tuple(out)


@dataclass(frozen=True)
class MolecularSystem:
    """A molecule in the fixed-nuclei Born-Oppenheimer approximation.

    Attributes
    ----------
    symbols:
        Element symbols, one per nucleus.
    nuclei:
        ``(n_nuclei, 3)`` nuclear coordinates in bohr.
    charges:
        ``(n_nuclei,)`` nuclear charges Z_m.  These are the full atomic
        numbers in the default all-electron treatment and the reduced charges
        Z - n_core when ``ecp`` is set.
    n_up, n_down:
        Electron counts per spin channel.  Following the paper's convention
        the first ``n_up`` electrons of a configuration are spin-up.
    name:
        Label used for output paths.
    ecp:
        Optional effective core potential, one entry per nucleus (``None``
        for nuclei treated all-electron).  ``None`` for the whole field --
        the default -- selects the all-electron Hamiltonian, which is the
        validated path; see ``README.md``.
    """

    symbols: tuple[str, ...]
    nuclei: np.ndarray
    charges: np.ndarray
    n_up: int
    n_down: int
    name: str = "molecule"
    ecp: "tuple[NuclearECP | None, ...] | None" = None

    def __post_init__(self) -> None:
        if self.nuclei.shape != (len(self.symbols), 3):
            raise ValueError("nuclei must have shape (n_nuclei, 3).")
        if self.charges.shape != (len(self.symbols),):
            raise ValueError("charges must have shape (n_nuclei,).")
        if self.n_up < 0 or self.n_down < 0:
            raise ValueError("electron counts must be non-negative.")
        if self.n_up + self.n_down < 1:
            raise ValueError("system must contain at least one electron.")
        if self.n_up < self.n_down:
            raise ValueError("use n_up >= n_down; flip the spin label instead.")
        if self.ecp is not None:
            if len(self.ecp) != len(self.symbols):
                raise ValueError("ecp must have one entry per nucleus (None where absent).")
            expected = np.array(
                [
                    atomic_number(symbol) if entry is None else entry.z_eff
                    for symbol, entry in zip(self.symbols, self.ecp)
                ],
                dtype=float,
            )
            if not np.allclose(self.charges, expected):
                raise ValueError(
                    "charges must be the ECP-reduced charges Z - n_core "
                    f"({expected.tolist()}), got {np.asarray(self.charges).tolist()}."
                )

    @property
    def n_nuclei(self) -> int:
        return len(self.symbols)

    @property
    def n_electrons(self) -> int:
        return self.n_up + self.n_down

    @property
    def spin(self) -> int:
        """2S, i.e. the number of unpaired electrons."""

        return self.n_up - self.n_down

    @property
    def charge(self) -> int:
        return int(round(float(self.charges.sum()))) - self.n_electrons

    @property
    def nuclear_repulsion(self) -> float:
        """Nucleus-nucleus Coulomb energy in hartree (last term of Eq. 9)."""

        if self.n_nuclei < 2:
            return 0.0
        diff = self.nuclei[:, None, :] - self.nuclei[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        pair = np.triu_indices(self.n_nuclei, k=1)
        return float(np.sum(self.charges[pair[0]] * self.charges[pair[1]] / distance[pair]))

    @property
    def has_ecp(self) -> bool:
        """True when any nucleus carries an effective core potential."""

        return self.ecp is not None and any(entry is not None for entry in self.ecp)

    @property
    def n_core_electrons(self) -> int:
        """Electrons removed by the ECP; zero in the all-electron treatment."""

        if self.ecp is None:
            return 0
        return sum(0 if entry is None else entry.n_core for entry in self.ecp)

    @property
    def ecp_name(self) -> str | None:
        """Name of the ECP family, or ``None`` for an all-electron system."""

        if self.ecp is None:
            return None
        names = {entry.name for entry in self.ecp if entry is not None}
        if not names:
            return None
        return "/".join(sorted(names))

    def pyscf_ecp_spec(self) -> dict[str, str]:
        """``ecp=`` argument reproducing this treatment in a PySCF ``Mole``."""

        if self.ecp is None:
            return {}
        return {
            symbol: entry.name
            for symbol, entry in zip(self.symbols, self.ecp)
            if entry is not None
        }

    def as_pyscf_atom(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Return the geometry in PySCF's ``atom`` format, in bohr."""

        return [
            (symbol, (float(x), float(y), float(z)))
            for symbol, (x, y, z) in zip(self.symbols, self.nuclei)
        ]

    def provenance(self) -> dict[str, object]:
        """Small JSON-serializable record of the electronic problem."""

        return {
            "name": self.name,
            "symbols": list(self.symbols),
            "n_nuclei": self.n_nuclei,
            "n_electrons": self.n_electrons,
            "n_up": self.n_up,
            "n_down": self.n_down,
            "charge": self.charge,
            "spin_2s": self.spin,
            "nuclear_repulsion_hartree": self.nuclear_repulsion,
            "core_treatment": "all-electron" if not self.has_ecp else f"ecp:{self.ecp_name}",
            "ecp": None if not self.has_ecp else self.ecp_name,
            "n_core_electrons_removed": self.n_core_electrons,
            "charges": [float(z) for z in self.charges],
        }


def system_from_arrays(
    symbols: "list[str] | tuple[str, ...]",
    coordinates: np.ndarray,
    *,
    charge: int = 0,
    spin: int = 0,
    units: str = "angstrom",
    name: str = "molecule",
    ecp: "str | dict[str, str | None] | None" = None,
) -> MolecularSystem:
    """Build a :class:`MolecularSystem` from symbols and coordinates.

    ``ecp`` is optional and defaults to ``None``, the all-electron treatment.
    Passing a family name such as ``"ccecp"`` replaces the core electrons of
    every element that has an entry, reducing both the nuclear charges and
    the electron count.
    """

    symbols = tuple(str(symbol) for symbol in symbols)
    coordinates = np.asarray(coordinates, dtype=float)
    unit = units.lower()
    if unit.startswith("ang"):
        coordinates = coordinates * ANGSTROM_TO_BOHR
    elif not unit.startswith("bohr") and not unit.startswith("au"):
        raise ValueError(f"unknown units {units!r}; use 'angstrom' or 'bohr'.")
    ecp_data = ecp_from_name(ecp, symbols)
    if ecp_data is None:
        charges = np.array([atomic_number(symbol) for symbol in symbols], dtype=float)
    else:
        charges = np.array(
            [
                atomic_number(symbol) if entry is None else entry.z_eff
                for symbol, entry in zip(symbols, ecp_data)
            ],
            dtype=float,
        )
    n_electrons = int(round(float(charges.sum()))) - int(charge)
    if n_electrons < 1:
        raise ValueError("charge leaves no electrons in the system.")
    if (n_electrons - int(spin)) % 2 != 0:
        raise ValueError(
            f"spin={spin} is incompatible with {n_electrons} electrons; "
            "n_electrons - spin must be even."
        )
    n_up = (n_electrons + int(spin)) // 2
    n_down = n_electrons - n_up
    return MolecularSystem(
        symbols=symbols,
        nuclei=coordinates,
        charges=charges,
        n_up=n_up,
        n_down=n_down,
        name=name,
        ecp=ecp_data,
    )


def system_from_xyz(
    path: str | Path,
    *,
    frame: int = 0,
    charge: int = 0,
    spin: int = 0,
    name: str | None = None,
    ecp: "str | dict[str, str | None] | None" = None,
) -> MolecularSystem:
    """Build a :class:`MolecularSystem` from one frame of an XYZ file."""

    from angel_src.geom.get_geom import read_xyz_trajectory

    source = Path(path).expanduser()
    frames = read_xyz_trajectory(source)
    if not -len(frames) <= frame < len(frames):
        raise IndexError(f"{source}: frame {frame} out of range ({len(frames)} frames).")
    geometry = frames[frame]
    label = name or f"{source.stem}_f{frame}" if len(frames) > 1 else name or source.stem
    return system_from_arrays(
        geometry.symbols,
        geometry.coordinates,
        charge=charge,
        spin=spin,
        units="angstrom",
        name=label,
        ecp=ecp,
    )
