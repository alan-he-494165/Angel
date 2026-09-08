"""Read single-frame XYZ files and multi-frame XYZ trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GeometryFrame:
    """One XYZ frame, with coordinates stored in Angstrom."""

    symbols: tuple[str, ...]
    coordinates: np.ndarray
    comment: str = ""

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def atom_string(self) -> str:
        """Return a PySCF-compatible atom string."""
        return "\n".join(
            f"{symbol} {x:.14g} {y:.14g} {z:.14g}"
            for symbol, (x, y, z) in zip(self.symbols, self.coordinates)
        )

    def as_pyscf_atom(self) -> list[tuple[str, tuple[float, float, float]]]:
        return [
            (symbol, (float(x), float(y), float(z)))
            for symbol, (x, y, z) in zip(self.symbols, self.coordinates)
        ]


def _parse_frame(lines: list[str], start: int, source: Path) -> tuple[GeometryFrame, int]:
    if start >= len(lines) or not lines[start].strip():
        raise ValueError(f"{source}: unexpected end of file while reading an XYZ frame.")
    try:
        natoms = int(lines[start].strip())
    except ValueError as error:
        raise ValueError(f"{source}: line {start + 1} must contain an atom count.") from error
    if natoms < 1:
        raise ValueError(f"{source}: XYZ atom count must be positive.")
    if start + 1 >= len(lines):
        raise ValueError(f"{source}: missing XYZ comment line for frame at line {start + 1}.")
    comment = lines[start + 1].rstrip("\n")
    atom_lines = lines[start + 2 : start + 2 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"{source}: incomplete XYZ frame starting at line {start + 1}.")
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for offset, line in enumerate(atom_lines, start=start + 3):
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{source}: line {offset} must contain symbol and x/y/z coordinates.")
        try:
            xyz = [float(value) for value in fields[1:4]]
        except ValueError as error:
            raise ValueError(f"{source}: invalid coordinates on line {offset}.") from error
        symbols.append(fields[0])
        coordinates.append(xyz)
    return GeometryFrame(tuple(symbols), np.asarray(coordinates, dtype=float), comment), start + 2 + natoms


def read_xyz_trajectory(path: str | Path) -> list[GeometryFrame]:
    """Read every frame from a single- or multi-frame XYZ file."""
    source = Path(path).expanduser()
    lines = source.read_text(encoding="utf-8").splitlines()
    frames: list[GeometryFrame] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        frame, index = _parse_frame(lines, index, source)
        frames.append(frame)
    if not frames:
        raise ValueError(f"{source}: no XYZ frames found.")
    return frames


def read_xyz(path: str | Path) -> GeometryFrame:
    """Read exactly one XYZ frame."""
    frames = read_xyz_trajectory(path)
    if len(frames) != 1:
        raise ValueError(f"{path}: expected one XYZ frame, found {len(frames)}.")
    return frames[0]


def get_geom(
    path: str | Path,
    frame: int | None = None,
    frames: Sequence[int] | None = None,
):
    """Load geometry from XYZ.

    With no selection all frames are returned.  With ``frame`` one
    :class:`GeometryFrame` is returned.  With ``frames`` a list of selected
    frames is returned.  Negative indices are supported.
    """
    if frame is not None and frames is not None:
        raise ValueError("Specify either frame or frames, not both.")
    all_frames = read_xyz_trajectory(path)
    if frame is None:
        if frames is None:
            return all_frames
        try:
            return [all_frames[index] for index in frames]
        except IndexError as error:
            raise IndexError(
                f"A requested XYZ frame index is out of range for {len(all_frames)} frames."
            ) from error
    try:
        return all_frames[frame]
    except IndexError as error:
        raise IndexError(
            f"XYZ frame index {frame} is out of range for {len(all_frames)} frames."
        ) from error
