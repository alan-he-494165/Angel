"""Portable NumPy/JSON checkpoints for FiRE parameter pytrees."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _checkpoint_paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path).expanduser()
    arrays = target if target.suffix == ".npz" else target.with_suffix(".npz")
    return arrays, arrays.with_suffix(".json")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    elif hasattr(value, "provenance"):
        value = value.provenance()
    elif dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return json.loads(json.dumps(value, default=str))


def _path_keys(path) -> list[str | int]:
    keys: list[str | int] = []
    for entry in path:
        if hasattr(entry, "key"):
            keys.append(entry.key)
        elif hasattr(entry, "idx"):
            keys.append(int(entry.idx))
        elif hasattr(entry, "name"):
            keys.append(entry.name)
        else:
            raise TypeError(f"unsupported pytree path entry {entry!r}")
    return keys


def save_params(path, params, *, config=None, system=None):
    """Save parameter leaves losslessly with a JSON structure sidecar."""

    arrays_path, metadata_path = _checkpoint_paths(path)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    arrays = {f"leaf_{i:06d}": np.asarray(leaf)
              for i, (_, leaf) in enumerate(path_leaves)}
    np.savez(arrays_path, **arrays)
    metadata = {
        "format_version": 1,
        "container_type": "dict",
        "leaves": [
            {
                "array": f"leaf_{i:06d}",
                "path": _path_keys(tree_path),
                "shape": list(np.asarray(leaf).shape),
                "dtype": str(np.asarray(leaf).dtype),
            }
            for i, (tree_path, leaf) in enumerate(path_leaves)
        ],
        "config": _json_safe(config),
        "system": _json_safe(system),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"arrays": arrays_path, "metadata": metadata_path}


def load_params(path):
    """Load a FiRE checkpoint as the plain nested dict returned by Flax init."""

    arrays_path, metadata_path = _checkpoint_paths(path)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format {metadata.get('format_version')!r}")
    root: dict[str, Any] = {}
    with np.load(arrays_path, allow_pickle=False) as archive:
        for leaf in metadata["leaves"]:
            path_keys = leaf["path"]
            if not path_keys or not all(isinstance(key, str) for key in path_keys):
                raise ValueError("FiRE parameter checkpoints require nested string-key dicts.")
            node = root
            for key in path_keys[:-1]:
                node = node.setdefault(key, {})
            value = archive[leaf["array"]]
            if list(value.shape) != leaf["shape"] or str(value.dtype) != leaf["dtype"]:
                raise ValueError(f"checkpoint leaf metadata mismatch at {path_keys!r}")
            node[path_keys[-1]] = jax.device_put(value)
    return root
