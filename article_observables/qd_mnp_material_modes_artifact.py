"""Lightweight NPZ artifact helpers for material-mode comparison workflows.

This module deliberately imports no QD--MNP mathematical model.  Calculation
programs use it to write self-contained, pickle-free artifacts and plotting
programs use it to validate and read those artifacts without importing a
solver.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = 1


def json_ready(value: Any) -> Any:
    """Convert common scientific-Python values to strict JSON data."""

    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite floating-point values are not valid metadata.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported metadata value of type {type(value).__name__}.")


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 hash of JSON-compatible data."""

    encoded = json.dumps(
        json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_provenance(project_root: str | Path) -> dict[str, Any]:
    """Best-effort local git provenance without mutating the checkout."""

    root = Path(project_root)

    def run(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def source_hashes(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash the exact source files used to produce an artifact."""

    result: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).resolve()
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def atomic_write_npz(
    path: str | Path,
    payload: Mapping[str, np.ndarray | Any],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write one compressed, pickle-free NPZ artifact."""

    output = Path(path)
    if output.suffix.lower() != ".npz":
        raise ValueError("The output artifact must have a .npz suffix.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")

    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == "metadata_json":
            raise ValueError("payload must not define the reserved metadata_json key.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"Artifact array {key!r} has forbidden object dtype.")
        arrays[str(key)] = array

    document = dict(metadata)
    document.setdefault("schema_version", SCHEMA_VERSION)
    document.setdefault("created_at", datetime.now().astimezone().isoformat())
    document["artifact_file"] = output.name
    document["array_keys"] = sorted([*arrays, "metadata_json"])
    metadata_json = json.dumps(
        json_ready(document),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}_",
        suffix=".npz",
        dir=output.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            metadata_json=np.asarray(metadata_json),
            **arrays,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_npz_artifact(
    path: str | Path,
    *,
    schema_name: str,
    required_arrays: Iterable[str] = (),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and strictly validate a material-comparison artifact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        actual_keys = sorted(archive.files)
        if "metadata_json" not in archive.files:
            raise ValueError("Artifact is missing metadata_json.")
        raw_metadata = np.asarray(archive["metadata_json"])
        if raw_metadata.ndim != 0 or raw_metadata.dtype.kind not in {"U", "S"}:
            raise ValueError("metadata_json must be one scalar string array.")
        metadata_item = raw_metadata.item()
        if isinstance(metadata_item, bytes):
            metadata_text = metadata_item.decode("utf-8")
        else:
            metadata_text = str(metadata_item)
        metadata = json.loads(metadata_text)
        payload = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }

    if metadata.get("schema_name") != schema_name:
        raise ValueError(
            f"Expected schema_name={schema_name!r}, got "
            f"{metadata.get('schema_name')!r}."
        )
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}."
        )
    declared_keys = metadata.get("array_keys")
    if declared_keys != actual_keys:
        raise ValueError("metadata array_keys does not match the NPZ contents.")
    missing = sorted(set(required_arrays) - set(payload))
    if missing:
        raise ValueError(f"Artifact is missing required arrays: {missing}.")
    for key, array in payload.items():
        if array.dtype.hasobject:
            raise TypeError(f"Artifact array {key!r} has forbidden object dtype.")
    return payload, metadata


def save_figure(
    figure: Any,
    output: str | Path,
    *,
    dpi: int = 220,
    show: bool = False,
) -> Path:
    """Save a Matplotlib figure and optionally display it."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(figure)
    return destination
