"""Shared I/O helpers: JSONL read/write, safe directory creation, pickling.

Centralizing these small utilities avoids each pipeline stage reinventing
(and subtly diverging on) file-handling logic — for example, every stage
that persists intermediate results uses JSON Lines so downstream stages can
stream records without loading an entire dataset into memory at once
(important on 8GB RAM).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterator

from medgraphrag.utils.exceptions import MedGraphRAGError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dicts to a JSON Lines file, one JSON object per line.

    Args:
        records: List of JSON-serializable dictionaries.
        path: Destination file path; parent directories are created if needed.

    Raises:
        MedGraphRAGError: If serialization or writing fails.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} records to {out_path}")
    except (TypeError, OSError) as exc:
        raise MedGraphRAGError(f"Failed to write JSONL to {out_path}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dicts.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed dictionaries, one per non-empty line.

    Raises:
        MedGraphRAGError: If the file is missing or contains invalid JSON.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise MedGraphRAGError(f"JSONL file not found: {in_path}")

    records: list[dict[str, Any]] = []
    try:
        with in_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise MedGraphRAGError(f"Invalid JSON at line {line_num} in {in_path}: {exc}") from exc

    logger.info(f"Read {len(records)} records from {in_path}")
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a JSON Lines file record-by-record without loading it fully into memory.

    Args:
        path: Path to the ``.jsonl`` file.

    Yields:
        One parsed dictionary per non-empty line.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise MedGraphRAGError(f"JSONL file not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Persist an arbitrary Python object (e.g. a NetworkX graph) via pickle.

    Args:
        obj: Object to serialize.
        path: Destination file path.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    try:
        with out_path.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Pickled object to {out_path}")
    except (pickle.PicklingError, OSError) as exc:
        raise MedGraphRAGError(f"Failed to pickle object to {out_path}: {exc}") from exc


def load_pickle(path: str | Path) -> Any:
    """Load a pickled Python object from disk.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized object.

    Raises:
        MedGraphRAGError: If the file is missing or cannot be unpickled.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise MedGraphRAGError(f"Pickle file not found: {in_path}")
    try:
        with in_path.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        raise MedGraphRAGError(f"Failed to load pickle from {in_path}: {exc}") from exc
