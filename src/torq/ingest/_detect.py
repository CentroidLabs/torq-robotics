"""Format auto-detection for Torq ingest.

Inspects file extensions, magic bytes, and directory structure to determine
which ingester to dispatch. Returns a format string consumed by
:func:`torq.ingest.ingest`.
"""

from __future__ import annotations

import logging
from pathlib import Path

__all__ = ["detect_format"]

logger = logging.getLogger(__name__)

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_SUPPORTED_FORMATS = ("mcap", "hdf5", "lerobot")


def detect_format(path: Path) -> str:
    """Detect dataset format from a file or directory path.

    Args:
        path: Path to a file or directory to classify.

    Returns:
        One of ``"mcap"``, ``"hdf5"``, ``"lerobot"``, ``"directory"``, or
        ``"unknown"``. ``"directory"`` is a sentinel indicating the caller
        should scan for individual files within.

    Raises:
        TorqIngestError: If the path is a ``.parquet`` file (not a dataset
            root — use the parent LeRobot directory instead).
    """
    if path.is_dir():
        if (path / "meta" / "info.json").exists():
            return "lerobot"
        return "directory"

    suffix = path.suffix.lower()
    if suffix == ".mcap":
        return "mcap"
    if suffix in (".hdf5", ".h5"):
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            if header != _HDF5_MAGIC:
                logger.warning(
                    "File '%s' has HDF5 extension but magic bytes do not match. "
                    "Proceeding with HDF5 ingestion — file may be corrupt.",
                    path,
                )
        except OSError:
            pass
        return "hdf5"
    if suffix == ".parquet":
        from torq.errors import TorqIngestError

        raise TorqIngestError(
            f"Cannot ingest '{path}': .parquet files are not dataset roots. "
            f"To ingest a LeRobot dataset, point to the parent directory "
            f"containing meta/info.json. Supported formats: {_SUPPORTED_FORMATS}."
        )
    return "unknown"
