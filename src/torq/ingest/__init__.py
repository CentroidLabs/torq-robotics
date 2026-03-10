"""Torq ingest sub-package — format parsers, auto-detection, and temporal alignment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from torq.ingest._detect import _SUPPORTED_FORMATS, detect_format
from torq.ingest.alignment import align
from torq.ingest.hdf5 import ingest as ingest_hdf5
from torq.ingest.lerobot import ingest as ingest_lerobot
from torq.ingest.mcap import ingest as ingest_mcap

if TYPE_CHECKING:
    from torq.episode import Episode

__all__ = [
    "align",
    "detect_format",
    "ingest",
    "ingest_hdf5",
    "ingest_lerobot",
    "ingest_mcap",
]

logger = logging.getLogger(__name__)


def ingest(
    path: str | Path,
    fmt: str = "auto",
    errors: list[dict] | None = None,
    stats: dict[str, int] | None = None,
) -> list[Episode]:
    """Ingest robot recordings from a file or directory into canonical Episodes.

    Supports MCAP (ROS 2), HDF5 (robomimic), and LeRobot v3.0 formats.
    When ``fmt='auto'`` (default), format is detected from file extension
    and magic bytes. When ``path`` is a directory, all supported files
    are ingested recursively; corrupt files are skipped with a warning.

    Args:
        path: Path to a single file, a LeRobot dataset directory, or a directory
            containing multiple recording files.
        fmt: Format override. One of ``"auto"``, ``"mcap"``, ``"hdf5"``,
            ``"lerobot"``. Default ``"auto"`` uses detection heuristics.
        errors: Optional output list. In directory bulk mode, dicts with
            ``{"path": str, "reason": str}`` are appended for each failed file.
            Ignored in single-file mode (errors raise ``TorqIngestError``).
        stats: Optional output dict. In directory bulk mode, populated with
            ``{"files_succeeded": int}`` — the number of files that produced
            at least one episode. In single-file mode, set to 1 on success.

    Returns:
        List of Episode objects. Empty list if directory contains no ingestible
        files.

    Raises:
        TorqIngestError: If a single file's format is unrecognised, or if a
            specified format does not match the file (bulk mode: corrupt files
            are skipped, not raised).

    Examples:
        >>> episodes = tq.ingest('./recordings/session.mcap')
        >>> episodes = tq.ingest('./lerobot_dataset/')
        >>> episodes = tq.ingest('./recordings/', fmt='auto')
        >>> failures: list[dict] = []
        >>> info: dict[str, int] = {}
        >>> episodes = tq.ingest('./recordings/', errors=failures, stats=info)
    """
    from torq.errors import TorqIngestError

    root = Path(path)

    if fmt != "auto":
        result = _dispatch(root, fmt)
        if stats is not None:
            stats["files_succeeded"] = 1
        return result

    detected = detect_format(root)

    if detected == "directory":
        return _ingest_directory(root, errors=errors, stats=stats)

    if detected == "unknown":
        raise TorqIngestError(
            f"Cannot ingest '{root}': detected format is 'unknown'. "
            f"Supported formats: {_SUPPORTED_FORMATS}. "
            f"If this is a directory of recordings, ensure it contains "
            f".mcap, .hdf5, .h5 files or a LeRobot dataset (meta/info.json)."
        )

    result = _dispatch(root, detected)
    if stats is not None:
        stats["files_succeeded"] = 1
    return result


def _dispatch(path: Path, fmt: str) -> list[Episode]:
    """Dispatch to the correct format-specific ingester."""
    from torq.errors import TorqIngestError

    if fmt == "mcap":
        return ingest_mcap(path)
    if fmt == "hdf5":
        return ingest_hdf5(path)
    if fmt == "lerobot":
        return ingest_lerobot(path)
    raise TorqIngestError(f"Unknown format '{fmt}'. Supported formats: {_SUPPORTED_FORMATS}.")


def _ingest_directory(
    root: Path,
    errors: list[dict] | None = None,
    stats: dict[str, int] | None = None,
) -> list[Episode]:
    """Recursively discover and ingest all supported files in a directory."""
    from torq._config import config
    from torq.errors import TorqIngestError

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover

        def tqdm(it, **_kw):  # type: ignore[misc]
            return it

    # Discover LeRobot subdatasets first (directories with meta/info.json)
    lerobot_dirs: set[Path] = set()
    for info_json in root.rglob("meta/info.json"):
        lerobot_dirs.add(info_json.parent.parent)

    # Discover flat files (MCAP, HDF5), excluding files under LeRobot dirs
    flat_files: list[Path] = []
    for pattern in ("**/*.mcap", "**/*.hdf5", "**/*.h5"):
        for f in root.glob(pattern):
            if not any(f.is_relative_to(d) for d in lerobot_dirs):
                flat_files.append(f)

    items: list[Path] = sorted(flat_files) + sorted(lerobot_dirs)

    if not items:
        logger.warning("Directory '%s' is empty — no ingestible files found.", root)
        return []

    all_episodes: list[Episode] = []
    failed: list[Path] = []
    success_count = 0

    for item in tqdm(items, desc="Ingesting", unit="file", disable=config.quiet):
        try:
            detected = detect_format(item)
            if detected == "unknown":
                raise TorqIngestError(
                    f"Cannot ingest '{item}': detected format is 'unknown'. "
                    f"Supported formats: {_SUPPORTED_FORMATS}."
                )
            if detected == "directory":
                continue
            eps = _dispatch(item, detected)
            all_episodes.extend(eps)
            success_count += 1
        except TorqIngestError as exc:
            logger.warning("Skipping '%s': %s", item, exc)
            failed.append(item)
            if errors is not None:
                errors.append({"path": str(item), "reason": str(exc)})
        except Exception as exc:
            reason = f"unexpected {type(exc).__name__}: {exc}"
            logger.warning("Skipping '%s': %s", item, reason)
            failed.append(item)
            if errors is not None:
                errors.append({"path": str(item), "reason": reason})

    if stats is not None:
        stats["files_succeeded"] = success_count

    if failed:
        logger.info(
            "Ingested %d episodes from %d files (%d files failed — see warnings).",
            len(all_episodes),
            success_count,
            len(failed),
        )

    return all_episodes
