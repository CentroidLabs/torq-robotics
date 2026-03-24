"""Sharded JSON index and Episode ID generation for the Torq storage layer.

This module is the single authoritative source for:
    - Episode ID generation (``ep_{n:04d}`` format)
    - String normalisation for task/embodiment categorical fields
    - Atomic JSON writes (write-to-tmp then os.replace)
    - Maintaining all four index shards: manifest, by_task, by_embodiment, quality
"""

from __future__ import annotations

import bisect
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torq.episode import Episode

__all__ = [
    "update_index",
    "read_manifest",
    "query_index",
    "read_snapshots",
    "write_snapshots",
    "read_experiments",
    "write_experiments",
    "update_manifest_lineage_counts",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def _normalise(s: str) -> str:
    """Normalise a categorical string for index bucket assignment.

    Applies three transforms in order:
        1. Lowercase
        2. Strip leading/trailing whitespace
        3. Remove separator characters (hyphens, spaces, underscores)

    This ensures that variations like ``"ALOHA-2"``, ``"aloha2"``, and
    ``"Aloha 2"`` all map to the same bucket key (``"aloha2"``).

    Args:
        s: Raw string value (e.g. task name or embodiment name).

    Returns:
        Normalised string suitable for use as an index bucket key.
    """
    return re.sub(r"[-\s_]+", "", s.lower().strip())


def _atomic_write_json(data: dict | list, path: Path) -> None:
    """Write JSON data atomically: write to a PID-namespaced ``.tmp`` file then ``os.replace()``.

    This guarantees the index is never left in a partial state — the file either
    contains the previous complete state or the new complete state.

    The temp file is named ``<path>.<pid>.tmp`` to avoid collisions between
    different processes writing to the same directory.

    .. note::
        R1 limitation: ``os.fsync()`` is not called before ``os.replace()``.
        On a hard-crash (power loss), the ``.tmp`` file may be partially written
        and the rename may not have occurred. The index would revert to the
        pre-save state rather than being corrupted, which is acceptable for R1.

        True concurrent multi-process writes are not protected against — R1
        assumes single-process, single-user operation.  The PID suffix prevents
        temp-file collisions between different processes but does not provide
        read-modify-write atomicity.

    Args:
        data: JSON-serialisable dict or list.
        path: Target file path (will be replaced atomically).

    Raises:
        OSError: If the write or rename fails.
    """
    tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _count_lineage_records(index_root: Path) -> tuple[int, int]:
    """Return (snapshot_count, experiment_count) from the lineage JSON files.

    Delegates to ``read_snapshots`` and ``read_experiments`` to avoid
    duplicating file I/O logic.  Returns ``(0, 0)`` if the files are absent.

    Args:
        index_root: Directory containing the JSON index shards.

    Returns:
        Tuple of ``(snapshot_count, experiment_count)``.
    """
    return len(read_snapshots(index_root)), len(read_experiments(index_root))


def update_manifest_lineage_counts(index_root: Path) -> None:
    """Refresh ``snapshot_count`` and ``experiment_count`` in ``manifest.json``.

    Should be called by snapshot/experiment code after persisting a new record.
    Creates ``manifest.json`` with zero episode_count if it does not yet exist.

    Args:
        index_root: Directory containing the JSON index shards.

    Raises:
        OSError: If the manifest write fails.
    """
    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / "manifest.json"

    manifest: dict = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"schema_version": SCHEMA_VERSION, "episode_count": 0}
    )

    snapshot_count, experiment_count = _count_lineage_records(index_root)
    manifest["snapshot_count"] = snapshot_count
    manifest["experiment_count"] = experiment_count
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    _atomic_write_json(manifest, manifest_path)
    logger.debug(
        "Manifest lineage counts updated: snapshots=%d experiments=%d",
        snapshot_count,
        experiment_count,
    )


def _next_episode_id(index_root: Path) -> str:
    """Generate the next Episode ID by reading the manifest counter.

    The counter represents the *current* episode count. A new episode gets
    ``counter + 1`` as its index, formatted as ``ep_{n:04d}``.

    **This is the ONLY function that generates Episode IDs in Torq.**

    Args:
        index_root: Directory containing the JSON index shards.

    Returns:
        Episode ID string in ``ep_{n:04d}`` format (e.g. ``"ep_0001"``).
    """
    manifest_path = index_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        n = manifest.get("episode_count", 0) + 1
    else:
        n = 1
    return f"ep_{n:04d}"


def update_index(episode_id: str, episode: Episode, index_root: Path) -> None:
    """Update all four JSON index shards with the newly saved episode.

    Shard files updated:
        - ``manifest.json`` — episode count, schema version, last updated timestamp
        - ``by_task.json`` — maps normalised task name → list of episode IDs
        - ``by_embodiment.json`` — maps normalised embodiment → list of episode IDs
        - ``quality.json`` — sorted ``[score_or_null, episode_id]`` pairs

    All four files are written atomically via ``_atomic_write_json()``.

    Args:
        episode_id: The ID assigned to this episode (e.g. ``"ep_0001"``).
        episode: The Episode whose metadata populates the index shards.
        index_root: Directory containing the JSON index shards.

    Raises:
        OSError: If any shard write fails (propagated from ``_atomic_write_json``).
    """
    index_root.mkdir(parents=True, exist_ok=True)

    meta = episode.metadata or {}

    # ── Load or create each shard ──
    manifest_path = index_root / "manifest.json"
    by_task_path = index_root / "by_task.json"
    by_embodiment_path = index_root / "by_embodiment.json"
    quality_path = index_root / "quality.json"

    manifest: dict = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"schema_version": SCHEMA_VERSION, "episode_count": 0}
    )
    by_task: dict[str, list[str]] = (
        json.loads(by_task_path.read_text(encoding="utf-8")) if by_task_path.exists() else {}
    )
    by_embodiment: dict[str, list[str]] = (
        json.loads(by_embodiment_path.read_text(encoding="utf-8"))
        if by_embodiment_path.exists()
        else {}
    )
    quality_list: list[list] = (
        json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else []
    )

    # ── Update by_task ──
    raw_task = meta.get("task", "")
    if raw_task:
        task_key = _normalise(str(raw_task))
        by_task.setdefault(task_key, [])
        if episode_id not in by_task[task_key]:
            by_task[task_key].append(episode_id)

    # ── Update by_embodiment ──
    raw_embodiment = meta.get("embodiment", "")
    if raw_embodiment:
        embodiment_key = _normalise(str(raw_embodiment))
        by_embodiment.setdefault(embodiment_key, [])
        if episode_id not in by_embodiment[embodiment_key]:
            by_embodiment[embodiment_key].append(episode_id)

    # ── Update quality (sorted ascending by score, nulls last) ──
    quality_score = getattr(episode.quality, "overall", None) if episode.quality else None
    # Remove any existing entry for this episode_id, then re-insert
    quality_list = [entry for entry in quality_list if entry[1] != episode_id]
    quality_list.append([quality_score, episode_id])
    # Sort: non-null scores first ascending, then nulls
    quality_list.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))

    # ── Update manifest ──
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["episode_count"] = manifest.get("episode_count", 0) + 1
    snapshot_count, experiment_count = _count_lineage_records(index_root)
    manifest["snapshot_count"] = snapshot_count
    manifest["experiment_count"] = experiment_count
    manifest["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Atomic write all shards ──
    _atomic_write_json(manifest, manifest_path)
    _atomic_write_json(by_task, by_task_path)
    _atomic_write_json(by_embodiment, by_embodiment_path)
    _atomic_write_json(quality_list, quality_path)

    logger.debug("Index updated for %s (total episodes: %d)", episode_id, manifest["episode_count"])


def read_snapshots(index_root: Path) -> dict:
    """Load snapshots.json from the given index directory.

    Args:
        index_root: Directory containing the JSON index shards.

    Returns:
        Dict mapping ``snapshot_id`` → snapshot record.
        Returns an empty dict if ``snapshots.json`` does not exist.
    """
    snapshots_path = index_root / "snapshots.json"
    if not snapshots_path.exists():
        return {}
    return json.loads(snapshots_path.read_text(encoding="utf-8"))


def write_snapshots(data: dict, index_root: Path) -> None:
    """Write snapshots.json atomically to the given index directory.

    Args:
        data: Dict mapping ``snapshot_id`` → snapshot record.
        index_root: Directory containing the JSON index shards.

    Raises:
        OSError: If the write or rename fails.
    """
    snapshots_path = index_root / "snapshots.json"
    _atomic_write_json(data, snapshots_path)


def read_experiments(index_root: Path) -> dict:
    """Load experiments.json from the given index directory.

    Args:
        index_root: Directory containing the JSON index shards.

    Returns:
        Dict mapping ``experiment_id`` → experiment record.
        Returns an empty dict if ``experiments.json`` does not exist.
    """
    experiments_path = index_root / "experiments.json"
    if not experiments_path.exists():
        return {}
    return json.loads(experiments_path.read_text(encoding="utf-8"))


def write_experiments(data: dict, index_root: Path) -> None:
    """Write experiments.json atomically to the given index directory.

    Args:
        data: Dict mapping ``experiment_id`` → experiment record.
        index_root: Directory containing the JSON index shards.

    Raises:
        OSError: If the write or rename fails.
    """
    experiments_path = index_root / "experiments.json"
    _atomic_write_json(data, experiments_path)


def read_manifest(index_root: Path) -> dict:
    """Load the manifest.json from the given index directory.

    Args:
        index_root: Directory containing the JSON index shards.

    Returns:
        Dict with keys: ``schema_version``, ``episode_count``, ``last_updated``.
        Returns an empty dict if manifest.json does not exist.
    """
    manifest_path = index_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def query_index(
    index_root: Path,
    *,
    task: str | list[str] | None = None,
    quality_min: float | None = None,
    quality_max: float | None = None,
    embodiment: str | list[str] | None = None,
) -> list[str]:
    """Return episode IDs matching the given filter criteria using set intersection.

    Reads the sharded JSON index to resolve each filter dimension independently,
    then intersects the results. This avoids a full Parquet scan — all filtering
    is done on the in-memory index shards.

    Algorithm:
        1. Load ``quality.json`` to build the universe of all known episode IDs.
        2. If ``task`` filter: load ``by_task.json``, union lookups → intersect.
        3. If ``embodiment`` filter: load ``by_embodiment.json``, union lookups → intersect.
        4. If ``quality_min`` or ``quality_max``: filter ``quality.json`` by range → intersect.
           Episodes with ``None`` quality score are excluded when any quality filter is active.
        5. Return sorted list of remaining IDs for determinism.

    Args:
        index_root: Directory containing the JSON index shards
            (e.g. ``Path(store_path) / "index"``).
        task: Task name(s) to filter by. A single string matches one task; a list
            performs OR logic (union) across all specified tasks. ``None`` = no filter.
        quality_min: Minimum quality score (inclusive). ``None`` = no lower bound.
        quality_max: Maximum quality score (inclusive). ``None`` = no upper bound.
        embodiment: Embodiment name(s) to filter by. Same OR logic as ``task``.
            ``None`` = no filter.

    Returns:
        Sorted list of episode ID strings matching all active filters.
        Returns an empty list if no episodes exist or no episodes match.
    """
    quality_path = index_root / "quality.json"
    if not quality_path.exists():
        return []

    quality_list: list[list] = json.loads(quality_path.read_text(encoding="utf-8"))
    universe: set[str] = {entry[1] for entry in quality_list}
    if not universe:
        return []

    current = universe.copy()

    # ── Task filter ──
    if task is not None:
        by_task_path = index_root / "by_task.json"
        by_task: dict[str, list[str]] = (
            json.loads(by_task_path.read_text(encoding="utf-8")) if by_task_path.exists() else {}
        )
        task_keys = [task] if isinstance(task, str) else task
        task_ids: set[str] = set()
        for t in task_keys:
            task_ids |= set(by_task.get(_normalise(t), []))
        current &= task_ids

    # ── Embodiment filter ──
    if embodiment is not None:
        by_embodiment_path = index_root / "by_embodiment.json"
        by_embodiment: dict[str, list[str]] = (
            json.loads(by_embodiment_path.read_text(encoding="utf-8"))
            if by_embodiment_path.exists()
            else {}
        )
        embodiment_keys = [embodiment] if isinstance(embodiment, str) else embodiment
        embodiment_ids: set[str] = set()
        for e in embodiment_keys:
            embodiment_ids |= set(by_embodiment.get(_normalise(e), []))
        current &= embodiment_ids

    # ── Quality range filter (binary search on pre-sorted list) ──
    if quality_min is not None or quality_max is not None:
        # quality_list is sorted ascending [score, ep_id], nulls last.
        # Separate scored entries and build a parallel scores list for bisect.
        scored = [(entry[0], entry[1]) for entry in quality_list if entry[0] is not None]
        scores = [s for s, _ in scored]
        lo = bisect.bisect_left(scores, quality_min) if quality_min is not None else 0
        hi = bisect.bisect_right(scores, quality_max) if quality_max is not None else len(scores)
        quality_ids: set[str] = {ep_id for _, ep_id in scored[lo:hi]}
        current &= quality_ids

    return sorted(current)
