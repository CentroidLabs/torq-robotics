"""LeRobot v3.0 dataset ingestion for the Torq SDK.

Parses LeRobot format datasets (Parquet + MP4) and returns canonical Episodes.
Requires ``meta/info.json`` to be present for feature schema discovery.

Timestamps in LeRobot are float seconds — converted to ``np.int64`` nanoseconds
at the ingest boundary (single conversion point).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from torq.episode import Episode
from torq.errors import TorqIngestError
from torq.media.image_sequence import ImageSequence

__all__ = ["ingest"]

logger = logging.getLogger(__name__)


def ingest(path: str | Path) -> list[Episode]:
    """Ingest a LeRobot v3.0 dataset directory and return one Episode per episode index.

    Args:
        path: Root directory of the LeRobot dataset (must contain ``meta/info.json``).

    Returns:
        List of Episode objects sorted by episode_index. Empty list if the
        dataset has no episodes or no Parquet data files.

    Raises:
        TorqIngestError: If ``meta/info.json`` is missing or malformed, or if
            required data files cannot be read.
    """
    root = Path(path)
    if not root.is_dir():
        raise TorqIngestError(
            f"Expected a LeRobot dataset directory, but '{root}' is not a directory. "
            f"Provide the root directory of the LeRobot dataset (containing meta/info.json)."
        )

    info = _load_info(root)
    fps: float = info.get("fps", 50)
    robot_type: str = info.get("robot_type", "")
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise TorqIngestError(
            f"'meta/info.json' at '{root}' has invalid 'features' field: expected a dict, "
            f"got {type(features).__name__}. Check the info.json schema."
        )

    # Identify camera feature keys (dtype "video"/"image" or key contains "images")
    camera_keys = {
        k
        for k, v in features.items()
        if isinstance(v, dict) and (v.get("dtype") in ("video", "image") or "images" in k)
    }

    # Load all Parquet data, grouped by episode_index
    combined = _load_parquet_chunks(root)
    if combined is None:
        return []

    # Group rows by episode_index
    try:
        episode_indices = combined.column("episode_index").to_pylist()
    except KeyError:
        raise TorqIngestError(
            f"Parquet data in '{root / 'data'}' is missing required 'episode_index' column. "
            f"LeRobot v3.0 Parquet files must contain an 'episode_index' column. "
            f"Available columns: {combined.column_names}"
        ) from None
    unique_episodes = sorted(set(episode_indices))

    if not unique_episodes:
        logger.warning("LeRobot dataset at '%s' has no episodes.", root)
        return []

    # Discover video files: per-episode (flat) and per-chunk (nested) layouts
    per_episode_videos, per_chunk_videos = _discover_videos(root)

    episodes: list[Episode] = []
    for ep_idx in unique_episodes:
        mask = pc.equal(combined.column("episode_index"), ep_idx)
        ep_table = combined.filter(mask)
        episode = _build_episode(
            ep_table,
            ep_idx,
            root,
            camera_keys,
            robot_type,
            fps,
            per_episode_videos=per_episode_videos,
            per_chunk_videos=per_chunk_videos,
            unique_episodes=unique_episodes,
        )
        episodes.append(episode)

    return episodes


def _load_info(root: Path) -> dict:
    """Load and parse meta/info.json from a LeRobot dataset.

    Args:
        root: Dataset root directory.

    Returns:
        Parsed JSON dict.

    Raises:
        TorqIngestError: If info.json is missing or malformed.
    """
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise TorqIngestError(
            f"LeRobot dataset at '{root}' is missing 'meta/info.json'. "
            f"This file is required for feature schema discovery. "
            f"Ensure the dataset follows LeRobot v3.0 format with a 'meta/' directory."
        )
    try:
        return json.loads(info_path.read_text())
    except json.JSONDecodeError as exc:
        raise TorqIngestError(
            f"'meta/info.json' at '{root}' is malformed JSON: {exc}. "
            f"Validate the file with json.load() to see the exact error."
        ) from exc


def _is_list_type(col_type: pa.DataType) -> bool:
    """Check if a PyArrow type is any list variant (fixed_size_list, list, large_list)."""
    return (
        pa.types.is_fixed_size_list(col_type)
        or pa.types.is_list(col_type)
        or pa.types.is_large_list(col_type)
    )


def _column_to_array(col: pa.ChunkedArray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert a PyArrow column to a numpy array, handling list-type and scalar columns.

    Args:
        col: PyArrow chunked array (single column from a table).
        dtype: Target numpy dtype for the output array.

    Returns:
        np.ndarray: Shape ``[T, D]`` for list-type columns, ``[T]`` for scalar columns.

    Raises:
        TorqIngestError: If a variable-length list column contains ragged (unequal-length) rows,
            or if the column contains null values.
    """
    if col.null_count > 0:
        raise TorqIngestError(
            f"Column contains {col.null_count} null value(s). "
            f"LeRobot datasets should not have nulls in state/action columns. "
            f"Check the source data for missing values."
        )
    col_type = col.type
    if _is_list_type(col_type):
        try:
            result = np.array(col.to_pylist(), dtype=dtype)
        except ValueError as exc:
            raise TorqIngestError(
                f"List-type column contains ragged (unequal-length) lists: {exc}"
            ) from exc
        if result.ndim != 2:
            raise TorqIngestError(
                f"Expected 2D array from list-type column, got shape {result.shape}. "
                f"Column may contain ragged (unequal-length) lists."
            )
        return result
    return col.to_numpy().astype(dtype, copy=True)


def _load_parquet_chunks(root: Path) -> pa.Table | None:
    """Load and concatenate all Parquet chunk files from data/chunk-*/.

    Args:
        root: Dataset root directory.

    Returns:
        Combined PyArrow table, or None if no data files found.
    """
    data_dir = root / "data"
    if not data_dir.exists():
        logger.warning("LeRobot dataset at '%s' has no 'data/' directory.", root)
        return None

    chunk_files = sorted(data_dir.glob("chunk-*/*.parquet"))
    if not chunk_files:
        logger.warning("LeRobot dataset at '%s' has no Parquet data files.", root)
        return None

    tables = []
    for chunk_path in chunk_files:
        try:
            tables.append(pq.read_table(str(chunk_path)))
        except (pa.ArrowInvalid, pa.ArrowIOError, OSError) as exc:
            logger.warning("Skipping unreadable Parquet chunk '%s': %s", chunk_path, exc)

    if not tables:
        return None
    try:
        return pa.concat_tables(tables)
    except pa.ArrowInvalid as exc:
        raise TorqIngestError(
            f"Parquet chunks in '{data_dir}' have incompatible schemas and cannot be "
            f"concatenated. Ensure all chunk files have the same columns: {exc}"
        ) from exc


def _discover_videos(
    root: Path,
) -> tuple[dict[tuple[str, int], Path], dict[str, list[Path]]]:
    """Discover video files in both flat and nested layouts.

    Returns:
        Tuple of:
        - per_episode_map: ``{(short_key, ep_idx): path}`` from flat layout
          (``videos/chunk-NNN/{key}_episode_NNN.mp4``)
        - per_chunk_map: ``{short_key: [path, ...]}`` from nested layout
          (``videos/{key}/chunk-NNN/file-NNN.mp4``)
    """
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return {}, {}

    per_episode_map: dict[tuple[str, int], Path] = {}
    per_chunk_map: dict[str, list[Path]] = {}

    # Flat layout: videos/chunk-NNN/{key}_episode_NNN.mp4
    for mp4 in sorted(videos_dir.glob("chunk-*/*.mp4")):
        stem = mp4.stem
        if "_episode_" not in stem:
            continue
        key_part, ep_part = stem.rsplit("_episode_", 1)
        try:
            ep_idx = int(ep_part)
        except ValueError:
            continue
        short_key = key_part.split(".")[-1]
        existing = per_episode_map.get((short_key, ep_idx))
        if existing is not None:
            logger.warning(
                "Video short-key collision: '%s' maps to both '%s' and '%s' for episode %d. "
                "Using the later file. Consider using distinct camera names.",
                short_key,
                existing.name,
                mp4.name,
                ep_idx,
            )
        per_episode_map[(short_key, ep_idx)] = mp4

    # Nested layout: videos/{video_key}/chunk-NNN/file-NNN.mp4
    for mp4 in sorted(videos_dir.glob("*/chunk-*/*.mp4")):
        key_dir = mp4.parent.parent.name  # e.g. "observation.images.top"
        # Skip if key_dir looks like "chunk-NNN" (would match flat layout)
        if key_dir.startswith("chunk-"):
            continue
        short_key = key_dir.split(".")[-1]
        existing_chunk = per_chunk_map.get(short_key)
        if existing_chunk and mp4.parent.parent != existing_chunk[0].parent.parent:
            logger.warning(
                "Video short-key collision: '%s' maps to multiple video keys in nested layout. "
                "Consider using distinct camera names.",
                short_key,
            )
        per_chunk_map.setdefault(short_key, []).append(mp4)

    return per_episode_map, per_chunk_map


def _group_observation_columns(table: pa.Table, camera_keys: set[str]) -> dict[str, list[str]]:
    """Group observation columns by feature prefix, excluding camera features.

    Handles both scalar per-dimension columns (``observation.state_0``, ...)
    and list-type columns (``observation.state`` as ``fixed_size_list``).

    Args:
        table: PyArrow table to inspect for column names and types.
        camera_keys: Camera feature keys from info.json.

    Returns:
        Dict mapping short observation name to sorted list of column names.
    """
    columns = set(table.column_names)
    obs_groups: dict[str, list[str]] = {}
    captured_prefixes: set[str] = set()

    # Pass 1: scalar per-dimension columns (observation.state_0, observation.state_1, ...)
    for c in sorted(columns):
        if not c.startswith("observation."):
            continue
        if any(c == ck or c.startswith(ck + "_") for ck in camera_keys):
            continue
        m = re.match(r"^(.+)_(\d+)$", c)
        if m:
            prefix = m.group(1)
            obs_groups.setdefault(prefix, []).append(c)
            captured_prefixes.add(prefix)

    # Pass 2: list-type columns (observation.state as fixed_size_list)
    for c in sorted(columns):
        if not c.startswith("observation."):
            continue
        if any(c == ck or c.startswith(ck + "_") for ck in camera_keys):
            continue
        if c in captured_prefixes:
            # Both scalar _N columns AND list-type column exist for same prefix
            # Prefer list-type (canonical v3.0 format)
            col_type = table.column(c).type
            if _is_list_type(col_type):
                logger.warning(
                    "Column '%s' exists as both scalar _N columns and list-type. "
                    "Using list-type (canonical v3.0 format).",
                    c,
                )
                obs_groups[c] = [c]  # overwrite scalar group with single list column
            continue
        # Skip columns already captured as dimension columns
        if re.match(r"^(.+)_(\d+)$", c):
            continue
        # Check if this is a list-type or plain scalar column not captured by Pass 1
        col_type = table.column(c).type
        if _is_list_type(col_type):
            obs_groups.setdefault(c, []).append(c)
        elif pa.types.is_floating(col_type) or pa.types.is_integer(col_type):
            # Plain scalar column without _N suffix (e.g. observation.velocity)
            obs_groups.setdefault(c, []).append(c)

    # Convert prefixes to short names: "observation.state" → "state"
    result: dict[str, list[str]] = {}
    for prefix, cols in obs_groups.items():
        short_name = prefix.split("observation.", 1)[1]
        result[short_name] = sorted(cols)
    return result


def _build_episode(
    table: pa.Table,
    ep_idx: int,
    root: Path,
    camera_keys: set[str],
    robot_type: str,
    fps: float,
    *,
    per_episode_videos: dict[tuple[str, int], Path],
    per_chunk_videos: dict[str, list[Path]],
    unique_episodes: list[int],
) -> Episode:
    """Build one Episode from a filtered PyArrow table (rows for one episode).

    Args:
        table: PyArrow table containing rows for a single episode.
        ep_idx: Episode index.
        root: Dataset root path (used as source_path).
        camera_keys: Feature keys identified as camera/video observations.
        robot_type: Robot type from info.json (used in metadata).
        fps: Frames per second from info.json (fallback for timestamp synthesis).
        per_episode_videos: Flat layout video map ``{(short_key, ep_idx): path}``.
        per_chunk_videos: Nested layout video map ``{short_key: [path, ...]}``.
        unique_episodes: All episode indices in the dataset (for multi-episode warning).

    Returns:
        Episode with observations, actions, and timestamps.
    """
    columns = set(table.column_names)

    # ── Timestamps: float seconds → np.int64 nanoseconds ──
    if "timestamp" in columns:
        ts_s = table.column("timestamp").to_numpy().astype(np.float64)
        timestamps = (ts_s * 1e9).astype(np.int64)
    else:
        # Synthesise at fps if timestamp column absent
        n_rows = len(table)
        step_ns = int(1e9 / fps)
        timestamps = np.arange(n_rows, dtype=np.int64) * step_ns

    # ── task_index (optional) ──
    metadata: dict = {"task": "", "embodiment": robot_type}
    if "task_index" in columns:
        task_idx_values = table.column("task_index").to_pylist()
        unique_task_indices = set(task_idx_values)
        if len(unique_task_indices) > 1:
            from collections import Counter

            logger.warning(
                "Episode %d has mixed task_index values %s — using most common.",
                ep_idx,
                unique_task_indices,
            )
            metadata["task_index"] = Counter(task_idx_values).most_common(1)[0][0]
        elif task_idx_values:
            metadata["task_index"] = task_idx_values[0]

    # ── Actions (prefix-based column discovery, list-type aware) ──
    action_cols = sorted(c for c in columns if c == "action" or re.match(r"^action_\d+$", c))
    # If both list-type "action" and scalar "action_N" exist, prefer list-type
    if "action" in action_cols and _is_list_type(table.column("action").type):
        if len(action_cols) > 1:
            logger.warning(
                "Column 'action' exists as both scalar _N columns and list-type. "
                "Using list-type (canonical v3.0 format).",
            )
        actions = _column_to_array(table.column("action"))
    elif action_cols:
        if len(action_cols) == 1 and _is_list_type(table.column(action_cols[0]).type):
            actions = _column_to_array(table.column(action_cols[0]))
        else:
            # Multiple scalar columns → stack into [T, D]
            actions = np.column_stack([_column_to_array(table.column(c)) for c in action_cols])
    else:
        actions = np.empty((len(timestamps), 0), dtype=np.float32)

    # ── Observations — discover ALL observation.* non-camera columns ──
    observations: dict[str, np.ndarray | ImageSequence] = {}
    obs_groups = _group_observation_columns(table, camera_keys)
    for short_name, dim_cols in obs_groups.items():
        if len(dim_cols) == 1 and _is_list_type(table.column(dim_cols[0]).type):
            # Single list-type column → already [T, D]
            observations[short_name] = _column_to_array(table.column(dim_cols[0]))
        else:
            # Multiple scalar columns → stack into [T, D]
            observations[short_name] = np.column_stack(
                [_column_to_array(table.column(c)) for c in dim_cols]
            )

    # ── Camera observations: per-episode first, per-chunk fallback ──
    for (short_key, vidx), mp4_path in per_episode_videos.items():
        if vidx == ep_idx:
            if short_key in observations and not isinstance(observations[short_key], ImageSequence):
                logger.warning(
                    "Camera '%s' overwrites existing observation array for episode %d."
                    " This may indicate a short-key collision between camera and"
                    " non-camera features.",
                    short_key,
                    ep_idx,
                )
            observations[short_key] = ImageSequence(mp4_path)

    for short_key, mp4_paths in per_chunk_videos.items():
        if short_key not in observations:  # don't overwrite per-episode match
            # Concatenated MP4 — assign full file, log warning
            observations[short_key] = ImageSequence(mp4_paths[0])
            if len(unique_episodes) > 1:
                logger.warning(
                    "Camera '%s' uses concatenated MP4 ('%s') containing multiple episodes. "
                    "Frame-level slicing is not yet supported — ImageSequence points to full file.",
                    short_key,
                    mp4_paths[0].name,
                )

    # Warn if camera features from info.json have no matching video
    for cam_key in camera_keys:
        short_cam = cam_key.split(".")[-1]
        if short_cam not in observations:
            logger.warning(
                "Camera feature '%s' in info.json has no matching MP4 for episode %d.",
                cam_key,
                ep_idx,
            )

    return Episode(
        episode_id="",
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        source_path=root,
        metadata=metadata,
    )
