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

    # Discover video files: {(short_camera_key, episode_idx) → Path}
    video_map = _discover_videos(root)

    episodes: list[Episode] = []
    for ep_idx in unique_episodes:
        mask = pc.equal(combined.column("episode_index"), ep_idx)
        ep_table = combined.filter(mask)
        episode = _build_episode(ep_table, ep_idx, root, camera_keys, video_map, robot_type, fps)
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
        except Exception as exc:
            logger.warning("Skipping unreadable Parquet chunk '%s': %s", chunk_path, exc)

    if not tables:
        return None
    return pa.concat_tables(tables)


def _discover_videos(root: Path) -> dict[tuple[str, int], Path]:
    """Map ``(short_camera_key, episode_idx)`` to MP4 file paths.

    Parses filenames like ``observation.images.top_episode_000000.mp4``
    and extracts the short key (``top``) and episode index (``0``).

    Args:
        root: Dataset root directory.

    Returns:
        Dict mapping ``(short_key, episode_idx)`` to the MP4 path.
    """
    videos_dir = root / "videos"
    if not videos_dir.exists():
        return {}

    video_map: dict[tuple[str, int], Path] = {}
    for mp4 in sorted(videos_dir.glob("chunk-*/*.mp4")):
        stem = mp4.stem  # e.g. "observation.images.top_episode_000000"
        if "_episode_" not in stem:
            continue
        key_part, ep_part = stem.rsplit("_episode_", 1)
        try:
            ep_idx = int(ep_part)
        except ValueError:
            continue
        # Normalise: "observation.images.top" → "top"
        short_key = key_part.split(".")[-1]
        video_map[(short_key, ep_idx)] = mp4
    return video_map


def _group_observation_columns(columns: set[str], camera_keys: set[str]) -> dict[str, list[str]]:
    """Group ``observation.*`` Parquet columns by feature prefix, excluding camera features.

    Args:
        columns: Set of all column names in the table.
        camera_keys: Camera feature keys from info.json (e.g., ``observation.images.top``).

    Returns:
        Dict mapping short observation name (e.g., ``state``) to sorted list of
        dimension columns (e.g., ``[observation.state_0, ..., observation.state_13]``).
    """
    obs_groups: dict[str, list[str]] = {}
    for c in sorted(columns):
        if not c.startswith("observation."):
            continue
        # Skip columns belonging to camera features
        if any(c == ck or c.startswith(ck + "_") for ck in camera_keys):
            continue
        # Determine feature prefix by stripping trailing _N dimension suffix
        m = re.match(r"^(.+)_(\d+)$", c)
        prefix = m.group(1) if m else c
        obs_groups.setdefault(prefix, []).append(c)

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
    video_map: dict[tuple[str, int], Path],
    robot_type: str,
    fps: float,
) -> Episode:
    """Build one Episode from a filtered PyArrow table (rows for one episode).

    Args:
        table: PyArrow table containing rows for a single episode.
        ep_idx: Episode index.
        root: Dataset root path (used as source_path).
        camera_keys: Feature keys identified as camera/video observations.
        video_map: Mapping of ``(short_key, ep_idx)`` to MP4 paths.
        robot_type: Robot type from info.json (used in metadata).
        fps: Frames per second from info.json (fallback for timestamp synthesis).

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

    # ── Actions (prefix-based column discovery) ──
    action_cols = sorted(c for c in columns if c == "action" or c.startswith("action_"))
    if action_cols:
        actions = np.column_stack(
            [table.column(c).to_numpy().astype(np.float32) for c in action_cols]
        )
    else:
        actions = np.empty((len(timestamps), 0), dtype=np.float32)

    # ── Observations — discover ALL observation.* non-camera columns ──
    observations: dict[str, np.ndarray | ImageSequence] = {}
    obs_groups = _group_observation_columns(columns, camera_keys)
    for short_name, dim_cols in obs_groups.items():
        observations[short_name] = np.column_stack(
            [table.column(c).to_numpy().astype(np.float32) for c in dim_cols]
        )

    # ── Camera observations (lazy ImageSequence) ──
    for (short_key, vidx), mp4_path in video_map.items():
        if vidx == ep_idx:
            observations[short_key] = ImageSequence(mp4_path)

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
        metadata={"task": "", "embodiment": robot_type},
    )
