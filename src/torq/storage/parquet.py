"""Parquet read/write for Episode non-image data.

Column name templates defined here are the single source of truth for
the on-disk Parquet schema. All other storage modules import these constants.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from torq.episode import Episode
from torq.errors import TorqStorageError

__all__ = [
    "save_parquet",
    "load_parquet",
    # Column name constants — import these instead of hard-coding strings
    "COL_TIMESTAMP_NS",
    "COL_OBS_PREFIX",
    "COL_ACTION_PREFIX",
    "COL_METADATA_SUCCESS",
    "COL_METADATA_TASK",
    "COL_METADATA_EMBODIMENT",
    "COL_EPISODE_ID",
    "COL_SOURCE_PATH",
]

logger = logging.getLogger(__name__)

# ── Column name templates — single source of truth for parquet schema ──
COL_TIMESTAMP_NS = "timestamp_ns"  # np.int64 nanoseconds
COL_OBS_PREFIX = "obs_"  # e.g. obs_joint_pos_0, obs_joint_vel_0
COL_ACTION_PREFIX = "action_"  # e.g. action_0, action_1 ...
COL_METADATA_SUCCESS = "metadata_success"  # bool
COL_METADATA_TASK = "metadata_task"  # str
COL_METADATA_EMBODIMENT = "metadata_embodiment"  # str
COL_EPISODE_ID = "episode_id"  # str
COL_SOURCE_PATH = "source_path"  # str (Path serialised)


def save_parquet(episode: Episode, episodes_dir: Path) -> Path:
    """Write Episode (non-image observations, actions, timestamps) to a Parquet file.

    The output path is ``{episodes_dir}/{episode_id}.parquet``.
    Uses an atomic write pattern: write to a ``.parquet.tmp`` temp file,
    then rename with ``os.replace()`` so a partial write never corrupts the index.

    Args:
        episode: The Episode to persist.
        episodes_dir: Directory to write the ``.parquet`` file into.

    Returns:
        Path to the written ``.parquet`` file.

    Note:
        **R1 metadata limitation**: Only three metadata keys survive the round-trip —
        ``task`` (str), ``embodiment`` (str), and ``success`` (bool). All other
        keys in ``episode.metadata`` are silently dropped. Full metadata
        serialisation (e.g. a JSON column) is planned for R2.

    Raises:
        TorqStorageError: If the write fails for any reason.
    """
    parquet_path = episodes_dir / f"{episode.episode_id}.parquet"
    tmp_path = parquet_path.with_suffix(".parquet.tmp")

    columns: dict[str, pa.Array] = {}

    # ── Timestamps ──
    columns[COL_TIMESTAMP_NS] = pa.array(episode.timestamps, type=pa.int64())

    # ── Observations (skip image data — those go to video.py) ──
    # Duck-typing: any obs with a .frames property is image data (ImageSequence
    # or _InMemoryFrames). Skip these — they are saved as MP4 by _impl.save().
    for key, arr in episode.observations.items():
        if hasattr(arr, "frames"):
            continue
        arr_np = np.asarray(arr)
        if arr_np.ndim == 1:
            arr_np = arr_np[:, np.newaxis]
        for i in range(arr_np.shape[1]):
            col_name = f"{COL_OBS_PREFIX}{key}_{i}"
            columns[col_name] = pa.array(arr_np[:, i].astype(np.float32), type=pa.float32())

    # ── Actions ──
    actions_np = np.asarray(episode.actions)
    if actions_np.ndim == 1:
        actions_np = actions_np[:, np.newaxis]
    for i in range(actions_np.shape[1]):
        columns[f"{COL_ACTION_PREFIX}{i}"] = pa.array(
            actions_np[:, i].astype(np.float32), type=pa.float32()
        )

    # ── Metadata scalar columns ──
    n_steps = len(episode.timestamps)
    meta = episode.metadata or {}

    task_val = meta.get("task", "")
    embodiment_val = meta.get("embodiment", "")
    success_val = meta.get("success", None)

    columns[COL_METADATA_TASK] = pa.array([str(task_val)] * n_steps, type=pa.string())
    columns[COL_METADATA_EMBODIMENT] = pa.array([str(embodiment_val)] * n_steps, type=pa.string())
    if success_val is not None:
        columns[COL_METADATA_SUCCESS] = pa.array([bool(success_val)] * n_steps, type=pa.bool_())

    # ── Episode ID and source path ──
    columns[COL_EPISODE_ID] = pa.array([episode.episode_id] * n_steps, type=pa.string())
    columns[COL_SOURCE_PATH] = pa.array([str(episode.source_path)] * n_steps, type=pa.string())

    table = pa.table(columns)

    try:
        pq.write_table(table, str(tmp_path), compression="snappy")
        os.replace(tmp_path, parquet_path)
    except Exception as exc:
        # Clean up temp file if it exists
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise TorqStorageError(
            f"Failed to write Parquet file '{parquet_path}': {exc}. "
            "Check that the directory exists and you have write permissions."
        ) from exc

    logger.debug("Saved Episode %s → %s", episode.episode_id, parquet_path)
    return parquet_path


def load_parquet(episode_id: str, dataset_root: Path) -> Episode:
    """Load an Episode from a Parquet file.

    Reconstructs observations, actions, and timestamps from the column template
    conventions in this module.

    Args:
        episode_id: Episode ID (e.g. ``"ep_0001"``).
        dataset_root: Root dataset directory containing an ``episodes/`` sub-directory.

    Returns:
        Episode with all numeric fields reconstructed.

    Raises:
        TorqStorageError: If the file is missing or cannot be parsed.
    """
    parquet_path = dataset_root / "episodes" / f"{episode_id}.parquet"
    if not parquet_path.exists():
        raise TorqStorageError(
            f"Parquet file not found: '{parquet_path}'. "
            f"Ensure episode '{episode_id}' was saved to '{dataset_root}'."
        )

    try:
        table = pq.read_table(str(parquet_path))
    except Exception as exc:
        raise TorqStorageError(
            f"Failed to read Parquet file '{parquet_path}': {exc}. The file may be corrupt."
        ) from exc

    df = table.to_pandas()

    # ── Timestamps ──
    timestamps = df[COL_TIMESTAMP_NS].to_numpy(dtype=np.int64)

    # ── Observations ──
    obs_columns = [c for c in df.columns if c.startswith(COL_OBS_PREFIX)]
    # Group by observation key (strip prefix and trailing _i)
    obs_keys: dict[str, list[tuple[int, str]]] = {}
    for col in obs_columns:
        # col format: obs_{key}_{i}  — find last _ as the index separator
        suffix = col[len(COL_OBS_PREFIX) :]  # "{key}_{i}"
        last_underscore = suffix.rfind("_")
        key = suffix[:last_underscore]
        idx = int(suffix[last_underscore + 1 :])
        obs_keys.setdefault(key, []).append((idx, col))

    observations: dict[str, np.ndarray] = {}
    for key, idx_cols in obs_keys.items():
        idx_cols.sort()  # sort by column index
        arrays = [df[col].to_numpy(dtype=np.float32) for _, col in idx_cols]
        observations[key] = np.stack(arrays, axis=1)  # shape [T, D]

    # ── Actions ──
    action_columns = sorted(
        [
            (int(c[len(COL_ACTION_PREFIX) :]), c)
            for c in df.columns
            if c.startswith(COL_ACTION_PREFIX)
        ]
    )
    if action_columns:
        action_arrays = [df[col].to_numpy(dtype=np.float32) for _, col in action_columns]
        actions = np.stack(action_arrays, axis=1)  # shape [T, action_dim]
    else:
        actions = np.empty((len(timestamps), 0), dtype=np.float32)

    # ── Metadata (take first row — scalar broadcast) ──
    metadata: dict = {}
    if COL_METADATA_TASK in df.columns:
        task_val = df[COL_METADATA_TASK].iloc[0]
        if task_val:
            metadata["task"] = task_val
    if COL_METADATA_EMBODIMENT in df.columns:
        emb_val = df[COL_METADATA_EMBODIMENT].iloc[0]
        if emb_val:
            metadata["embodiment"] = emb_val
    if COL_METADATA_SUCCESS in df.columns:
        metadata["success"] = bool(df[COL_METADATA_SUCCESS].iloc[0])

    # ── Provenance ──
    source_path_str = df[COL_SOURCE_PATH].iloc[0] if COL_SOURCE_PATH in df.columns else ""

    return Episode(
        episode_id=episode_id,
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        source_path=Path(source_path_str) if source_path_str else parquet_path,
        metadata=metadata,
    )
