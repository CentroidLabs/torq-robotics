"""HDF5 (robomimic format) ingestion for the Torq SDK.

Parses robomimic-format HDF5 files and returns canonical Episode objects.
Supports joint_pos, joint_vel, actions, and image observations.

Timestamps are synthesised at 50 Hz (no timestamp data in robomimic format).
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from torq.episode import Episode
from torq.errors import TorqIngestError

__all__ = ["ingest"]

logger = logging.getLogger(__name__)

# Image observation key patterns — these get wrapped as _InMemoryFrames
_IMAGE_KEY_SUFFIXES = ("_image", "_img", "_rgb", "_depth")
_KNOWN_IMAGE_KEYS = {"agentview_image", "robot0_eye_in_hand", "eye_in_hand_image"}

# Timestamp synthesis: 50 Hz (no timestamps in robomimic format)
_SYNTH_HZ = 50
_SYNTH_STEP_NS: int = int(1e9 / _SYNTH_HZ)  # 20_000_000 ns


class _InMemoryFrames:
    """Minimal image sequence backed by an in-memory numpy array.

    Satisfies the same interface as ImageSequence (``.frames`` property)
    without requiring a video file. Used for robomimic image observations
    which are already fully decoded in the HDF5 file.

    Args:
        frames: RGB frames array of shape ``[T, H, W, C]``, dtype ``uint8``.
    """

    def __init__(self, frames: np.ndarray) -> None:
        if frames.ndim != 4:
            raise TorqIngestError(
                f"_InMemoryFrames requires 4D array [T, H, W, C], got {frames.ndim}D "
                f"with shape {frames.shape}. Check that the image observation key is correct "
                f"and the HDF5 dataset has shape [T, H, W, C]."
            )
        self._frames = frames.copy()
        self._frames.flags.writeable = False

    @property
    def frames(self) -> np.ndarray:
        """Return frames array of shape ``[T, H, W, C]``, dtype ``uint8``."""
        return self._frames

    def __repr__(self) -> str:
        t, h, w, c = self._frames.shape
        return f"_InMemoryFrames(shape=[{t}, {h}, {w}, {c}])"


def _is_image_key(key: str) -> bool:
    """Check whether an observation key corresponds to image data."""
    return key in _KNOWN_IMAGE_KEYS or any(key.endswith(s) for s in _IMAGE_KEY_SUFFIXES)


def ingest(path: str | Path) -> list[Episode]:
    """Ingest a robomimic-format HDF5 file and return one Episode per demo.

    Args:
        path: Path to the ``.hdf5`` file.

    Returns:
        List of Episode objects, one per ``/data/demo_*`` group, sorted by
        demo index. Empty list if no demo groups are found.

    Raises:
        TorqIngestError: If the file cannot be opened, is corrupt/truncated,
            or does not have the expected ``/data/demo_*`` group structure.
    """
    path = Path(path)

    try:
        f = h5py.File(path, "r")
    except Exception as exc:
        raise TorqIngestError(
            f"Cannot open HDF5 file '{path}': {exc}. "
            f"The file may be truncated or corrupt. "
            f"Validate the file independently with h5py.File(path, 'r')."
        ) from exc

    with f:
        if "data" not in f:
            raise TorqIngestError(
                f"HDF5 file '{path}' is missing the '/data' group. "
                f"Expected robomimic format with '/data/demo_*' groups. "
                f"Check that the file was created with robomimic's data collection pipeline."
            )

        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda k: int(k.split("_")[1]),
        )

        if not demo_keys:
            logger.warning(
                "HDF5 file '%s' has a '/data' group but no 'demo_*' sub-groups. "
                "Returning empty list.",
                path,
            )
            return []

        episodes: list[Episode] = []
        for demo_key in demo_keys:
            episode = _ingest_demo(f, demo_key, path)
            episodes.append(episode)

    return episodes


def _ingest_demo(f: h5py.File, demo_key: str, source_path: Path) -> Episode:
    """Extract a single demo group as an Episode.

    Args:
        f: Open h5py.File handle.
        demo_key: Demo group name (e.g. ``"demo_0"``).
        source_path: Original file path for provenance.

    Returns:
        Episode with observations, actions, and synthesised timestamps.
    """
    demo = f["data"][demo_key]

    # ── Validate required keys ──
    if "actions" not in demo:
        raise TorqIngestError(
            f"HDF5 demo group '/data/{demo_key}' in '{source_path}' is missing required "
            f"key 'actions'. Expected robomimic format with 'actions' dataset and 'obs/' group."
        )
    if "obs" not in demo:
        raise TorqIngestError(
            f"HDF5 demo group '/data/{demo_key}' in '{source_path}' is missing required "
            f"key 'obs'. Expected robomimic format with 'actions' dataset and 'obs/' group."
        )

    # ── Actions ──
    actions = np.array(demo["actions"], dtype=np.float32)
    t_len = len(actions)

    # ── Synthesise timestamps at 50Hz (robomimic has no timestamps) ──
    timestamps = np.arange(t_len, dtype=np.int64) * _SYNTH_STEP_NS

    # ── Observations ──
    # Values are np.ndarray for continuous obs or _InMemoryFrames for image obs.
    # Both satisfy the duck-typing contract (storage uses hasattr(x, 'frames')).
    observations: dict[str, np.ndarray | _InMemoryFrames] = {}
    obs_group = demo["obs"]

    for key in obs_group.keys():
        data = np.array(obs_group[key])
        if _is_image_key(key):
            observations[key] = _InMemoryFrames(data.astype(np.uint8))
        else:
            observations[key] = data.astype(np.float32)

    return Episode(
        episode_id="",
        observations=observations,
        actions=actions,
        timestamps=timestamps,
        source_path=source_path,
        metadata={"task": "", "embodiment": ""},
    )
