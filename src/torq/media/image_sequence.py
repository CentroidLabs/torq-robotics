"""Lazy-loading image sequence backed by an MP4 file.

Frames are decoded from disk only on first access to the ``.frames`` property.
Decoded frames are cached in memory; subsequent accesses return the same array.

Requires the ``[vision]`` extra: ``pip install torq-robotics[vision]``
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from torq.errors import TorqImportError

__all__ = ["ImageSequence"]

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
_LARGE_FRAME_THRESHOLD = 10_000


class ImageSequence:
    """A lazy-loading sequence of RGB frames backed by an MP4 file.

    Args:
        path: Path to the MP4 file (str or pathlib.Path).

    Example:
        >>> seq = ImageSequence("videos/ep_0001.mp4")
        >>> frames = seq.frames   # decoded on first access
        >>> frames.shape          # (T, H, W, C)
        (120, 480, 640, 3)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: np.ndarray | None = None
        if self._path.suffix.lower() not in _VIDEO_EXTENSIONS:
            logger.warning(
                "ImageSequence initialized with non-standard video extension '%s'. "
                "Expected one of: %s. Decoding may fail.",
                self._path.suffix or "(none)",
                ", ".join(sorted(_VIDEO_EXTENSIONS)),
            )

    @property
    def frames(self) -> np.ndarray:
        """Decoded frames as an array of shape [T, H, W, C], dtype uint8.

        Frames are decoded from disk on first access and cached in memory.

        Returns:
            np.ndarray: RGB frames, shape [T, H, W, C], dtype uint8.

        Raises:
            TorqImportError: If imageio is not installed.
            TorqStorageError: If the file cannot be read.
        """
        if self._cache is not None:
            return self._cache
        self._cache = self._load_frames()
        return self._cache

    def _load_frames(self) -> np.ndarray:
        """Load and decode all frames from the MP4 file using the imageio v3 API."""
        try:
            import imageio.v3 as iio  # noqa: PLC0415 — intentional lazy import
        except ImportError:
            raise TorqImportError(
                "imageio is required to decode video frames. "
                "Install it with: pip install torq-robotics[vision]"
            ) from None

        from torq.errors import TorqStorageError  # local import to avoid circular deps

        if not self._path.exists():
            raise TorqStorageError(
                f"Video file not found: '{self._path}'. "
                "Ensure the path is correct and the file has not been moved."
            )

        # imageio.v3.imread with index=None reads all frames in one call and handles
        # resource cleanup internally — no reader object to close manually.
        frames: np.ndarray = iio.imread(str(self._path), index=None)  # [T, H, W, C]

        if frames.shape[0] > _LARGE_FRAME_THRESHOLD:
            logger.warning(
                "ImageSequence loaded %d frames from '%s' — this may use significant memory. "
                "Consider slicing or downsampling for large recordings.",
                frames.shape[0],
                self._path.name,
            )

        return frames

    def __repr__(self) -> str:
        loaded = "loaded" if self._cache is not None else "lazy"
        return f"ImageSequence(path={self._path.name!r}, status={loaded!r})"
