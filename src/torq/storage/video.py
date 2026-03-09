"""MP4 read/write for Episode image data.

imageio is an optional dependency (``pip install torq-robotics[vision]``).
All imports of imageio and av are deferred to inside function bodies — never at module level.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from torq.errors import TorqImportError, TorqStorageError
from torq.types import FrameProvider

if TYPE_CHECKING:
    from torq.media.image_sequence import ImageSequence

__all__ = ["save_video", "load_video"]

logger = logging.getLogger(__name__)


def save_video(image_seq: FrameProvider, path: Path) -> Path:
    """Write image frame data to an MP4 file.

    Accepts any object satisfying the ``FrameProvider`` protocol (e.g.
    ``ImageSequence`` or ``_InMemoryFrames``). Requires the ``[vision]``
    extra: ``pip install torq-robotics[vision]``.

    Args:
        image_seq: Frame provider whose ``.frames`` property returns an
            ``np.ndarray`` of shape ``[T, H, W, C]`` uint8.
        path: Destination MP4 file path (e.g. ``videos/ep_0001_camera.mp4``).

    Returns:
        Path to the written MP4 file.

    Raises:
        TorqImportError: When imageio (vision extras) is not installed.
        TorqStorageError: On any file I/O failure.
    """
    try:
        import imageio  # noqa: PLC0415 — intentional lazy import; validates vision extra

        _imageio_version = imageio.__version__  # referenced to confirm module is live
    except (ImportError, TypeError):
        raise TorqImportError(
            "imageio is required to save video frames. "
            "Install it with: pip install torq-robotics[vision]"
        ) from None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames = image_seq.frames  # [T, H, W, C] uint8
    n_frames, height, width, _ = frames.shape

    # Use PyAV (av) for the actual write — imageio v3 wraps av internally.
    # We call av directly because imageio 2.33 has a known incompatibility with
    # PyAV 16.x in the time_base initialization inside write_frame().
    try:
        import av  # noqa: PLC0415 — intentional lazy import; av is a peer dep of imageio

        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for frame_np in frames:
                av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except ImportError:
        raise TorqImportError(
            "av (PyAV) is required to save video frames. "
            "Install it with: pip install torq-robotics[vision]"
        ) from None
    except Exception as exc:
        raise TorqStorageError(
            f"Failed to write MP4 file '{path}': {exc}. "
            "Check that the directory exists and imageio-ffmpeg/av is installed."
        ) from exc

    logger.debug("Saved video → %s (%d frames, %dx%d)", path, n_frames, width, height)
    return path


def load_video(path: Path) -> ImageSequence:
    """Load an MP4 file as a lazy ImageSequence.

    The returned ImageSequence will decode frames from disk on first access
    to ``.frames``.

    Args:
        path: Path to the MP4 file.

    Returns:
        ImageSequence backed by the given path.

    Raises:
        TorqStorageError: If the file does not exist.
    """
    from torq.media.image_sequence import ImageSequence  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise TorqStorageError(
            f"Video file not found: '{path}'. Ensure the episode was saved with image observations."
        )
    return ImageSequence(path)
