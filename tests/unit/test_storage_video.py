"""Unit tests for torq.storage.video — MP4 read/write via imageio conditional import.

Covers:
    - MP4 file created when saving an ImageSequence observation
    - TorqImportError raised when imageio is not installed
    - Error message contains the install command
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from torq.errors import TorqImportError
from torq.media.image_sequence import ImageSequence


@pytest.fixture
def sample_frames() -> np.ndarray:
    """Minimal [T, H, W, C] uint8 frame array — 4 frames, 8x8 RGB."""
    return np.zeros((4, 8, 8, 3), dtype=np.uint8)


@pytest.fixture
def saved_mp4(tmp_path: Path, sample_frames: np.ndarray) -> Path:
    """Write a minimal MP4 and return the path."""
    from torq.storage.video import save_video

    seq = ImageSequence.__new__(ImageSequence)
    seq._path = tmp_path / "dummy.mp4"
    seq._cache = sample_frames

    mp4_path = tmp_path / "ep_0001_camera.mp4"
    save_video(seq, mp4_path)
    return mp4_path


class TestSaveVideo:
    def test_save_video_creates_mp4(self, tmp_path: Path, sample_frames: np.ndarray) -> None:
        """save_video() must write an .mp4 file at the given path."""
        from torq.storage.video import save_video

        seq = ImageSequence.__new__(ImageSequence)
        seq._path = tmp_path / "dummy.mp4"
        seq._cache = sample_frames

        mp4_path = tmp_path / "ep_0001_camera.mp4"
        save_video(seq, mp4_path)

        assert mp4_path.exists()
        assert mp4_path.stat().st_size > 0

    def test_imageio_not_installed_raises_torq_import_error(
        self, tmp_path: Path, sample_frames: np.ndarray
    ) -> None:
        """save_video() must raise TorqImportError when imageio is not installed."""
        from torq.storage.video import save_video

        seq = ImageSequence.__new__(ImageSequence)
        seq._path = tmp_path / "dummy.mp4"
        seq._cache = sample_frames

        mp4_path = tmp_path / "ep_0001_camera.mp4"

        with patch.dict("sys.modules", {"imageio": None, "imageio.v3": None}):
            with pytest.raises(TorqImportError):
                save_video(seq, mp4_path)

    def test_video_import_error_mentions_install_command(
        self, tmp_path: Path, sample_frames: np.ndarray
    ) -> None:
        """TorqImportError message must contain 'pip install torq-robotics[vision]'."""
        from torq.storage.video import save_video

        seq = ImageSequence.__new__(ImageSequence)
        seq._path = tmp_path / "dummy.mp4"
        seq._cache = sample_frames

        mp4_path = tmp_path / "ep_0001_camera.mp4"

        with patch.dict("sys.modules", {"imageio": None, "imageio.v3": None}):
            with pytest.raises(TorqImportError, match=r"pip install torq-robotics\[vision\]"):
                save_video(seq, mp4_path)
