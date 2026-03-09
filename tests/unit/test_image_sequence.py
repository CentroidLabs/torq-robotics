"""Unit tests for torq.media.image_sequence — ImageSequence lazy loader.

12 tests covering construction, lazy loading, caching, error handling, and import compliance.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from torq.errors import TorqImportError, TorqStorageError
from torq.media.image_sequence import ImageSequence


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_array(t: int = 4, h: int = 8, w: int = 8) -> np.ndarray:
    """Return a uint8 ndarray of shape (T, H, W, 3) representing decoded frames."""
    return np.zeros((t, h, w, 3), dtype=np.uint8)


# ── AC #4: Construction ────────────────────────────────────────────────────────


def test_construction_from_path(tmp_path: Path) -> None:
    """ImageSequence constructs without error (file need not exist at init)."""
    seq = ImageSequence(tmp_path / "clip.mp4")
    assert seq is not None


def test_lazy_no_frames_at_construction(tmp_path: Path) -> None:
    """_cache is None immediately after construction — no I/O happens."""
    seq = ImageSequence(tmp_path / "clip.mp4")
    assert seq._cache is None


def test_accepts_pathlib_path(tmp_path: Path) -> None:
    """Constructor works with pathlib.Path."""
    seq = ImageSequence(tmp_path / "clip.mp4")
    assert isinstance(seq._path, Path)


def test_accepts_str_path(tmp_path: Path) -> None:
    """Constructor works with str path."""
    seq = ImageSequence(str(tmp_path / "clip.mp4"))
    assert isinstance(seq._path, Path)


# ── AC #4: Lazy loading and shape ──────────────────────────────────────────────


def test_frames_access_triggers_load(tmp_path: Path) -> None:
    """Accessing .frames returns an ndarray (triggers decode)."""
    fake_path = tmp_path / "clip.mp4"
    fake_path.touch()

    seq = ImageSequence(fake_path)
    with patch("imageio.v3.imread", return_value=_fake_array(t=4)):
        result = seq.frames
    assert isinstance(result, np.ndarray)


def test_frames_shape_is_t_h_w_c(tmp_path: Path) -> None:
    """Frames array has 4 dimensions: (T, H, W, C)."""
    fake_path = tmp_path / "clip.mp4"
    fake_path.touch()

    t, h, w = 4, 8, 10
    seq = ImageSequence(fake_path)
    with patch("imageio.v3.imread", return_value=_fake_array(t=t, h=h, w=w)):
        result = seq.frames
    assert result.shape == (t, h, w, 3)


def test_frames_dtype_uint8(tmp_path: Path) -> None:
    """Frames array dtype is uint8."""
    fake_path = tmp_path / "clip.mp4"
    fake_path.touch()

    seq = ImageSequence(fake_path)
    with patch("imageio.v3.imread", return_value=_fake_array()):
        result = seq.frames
    assert result.dtype == np.uint8


def test_frames_cached_on_second_access(tmp_path: Path) -> None:
    """Second .frames access returns the same array object (identity check)."""
    fake_path = tmp_path / "clip.mp4"
    fake_path.touch()

    seq = ImageSequence(fake_path)
    with patch("imageio.v3.imread", return_value=_fake_array()) as mock_read:
        first = seq.frames
        second = seq.frames
    # imageio.v3.imread must only be called once — second access uses cache
    mock_read.assert_called_once()
    assert first is second


# ── Error handling ─────────────────────────────────────────────────────────────


def test_missing_file_raises_torq_storage_error(tmp_path: Path) -> None:
    """Accessing .frames for a missing file raises TorqStorageError specifically."""
    pytest.importorskip("imageio", reason="imageio required for this test path")
    missing = tmp_path / "does_not_exist.mp4"
    seq = ImageSequence(missing)
    # imageio import succeeds; file-existence check fires before any decode attempt
    with pytest.raises(TorqStorageError):
        seq.frames


def test_imageio_not_installed_raises_torq_import_error(tmp_path: Path) -> None:
    """When imageio is not installed, accessing .frames raises TorqImportError."""
    seq = ImageSequence(tmp_path / "clip.mp4")
    with patch.dict("sys.modules", {"imageio": None, "imageio.v3": None}):
        with pytest.raises(TorqImportError):
            seq.frames


def test_torq_import_error_mentions_install_command(tmp_path: Path) -> None:
    """TorqImportError message contains pip install command."""
    seq = ImageSequence(tmp_path / "clip.mp4")
    with patch.dict("sys.modules", {"imageio": None, "imageio.v3": None}):
        with pytest.raises(TorqImportError) as exc_info:
            seq.frames
    assert "pip install" in str(exc_info.value)


# ── Import compliance ──────────────────────────────────────────────────────────


def test_image_sequence_imports_nothing_from_torq_except_errors() -> None:
    """image_sequence.py must not import from torq at module level except torq.errors."""
    src_path = Path(__file__).parent.parent.parent / "src" / "torq" / "media" / "image_sequence.py"
    src = src_path.read_text()

    # Only check top-level (non-indented) import lines
    bad_imports = []
    for line in src.splitlines():
        if not line.startswith(" ") and not line.startswith("\t"):
            if ("from torq." in line or "import torq." in line) and "torq.errors" not in line:
                bad_imports.append(line)
    assert bad_imports == [], f"Forbidden top-level imports in image_sequence.py: {bad_imports}"
