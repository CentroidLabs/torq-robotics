"""Unit tests for torq.ingest.hdf5 — HDF5 (robomimic) ingestion.

Covers:
    - One Episode returned per demo group (AC #1)
    - Image observations wrapped as _InMemoryFrames with .frames property (AC #2)
    - Corrupt/truncated HDF5 raises TorqIngestError (AC #3)
    - Edge cases: missing /data, no demo_* groups, missing actions/obs keys
    - Storage layer compatibility: _InMemoryFrames works with save/load roundtrip

Fixtures: robomimic_hdf5, robomimic_images_hdf5, corrupt_hdf5 are provided
by tests/fixtures/conftest.py (shared across test modules).
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


class TestHdf5Ingest:
    def test_ingest_hdf5_returns_one_episode_per_demo(self, robomimic_hdf5: Path) -> None:
        """robomimic_simple.hdf5 has 2 demos → must return 2 Episodes."""
        from torq.episode import Episode
        from torq.ingest.hdf5 import ingest

        episodes = ingest(robomimic_hdf5)

        assert len(episodes) == 2, f"Expected 2 episodes (2 demos), got {len(episodes)}"
        for ep in episodes:
            assert isinstance(ep, Episode)
            assert "joint_pos" in ep.observations
            assert ep.actions is not None and ep.actions.size > 0
            assert ep.timestamps.dtype == np.int64
            assert ep.source_path == robomimic_hdf5
            assert ep.episode_id == ""

        # Demo 0 has 30 timesteps, Demo 1 has 20
        assert len(episodes[0].timestamps) == 30
        assert len(episodes[1].timestamps) == 20

    def test_ingest_hdf5_image_obs_is_image_sequence(self, robomimic_images_hdf5: Path) -> None:
        """robomimic_images.hdf5 → observations["agentview_image"] has .frames [T,H,W,C]."""
        from torq.ingest.hdf5 import ingest

        episodes = ingest(robomimic_images_hdf5)

        assert len(episodes) == 1
        ep = episodes[0]
        assert "agentview_image" in ep.observations

        img_obs = ep.observations["agentview_image"]
        assert hasattr(img_obs, "frames"), "Image observation must have .frames property"

        frames = img_obs.frames
        assert frames.ndim == 4, f"Expected 4D [T,H,W,C], got {frames.ndim}D"
        assert frames.shape == (10, 48, 64, 3), f"Unexpected shape: {frames.shape}"
        assert frames.dtype == np.uint8

    def test_ingest_hdf5_corrupt_file_raises_torq_ingest_error(self, corrupt_hdf5: Path) -> None:
        """Truncated HDF5 must raise TorqIngestError with file path in message."""
        from torq.errors import TorqIngestError
        from torq.ingest.hdf5 import ingest

        with pytest.raises(TorqIngestError, match=str(corrupt_hdf5)):
            ingest(corrupt_hdf5)

    def test_ingest_accepts_str_path(self, robomimic_hdf5: Path) -> None:
        """ingest() must accept str path, not just Path objects."""
        from torq.ingest.hdf5 import ingest

        episodes = ingest(str(robomimic_hdf5))
        assert len(episodes) == 2


class TestHdf5IngestEdgeCases:
    """Edge case tests for HDF5 ingestion (review follow-ups)."""

    def test_missing_data_group_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """HDF5 file without /data group → TorqIngestError."""
        from torq.errors import TorqIngestError
        from torq.ingest.hdf5 import ingest

        hdf5_path = tmp_path / "no_data.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_group("other_stuff")

        with pytest.raises(TorqIngestError, match="missing the '/data' group"):
            ingest(hdf5_path)

    def test_no_demo_groups_returns_empty_list(self, tmp_path: Path) -> None:
        """HDF5 with /data but no demo_* groups → empty list + warning."""
        from torq.ingest.hdf5 import ingest

        hdf5_path = tmp_path / "empty_data.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            f.create_group("data")

        episodes = ingest(hdf5_path)
        assert episodes == []

    def test_demo_missing_actions_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """Demo group without 'actions' key → TorqIngestError."""
        from torq.errors import TorqIngestError
        from torq.ingest.hdf5 import ingest

        hdf5_path = tmp_path / "no_actions.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            data = f.create_group("data")
            demo = data.create_group("demo_0")
            obs = demo.create_group("obs")
            obs.create_dataset("joint_pos", data=np.zeros((10, 6), dtype=np.float32))

        with pytest.raises(TorqIngestError, match="missing required key 'actions'"):
            ingest(hdf5_path)

    def test_demo_missing_obs_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """Demo group without 'obs' key → TorqIngestError."""
        from torq.errors import TorqIngestError
        from torq.ingest.hdf5 import ingest

        hdf5_path = tmp_path / "no_obs.hdf5"
        with h5py.File(hdf5_path, "w") as f:
            data = f.create_group("data")
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=np.zeros((10, 6), dtype=np.float32))

        with pytest.raises(TorqIngestError, match="missing required key 'obs'"):
            ingest(hdf5_path)


class TestInMemoryFrames:
    """Tests for _InMemoryFrames validation and immutability (review R2 follow-ups)."""

    def test_non_4d_array_raises_torq_ingest_error(self) -> None:
        """_InMemoryFrames rejects arrays that aren't 4D [T, H, W, C]."""
        from torq.errors import TorqIngestError
        from torq.ingest.hdf5 import _InMemoryFrames

        with pytest.raises(TorqIngestError, match="requires 4D array"):
            _InMemoryFrames(np.zeros((10, 6), dtype=np.uint8))

    def test_frames_are_read_only(self) -> None:
        """_InMemoryFrames.frames must not be writable to prevent silent corruption."""
        from torq.ingest.hdf5 import _InMemoryFrames

        frames_arr = np.zeros((5, 48, 64, 3), dtype=np.uint8)
        imf = _InMemoryFrames(frames_arr)
        with pytest.raises(ValueError, match="read-only"):
            imf.frames[0, 0, 0, 0] = 255


class TestHdf5StorageCompatibility:
    """Verify _InMemoryFrames works with storage layer save/load."""

    av = pytest.importorskip("av", reason="vision extras required for storage compat tests")

    def test_save_parquet_skips_inmemoryframes(self, robomimic_images_hdf5: Path) -> None:
        """save_parquet must skip _InMemoryFrames obs (not crash or corrupt)."""
        from torq.ingest.hdf5 import ingest
        from torq.storage.parquet import save_parquet

        episodes = ingest(robomimic_images_hdf5)
        ep = episodes[0]
        assert "agentview_image" in ep.observations

        # Assign an episode ID (storage requires it)
        object.__setattr__(ep, "episode_id", "ep_test_0001")

        # save_parquet should skip image obs without crashing
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = save_parquet(ep, Path(tmpdir))
            assert parquet_path.exists()

            # Verify the parquet file doesn't contain image columns
            import pyarrow.parquet as pq

            table = pq.read_table(str(parquet_path))
            col_names = table.column_names
            assert not any("agentview_image" in c for c in col_names)
            # But joint_pos columns should be there
            assert any("joint_pos" in c for c in col_names)

    def test_save_episode_handles_inmemoryframes_as_video(
        self, robomimic_images_hdf5: Path
    ) -> None:
        """_impl.save() must detect _InMemoryFrames and route to save_video."""
        from torq.ingest.hdf5 import ingest

        episodes = ingest(robomimic_images_hdf5)
        ep = episodes[0]

        # Verify the image obs has .frames (duck-typing contract)
        img_obs = ep.observations["agentview_image"]
        assert hasattr(img_obs, "frames")

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from torq.storage._impl import save

            save(ep, tmpdir)
            # Verify video file was created
            videos_dir = Path(tmpdir) / "videos"
            assert videos_dir.exists(), "videos/ dir should be created for image obs"
            mp4_files = list(videos_dir.glob("*.mp4"))
            assert len(mp4_files) == 1, f"Expected 1 MP4, got {len(mp4_files)}"
            assert "agentview_image" in mp4_files[0].name
