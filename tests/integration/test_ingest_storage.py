"""Integration tests for the storage save/load round-trip.

Tests follow ARCHITECTURE.md storage conventions:
    - save(episode, path) → writes parquet + optional video + index shards
    - load(episode_id, path) → reconstructs Episode with identical arrays

These tests perform real disk I/O, so they are marked @pytest.mark.slow.
"""

from pathlib import Path

import numpy as np
import pytest

from torq.episode import Episode
from torq.media.image_sequence import ImageSequence


@pytest.fixture
def numeric_episode(tmp_path: Path) -> Episode:
    """Episode with only numeric observations (no images)."""
    return Episode(
        episode_id="ep_TEMP",
        observations={
            "joint_pos": np.array(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
            ),
            "joint_vel": np.array([[1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float32),
        },
        actions=np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
        timestamps=np.array([0, 1_000_000], dtype=np.int64),
        source_path=tmp_path / "demo.mcap",
        metadata={"task": "pick_and_place", "embodiment": "aloha", "success": True},
    )


@pytest.fixture
def image_episode(tmp_path: Path) -> Episode:
    """Episode with an ImageSequence observation."""
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    seq = ImageSequence.__new__(ImageSequence)
    seq._path = tmp_path / "camera.mp4"
    seq._cache = frames
    return Episode(
        episode_id="ep_TEMP",
        observations={
            "joint_pos": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "camera": seq,
        },
        actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        timestamps=np.array([0, 1_000_000], dtype=np.int64),
        source_path=tmp_path / "demo.mcap",
        metadata={"task": "grasp"},
    )


@pytest.mark.slow
class TestSaveLoadRoundTrip:
    def test_save_and_load_full_round_trip(
        self, tmp_path: Path, numeric_episode: Episode
    ) -> None:
        """save() then load() must return an Episode with identical fields."""
        import torq as tq

        dataset_path = tmp_path / "dataset"
        saved = tq.save(numeric_episode, dataset_path)

        assert saved.episode_id == "ep_0001"
        assert (dataset_path / "episodes" / "ep_0001.parquet").exists()
        assert (dataset_path / "index" / "manifest.json").exists()

        loaded = tq.load("ep_0001", dataset_path)

        assert loaded.episode_id == "ep_0001"
        np.testing.assert_array_equal(loaded.timestamps, numeric_episode.timestamps)
        np.testing.assert_array_almost_equal(
            loaded.observations["joint_pos"], numeric_episode.observations["joint_pos"]
        )
        np.testing.assert_array_almost_equal(
            loaded.observations["joint_vel"], numeric_episode.observations["joint_vel"]
        )
        np.testing.assert_array_almost_equal(loaded.actions, numeric_episode.actions)

    @pytest.mark.slow
    def test_save_and_load_with_image_sequence(
        self, tmp_path: Path, image_episode: Episode
    ) -> None:
        """save() with ImageSequence observation must write an MP4 file.

        The loaded episode must have numeric observations restored correctly.
        """
        import torq as tq

        dataset_path = tmp_path / "dataset"
        saved = tq.save(image_episode, dataset_path)

        assert saved.episode_id == "ep_0001"
        # MP4 file written for the camera observation
        mp4_files = list((dataset_path / "videos").glob("*.mp4"))
        assert len(mp4_files) == 1, f"Expected 1 mp4, found: {mp4_files}"

        loaded = tq.load("ep_0001", dataset_path)
        assert loaded.episode_id == "ep_0001"
        # numeric observation round-trips
        np.testing.assert_array_almost_equal(
            loaded.observations["joint_pos"], image_episode.observations["joint_pos"]
        )
