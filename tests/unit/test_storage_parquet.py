"""Unit tests for torq.storage.parquet — Parquet read/write with column name templates.

Tests follow ARCHITECTURE.md Parquet Schema Conventions:
    - timestamp_ns (int64 nanoseconds)
    - obs_{key}_{i} for observations
    - action_{i} for actions
    - metadata_* columns
    - atomic write pattern (write to .tmp then os.replace)
"""

from pathlib import Path

import numpy as np
import pytest

from torq.episode import Episode


@pytest.fixture
def sample_episode(tmp_path: Path) -> Episode:
    """Minimal Episode with numeric observations and actions."""
    return Episode(
        episode_id="ep_0001",
        observations={"joint_pos": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        timestamps=np.array([0, 1_000_000], dtype=np.int64),
        source_path=tmp_path / "source.mcap",
        metadata={"task": "pick_place", "embodiment": "aloha", "success": True},
    )


class TestSaveParquet:
    def test_save_parquet_creates_file(self, tmp_path: Path, sample_episode: Episode) -> None:
        """save_parquet() must write a .parquet file at episodes/{episode_id}.parquet."""
        from torq.storage.parquet import save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        result_path = save_parquet(sample_episode, episodes_dir)

        assert result_path.exists()
        assert result_path.suffix == ".parquet"
        assert result_path.name == "ep_0001.parquet"

    def test_load_parquet_round_trip(self, tmp_path: Path, sample_episode: Episode) -> None:
        """load_parquet() must return an Episode with identical arrays to the original."""
        from torq.storage.parquet import load_parquet, save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        save_parquet(sample_episode, episodes_dir)
        loaded = load_parquet("ep_0001", tmp_path)

        assert loaded.episode_id == sample_episode.episode_id
        np.testing.assert_array_equal(loaded.timestamps, sample_episode.timestamps)
        np.testing.assert_array_almost_equal(
            loaded.observations["joint_pos"], sample_episode.observations["joint_pos"]
        )

    def test_timestamps_stored_as_int64(self, tmp_path: Path, sample_episode: Episode) -> None:
        """The parquet file must have a timestamp_ns column with dtype int64."""
        import pyarrow.parquet as pq

        from torq.storage.parquet import save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        save_parquet(sample_episode, episodes_dir)
        parquet_path = episodes_dir / "ep_0001.parquet"
        table = pq.read_table(str(parquet_path))

        assert "timestamp_ns" in table.schema.names
        import pyarrow as pa

        assert table.schema.field("timestamp_ns").type == pa.int64()

    def test_actions_round_trip(self, tmp_path: Path, sample_episode: Episode) -> None:
        """Actions arrays must be numerically identical after save/load."""
        from torq.storage.parquet import load_parquet, save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        save_parquet(sample_episode, episodes_dir)
        loaded = load_parquet("ep_0001", tmp_path)

        np.testing.assert_array_almost_equal(loaded.actions, sample_episode.actions)

    def test_observations_round_trip(self, tmp_path: Path, sample_episode: Episode) -> None:
        """Observation dict keys and array values must match after save/load."""
        from torq.storage.parquet import load_parquet, save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        save_parquet(sample_episode, episodes_dir)
        loaded = load_parquet("ep_0001", tmp_path)

        assert set(loaded.observations.keys()) == set(sample_episode.observations.keys())
        for key in sample_episode.observations:
            np.testing.assert_array_almost_equal(
                loaded.observations[key], sample_episode.observations[key]
            )

    def test_parquet_uses_atomic_write(self, tmp_path: Path, sample_episode: Episode) -> None:
        """save_parquet() must call os.replace() for the atomic rename.

        We mock os.replace to verify it is called with the expected .tmp and .parquet
        paths, confirming the write-then-rename pattern is used.
        """
        import os
        from unittest.mock import call, patch

        import torq.storage.parquet as parquet_module
        from torq.storage.parquet import save_parquet

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()

        real_replace = os.replace
        replace_calls: list = []

        def recording_replace(src, dst):
            replace_calls.append((str(src), str(dst)))
            real_replace(src, dst)

        with patch.object(parquet_module.os, "replace", side_effect=recording_replace):
            save_parquet(sample_episode, episodes_dir)

        assert len(replace_calls) == 1, f"Expected 1 os.replace call, got: {replace_calls}"
        src, dst = replace_calls[0]
        assert src.endswith(".parquet.tmp"), f"Source was not a .tmp file: {src}"
        assert dst.endswith("ep_0001.parquet"), f"Destination was not the parquet file: {dst}"
        assert (episodes_dir / "ep_0001.parquet").exists()
        assert not list(episodes_dir.glob("*.tmp")), "Temp files left behind"
