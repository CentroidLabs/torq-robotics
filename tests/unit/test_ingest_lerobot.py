"""Unit tests for torq.ingest.lerobot — LeRobot v3.0 dataset ingestion.

Covers:
    - One Episode returned per episode_index (AC #1)
    - State and action arrays correctly populated (AC #1)
    - Timestamps are int64 nanoseconds (AC #1)
    - Camera observations are lazy ImageSequence (AC #2)
    - Missing info.json raises TorqIngestError (AC #3)

Fixtures: lerobot_dataset, lerobot_no_info are provided
by tests/conftest.py (shared across test modules).
"""

from pathlib import Path

import numpy as np
import pytest


class TestLerobotIngest:
    def test_ingest_lerobot_returns_one_episode_per_episode_index(
        self, lerobot_dataset: Path
    ) -> None:
        """Fixture has 2 episodes → must return 2 Episodes."""
        from torq.episode import Episode
        from torq.ingest.lerobot import ingest

        episodes = ingest(lerobot_dataset)

        assert len(episodes) == 2, f"Expected 2 episodes, got {len(episodes)}"
        for ep in episodes:
            assert isinstance(ep, Episode)
            assert ep.episode_id == ""
            assert ep.source_path == lerobot_dataset
            assert ep.metadata["embodiment"] == "aloha"

    def test_ingest_lerobot_state_and_action_arrays_populated(self, lerobot_dataset: Path) -> None:
        """obs["state"] and actions must be non-empty float32 arrays with correct shape."""
        from torq.ingest.lerobot import ingest

        episodes = ingest(lerobot_dataset)
        ep = episodes[0]

        assert "state" in ep.observations
        state = ep.observations["state"]
        assert isinstance(state, np.ndarray)
        assert state.dtype == np.float32
        assert state.shape == (30, 14), f"Expected (30, 14), got {state.shape}"

        assert ep.actions.dtype == np.float32
        assert ep.actions.shape == (30, 14), f"Expected (30, 14), got {ep.actions.shape}"

    def test_ingest_lerobot_timestamps_are_int64_nanoseconds(self, lerobot_dataset: Path) -> None:
        """Timestamps must be np.int64 nanoseconds, converted from float seconds."""
        from torq.ingest.lerobot import ingest

        episodes = ingest(lerobot_dataset)
        ep = episodes[0]

        assert ep.timestamps.dtype == np.int64
        assert len(ep.timestamps) == 30
        # First timestamp at 0.0s → 0 ns
        assert ep.timestamps[0] == 0
        # Second timestamp at ~0.02s → ~20_000_000 ns (allow float32 precision loss)
        assert abs(ep.timestamps[1] - 20_000_000) < 100, (
            f"Expected ~20_000_000 ns (0.02s at 50Hz), got {ep.timestamps[1]}"
        )

    def test_ingest_lerobot_camera_is_lazy_image_sequence(self, lerobot_dataset: Path) -> None:
        """Camera obs must be ImageSequence with _cache=None (lazy, no decoding at ingest)."""
        from torq.ingest.lerobot import ingest
        from torq.media.image_sequence import ImageSequence

        episodes = ingest(lerobot_dataset)
        ep = episodes[0]

        assert "top" in ep.observations, (
            f"Expected 'top' camera obs, got keys: {list(ep.observations.keys())}"
        )
        img_obs = ep.observations["top"]
        assert isinstance(img_obs, ImageSequence)
        assert img_obs._cache is None, "ImageSequence should not decode frames at ingest time"

    def test_ingest_lerobot_missing_info_json_raises_torq_ingest_error(
        self, lerobot_no_info: Path
    ) -> None:
        """Missing meta/info.json must raise TorqIngestError with 'info.json' in message."""
        from torq.errors import TorqIngestError
        from torq.ingest.lerobot import ingest

        with pytest.raises(TorqIngestError, match="info.json"):
            ingest(lerobot_no_info)

    def test_ingest_lerobot_malformed_info_json_raises_torq_ingest_error(
        self, tmp_path: Path
    ) -> None:
        """Malformed meta/info.json must raise TorqIngestError with 'malformed' in message."""
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text("{bad json")
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)

        from torq.errors import TorqIngestError
        from torq.ingest.lerobot import ingest

        with pytest.raises(TorqIngestError, match="malformed"):
            ingest(tmp_path)

    def test_ingest_lerobot_accepts_str_path(self, lerobot_dataset: Path) -> None:
        """ingest() must accept str path in addition to Path objects."""
        from torq.ingest.lerobot import ingest

        episodes = ingest(str(lerobot_dataset))
        assert len(episodes) == 2

    def test_ingest_lerobot_file_path_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """Passing a file path instead of directory must raise TorqIngestError."""
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("not a directory")

        from torq.errors import TorqIngestError
        from torq.ingest.lerobot import ingest

        with pytest.raises(TorqIngestError, match="not a directory"):
            ingest(file_path)

    def test_ingest_lerobot_null_features_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """info.json with features: null must raise TorqIngestError, not AttributeError."""
        import json

        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 50, "features": None}))
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)

        from torq.errors import TorqIngestError
        from torq.ingest.lerobot import ingest

        with pytest.raises(TorqIngestError, match="features"):
            ingest(tmp_path)

    def test_ingest_lerobot_empty_dataset_returns_empty_list(self, tmp_path: Path) -> None:
        """Dataset with valid structure but no Parquet files returns empty list."""
        import json

        (tmp_path / "meta").mkdir()
        (tmp_path / "meta" / "info.json").write_text(
            json.dumps({"fps": 50, "robot_type": "test", "total_episodes": 0, "features": {}})
        )
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)

        from torq.ingest.lerobot import ingest

        episodes = ingest(tmp_path)
        assert episodes == []

    def test_ingest_lerobot_no_videos_dir_returns_episodes(self, tmp_path: Path) -> None:
        """Dataset with Parquet data but no videos/ returns episodes without camera obs."""
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(99)
        (tmp_path / "meta").mkdir()
        info = {
            "fps": 50,
            "robot_type": "test",
            "total_episodes": 1,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [2]},
                "action": {"dtype": "float32", "shape": [2]},
            },
        }
        (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)

        t = 10
        rows = {
            "episode_index": pa.array([0] * t, type=pa.int64()),
            "frame_index": pa.array(list(range(t)), type=pa.int64()),
            "index": pa.array(list(range(t)), type=pa.int64()),
            "timestamp": pa.array([i / 50.0 for i in range(t)], type=pa.float32()),
        }
        for i in range(2):
            rows[f"observation.state_{i}"] = pa.array(
                rng.standard_normal(t).astype(np.float32).tolist(), type=pa.float32()
            )
            rows[f"action_{i}"] = pa.array(
                rng.standard_normal(t).astype(np.float32).tolist(), type=pa.float32()
            )
        pq.write_table(
            pa.table(rows),
            str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet"),
        )

        from torq.ingest.lerobot import ingest

        episodes = ingest(tmp_path)
        assert len(episodes) == 1
        assert "state" in episodes[0].observations
        assert all(isinstance(v, np.ndarray) for v in episodes[0].observations.values())

    def test_ingest_lerobot_skips_unreadable_parquet_chunk(self, tmp_path: Path) -> None:
        """Unreadable Parquet chunk should be skipped with warning, not crash."""
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.default_rng(99)
        (tmp_path / "meta").mkdir()
        info = {
            "fps": 50,
            "robot_type": "test",
            "total_episodes": 2,
            "features": {"observation.state": {"dtype": "float32", "shape": [2]}},
        }
        (tmp_path / "meta" / "info.json").write_text(json.dumps(info))
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)

        t = 10
        rows = {
            "episode_index": pa.array([0] * t, type=pa.int64()),
            "frame_index": pa.array(list(range(t)), type=pa.int64()),
            "index": pa.array(list(range(t)), type=pa.int64()),
            "timestamp": pa.array([i / 50.0 for i in range(t)], type=pa.float32()),
        }
        for i in range(2):
            rows[f"observation.state_{i}"] = pa.array(
                rng.standard_normal(t).astype(np.float32).tolist(), type=pa.float32()
            )
        pq.write_table(
            pa.table(rows),
            str(tmp_path / "data" / "chunk-000" / "episode_000000.parquet"),
        )
        # Write corrupt Parquet file
        (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").write_bytes(
            b"not a parquet file"
        )

        from torq.ingest.lerobot import ingest

        episodes = ingest(tmp_path)
        assert len(episodes) == 1
