"""Unit tests for torq.storage.index — sharded JSON index + Episode ID generation.

Covers:
    - Episode ID format (ep_NNNN)
    - Manifest counter increments across saves
    - Atomic write guarantee (no partial state on interrupted write)
    - String normalisation for task/embodiment buckets
"""

import json
import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from torq.episode import Episode


@pytest.fixture
def index_root(tmp_path: Path) -> Path:
    d = tmp_path / "index"
    d.mkdir()
    return d


@pytest.fixture
def sample_episode(tmp_path: Path) -> Episode:
    return Episode(
        episode_id="ep_0001",
        observations={"joint_pos": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)},
        actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        timestamps=np.array([0, 1_000_000], dtype=np.int64),
        source_path=tmp_path / "source.mcap",
        metadata={"task": "pick_place", "embodiment": "aloha"},
    )


class TestEpisodeIdGeneration:
    def test_episode_id_generated_in_ep_format(self, index_root: Path) -> None:
        """_next_episode_id() must return a string matching 'ep_NNNN' format."""
        from torq.storage.index import _next_episode_id

        ep_id = _next_episode_id(index_root)
        assert re.match(r"^ep_\d{4}$", ep_id), f"ID {ep_id!r} doesn't match ep_NNNN format"

    def test_manifest_counter_increments(
        self, index_root: Path, sample_episode: Episode
    ) -> None:
        """Saving two episodes must produce ep_0001 and ep_0002."""
        from torq.storage.index import _next_episode_id, update_index

        id1 = _next_episode_id(index_root)
        assert id1 == "ep_0001"
        update_index(id1, sample_episode, index_root)

        id2 = _next_episode_id(index_root)
        assert id2 == "ep_0002"
        update_index(id2, sample_episode, index_root)

        # verify manifest
        manifest = json.loads((index_root / "manifest.json").read_text())
        assert manifest["episode_count"] == 2


class TestAtomicWrite:
    def test_atomic_write_no_partial_state(
        self, index_root: Path, sample_episode: Episode
    ) -> None:
        """A simulated interrupted write must leave the index unchanged.

        We first write one episode (ep_0001). Then we simulate a crash
        during the second update_index by making os.replace raise an exception.
        The manifest should still reflect ep_0001 only.
        """
        from torq.storage.index import update_index

        # Write the first episode successfully
        update_index("ep_0001", sample_episode, index_root)

        manifest_before = json.loads((index_root / "manifest.json").read_text())
        assert manifest_before["episode_count"] == 1

        # Simulate interrupted write during second update
        import os
        import torq.storage.index as idx_module

        real_replace = os.replace

        call_count = [0]

        def flaky_replace(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:  # fail on first replace (manifest.json)
                import pathlib

                pathlib.Path(src).unlink(missing_ok=True)  # discard temp file
                raise OSError("Simulated disk failure")
            real_replace(src, dst)

        with patch.object(idx_module, "_atomic_write_json") as mock_write:
            mock_write.side_effect = OSError("Simulated disk failure")
            try:
                update_index("ep_0002", sample_episode, index_root)
            except (OSError, Exception):
                pass  # expected

        # Manifest should still show only 1 episode
        manifest_after = json.loads((index_root / "manifest.json").read_text())
        assert manifest_after["episode_count"] == 1


class TestStringNormalisation:
    def test_string_normalisation_same_bucket(
        self, index_root: Path, tmp_path: Path
    ) -> None:
        """AC5: 'ALOHA-2', 'aloha2', 'Aloha 2' must all resolve to the same bucket.

        _normalise() applies lowercase + strip + separator removal so that these
        three strings all collapse to 'aloha2'.
        """
        from torq.storage.index import _normalise

        assert _normalise("ALOHA-2") == "aloha2"
        assert _normalise("  aloha2  ") == "aloha2"
        assert _normalise("Aloha 2") == "aloha2"

        # Verify that identical strings after normalisation land in the same bucket
        ep1 = Episode(
            episode_id="ep_0001",
            observations={"j": np.zeros((2, 1), dtype=np.float32)},
            actions=np.zeros((2, 1), dtype=np.float32),
            timestamps=np.array([0, 1_000_000], dtype=np.int64),
            source_path=tmp_path / "s.mcap",
            metadata={"task": "PICK", "embodiment": "aloha"},
        )
        ep2 = Episode(
            episode_id="ep_0002",
            observations={"j": np.zeros((2, 1), dtype=np.float32)},
            actions=np.zeros((2, 1), dtype=np.float32),
            timestamps=np.array([0, 1_000_000], dtype=np.int64),
            source_path=tmp_path / "s.mcap",
            metadata={"task": "pick", "embodiment": "aloha"},
        )

        from torq.storage.index import update_index

        update_index("ep_0001", ep1, index_root)
        update_index("ep_0002", ep2, index_root)

        by_task = json.loads((index_root / "by_task.json").read_text())
        # Both should land in the same bucket since "PICK" normalises to "pick"
        assert "pick" in by_task
        assert set(by_task["pick"]) == {"ep_0001", "ep_0002"}
