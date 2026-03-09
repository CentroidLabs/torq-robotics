"""Unit tests for query_index() (storage.index) and tq.query() (compose).

Covers:
    - query_index() with populated index: task + quality_min → correct IDs
    - query_index() with no filters: returns all IDs in sorted order
    - query_index() with compound task list (OR logic) intersected with other filters
    - query_index() with zero results: empty list, no exception
    - query_index() on empty index: empty list
    - tq.query() returns lazy iterator that yields Episode objects
    - tq.query() with no matches: empty iterator + warning logged
    - tq.query() episode ID order is deterministic (sorted)
    - tq.query() raises TorqComposeError if store_path is not provided
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torq.errors import TorqComposeError
from torq.storage.index import query_index


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_index(index_root: Path, *, by_task: dict, by_embodiment: dict, quality_list: list) -> None:
    """Write the four index shards to disk."""
    index_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "episode_count": len(quality_list),
        "last_updated": "2026-03-09T00:00:00Z",
    }
    (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (index_root / "by_task.json").write_text(json.dumps(by_task), encoding="utf-8")
    (index_root / "by_embodiment.json").write_text(json.dumps(by_embodiment), encoding="utf-8")
    (index_root / "quality.json").write_text(json.dumps(quality_list), encoding="utf-8")


@pytest.fixture
def index_root(tmp_path: Path) -> Path:
    root = tmp_path / "index"
    _write_index(
        root,
        by_task={
            "pick": ["ep_0001", "ep_0002"],
            "place": ["ep_0003", "ep_0004"],
            "pickandplace": ["ep_0005"],
        },
        by_embodiment={
            "aloha2": ["ep_0001", "ep_0003"],
            "franka": ["ep_0002", "ep_0004"],
            "ur5": ["ep_0005"],
        },
        quality_list=[
            [0.60, "ep_0001"],
            [0.72, "ep_0002"],
            [0.80, "ep_0003"],
            [0.91, "ep_0004"],
            [None, "ep_0005"],
        ],
    )
    return root


# ── query_index() ─────────────────────────────────────────────────────────────

class TestQueryIndex:
    def test_task_and_quality_min_filter(self, index_root: Path) -> None:
        """task='pick' ∩ quality_min=0.65 → ep_0002 (ep_0001=0.60 excluded)."""
        result = query_index(index_root, task="pick", quality_min=0.65)
        assert result == ["ep_0002"]

    def test_no_filters_returns_all_ids_sorted(self, index_root: Path) -> None:
        """No filters → all episode IDs in sorted order."""
        result = query_index(index_root)
        assert result == ["ep_0001", "ep_0002", "ep_0003", "ep_0004", "ep_0005"]

    def test_compound_task_list_union_intersected_with_quality(self, index_root: Path) -> None:
        """task=['pick','place'] union = {ep_0001..ep_0004}; quality_min=0.75 → {ep_0003, ep_0004}."""
        result = query_index(index_root, task=["pick", "place"], quality_min=0.75)
        assert result == ["ep_0003", "ep_0004"]

    def test_zero_results_returns_empty_list(self, index_root: Path) -> None:
        """No matching episodes → empty list, no exception."""
        result = query_index(index_root, task="grasp")
        assert result == []

    def test_empty_index_returns_empty_list(self, tmp_path: Path) -> None:
        """If no quality.json exists, return empty list."""
        empty_root = tmp_path / "empty_index"
        empty_root.mkdir()
        result = query_index(empty_root)
        assert result == []

    def test_embodiment_filter(self, index_root: Path) -> None:
        result = query_index(index_root, embodiment="aloha2")
        assert result == ["ep_0001", "ep_0003"]

    def test_task_embodiment_intersection(self, index_root: Path) -> None:
        """task='pick' {ep_0001, ep_0002} ∩ embodiment='franka' {ep_0002, ep_0004} = {ep_0002}."""
        result = query_index(index_root, task="pick", embodiment="franka")
        assert result == ["ep_0002"]

    def test_quality_max_excludes_high_scores(self, index_root: Path) -> None:
        result = query_index(index_root, quality_max=0.70)
        assert result == ["ep_0001"]

    def test_quality_range_filter(self, index_root: Path) -> None:
        result = query_index(index_root, quality_min=0.70, quality_max=0.85)
        assert result == ["ep_0002", "ep_0003"]

    def test_episodes_with_none_quality_excluded_by_quality_filter(self, index_root: Path) -> None:
        """ep_0005 has None quality score — excluded when any quality filter active."""
        result = query_index(index_root, quality_min=0.0)
        assert "ep_0005" not in result

    def test_normalisation_applied_to_task_query(self, index_root: Path) -> None:
        """'Pick' (uppercase) should match normalised key 'pick'."""
        result = query_index(index_root, task="Pick")
        assert result == ["ep_0001", "ep_0002"]

    def test_result_is_sorted(self, index_root: Path) -> None:
        """Result list must be deterministically sorted."""
        result = query_index(index_root, task=["pick", "place"])
        assert result == sorted(result)


# ── tq.query() ────────────────────────────────────────────────────────────────

class TestTqQuery:
    def test_raises_type_error_without_store_path(self) -> None:
        import torq as tq

        # store_path is now a truly required keyword-only argument; Python raises
        # TypeError at the call site so type checkers can also catch it statically.
        with pytest.raises(TypeError, match="store_path"):
            next(tq.query(task="pick"))  # type: ignore[call-arg]

    def test_returns_lazy_iterator_that_yields_episodes(self, tmp_path: Path) -> None:
        """tq.query() must return an iterator; consuming it calls storage.load() per episode."""
        import torq as tq

        # Write minimal index
        index_root = tmp_path / "index"
        _write_index(
            index_root,
            by_task={"pick": ["ep_0001"]},
            by_embodiment={},
            quality_list=[[0.80, "ep_0001"]],
        )

        mock_episode = MagicMock()
        with patch("torq.compose._query.load", return_value=mock_episode) as mock_load:
            result = tq.query(task="pick", store_path=tmp_path)
            # Must be a generator/iterator, not a list
            import types
            assert isinstance(result, types.GeneratorType)
            episodes = list(result)

        assert len(episodes) == 1
        assert episodes[0] is mock_episode
        mock_load.assert_called_once_with("ep_0001", tmp_path)

    def test_empty_iterator_and_warning_when_no_matches(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Zero matching episodes → empty iterator + logger.warning emitted."""
        import torq as tq

        index_root = tmp_path / "index"
        _write_index(
            index_root,
            by_task={"pick": ["ep_0001"]},
            by_embodiment={},
            quality_list=[[0.80, "ep_0001"]],
        )

        with caplog.at_level(logging.WARNING, logger="torq.compose._query"):
            result = list(tq.query(task="grasp", store_path=tmp_path))

        assert result == []
        assert any("0 episodes" in record.message for record in caplog.records)

    def test_episode_id_order_is_deterministic(self, tmp_path: Path) -> None:
        """Episodes must be yielded in sorted episode ID order."""
        index_root = tmp_path / "index"
        # Write quality_list in reverse order to test sorting
        _write_index(
            index_root,
            by_task={},
            by_embodiment={},
            quality_list=[
                [0.90, "ep_0003"],
                [0.70, "ep_0001"],
                [0.80, "ep_0002"],
            ],
        )

        yielded_ids = []

        def fake_load(ep_id: str, path: object) -> MagicMock:
            yielded_ids.append(ep_id)
            return MagicMock()

        from torq.compose._query import query as compose_query

        with patch("torq.compose._query.load", side_effect=fake_load):
            list(compose_query(store_path=tmp_path))

        assert yielded_ids == sorted(yielded_ids)

    def test_query_with_empty_index(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """If store_path has no index, returns empty iterator without raising."""
        import torq as tq

        (tmp_path / "index").mkdir()

        with caplog.at_level(logging.WARNING, logger="torq.compose._query"):
            result = list(tq.query(store_path=tmp_path))

        assert result == []
        # Warning should be emitted
        assert any("0 episodes" in record.message for record in caplog.records)


# ── Performance test (AC #1: under 1 second at 100K+ episodes) ───────────────

class TestQueryPerformance:
    def test_query_index_under_one_second_at_100k_episodes(self, tmp_path: Path) -> None:
        """query_index() must complete in < 1 second on a 100K-episode index (AC #1)."""
        import time

        n = 100_000
        index_root = tmp_path / "index"
        index_root.mkdir()

        # Generate synthetic index shards for 100K episodes
        tasks = ["pick", "place", "push", "grasp", "insert"]
        embodiments = ["aloha2", "franka", "ur5", "spot", "g1"]

        by_task: dict[str, list[str]] = {t: [] for t in tasks}
        by_embodiment: dict[str, list[str]] = {e: [] for e in embodiments}
        quality_list: list[list] = []

        for i in range(1, n + 1):
            ep_id = f"ep_{i:06d}"
            task_key = tasks[i % len(tasks)]
            emb_key = embodiments[i % len(embodiments)]
            score = round((i % 100) / 100, 2)

            by_task[task_key].append(ep_id)
            by_embodiment[emb_key].append(ep_id)
            quality_list.append([score, ep_id])

        # Sort quality list ascending (as storage layer does)
        quality_list.sort(key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0))

        manifest = {"schema_version": "1.0", "episode_count": n, "last_updated": "2026-03-09T00:00:00Z"}
        (index_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (index_root / "by_task.json").write_text(json.dumps(by_task), encoding="utf-8")
        (index_root / "by_embodiment.json").write_text(json.dumps(by_embodiment), encoding="utf-8")
        (index_root / "quality.json").write_text(json.dumps(quality_list), encoding="utf-8")

        # Time the query (excludes fixture setup)
        start = time.monotonic()
        result = query_index(index_root, task="pick", quality_min=0.8, embodiment="aloha2")
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"query_index() took {elapsed:.3f}s on 100K episodes (limit: 1.0s)"
        # Sanity: results should be a non-empty sorted list
        assert isinstance(result, list)
        assert result == sorted(result)
