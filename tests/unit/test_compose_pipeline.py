"""Unit tests for tq.compose() — the orchestrating pipeline entry point.

Covers:
    - Full pipeline: filter → load → sample → Dataset
    - Recipe populated with all input parameters
    - source_episode_ids (pre-sampling) vs sampled_episode_ids (post-sampling)
    - Low episode count warning (< 5 episodes)
    - Zero episode result: empty Dataset + warning
    - sampling='stratified' wired end-to-end
    - sampling='quality_weighted' wired end-to-end
    - Determinism: same seed → same order
    - name stored in dataset and recipe
    - store_path required (raises when missing)
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import torq as tq
from torq.compose._compose import compose
from torq.episode import Episode
from torq.errors import TorqComposeError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_episode(ep_id: str, task: str = "pick", quality: float | None = 0.8) -> MagicMock:
    ep = MagicMock(spec=Episode)
    ep.episode_id = ep_id
    ep.metadata = {"task": task}
    if quality is None:
        ep.quality = None
    else:
        ep.quality = MagicMock()
        ep.quality.overall = quality
    return ep


def _make_episodes(ids_tasks: list[tuple[str, str, float]]) -> list[MagicMock]:
    return [_mock_episode(ep_id, task, q) for ep_id, task, q in ids_tasks]


# ── store_path required ───────────────────────────────────────────────────────

class TestStorePath:
    def test_missing_store_path_raises(self) -> None:
        # store_path is now a truly required keyword-only argument — Python raises
        # TypeError at the call site, which type checkers can also catch statically.
        with pytest.raises(TypeError, match="store_path"):
            compose()  # type: ignore[call-arg]


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestFullPipeline:
    def test_returns_dataset_with_matched_episodes(self, tmp_path: Path) -> None:
        eps = _make_episodes([
            ("ep-001", "pick", 0.9),
            ("ep-002", "pick", 0.85),
            ("ep-003", "pick", 0.7),
        ])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001", "ep-002", "ep-003"]),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: {e.episode_id: e for e in eps}[eid]),
        ):
            ds = compose(task="pick", store_path=tmp_path)

        assert len(ds) == 3
        assert ds.name == "dataset"

    def test_name_stored_in_dataset(self, tmp_path: Path) -> None:
        eps = _make_episodes([("ep-001", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]),
            patch("torq.compose._compose.load", return_value=eps[0]),
        ):
            ds = compose(name="pick_v1", store_path=tmp_path)

        assert ds.name == "pick_v1"

    def test_name_stored_in_recipe(self, tmp_path: Path) -> None:
        eps = _make_episodes([("ep-001", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]),
            patch("torq.compose._compose.load", return_value=eps[0]),
        ):
            ds = compose(name="my_dataset", store_path=tmp_path)

        assert ds.recipe["name"] == "my_dataset"

    def test_accessible_via_tq_dot_compose(self, tmp_path: Path) -> None:
        """tq.compose should be the same function as compose()."""
        assert tq.compose is compose

    def test_query_index_called_with_correct_args(self, tmp_path: Path) -> None:
        """Verify compose() wires store_path → index_root and passes all filters."""
        eps = _make_episodes([("ep-001", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]) as mock_qi,
            patch("torq.compose._compose.load", return_value=eps[0]),
        ):
            compose(
                task="pick",
                quality_min=0.75,
                quality_max=1.0,
                embodiment="franka",
                store_path=tmp_path,
            )

        mock_qi.assert_called_once_with(
            tmp_path / "index",
            task="pick",
            quality_min=0.75,
            quality_max=1.0,
            embodiment="franka",
        )

    def test_load_called_per_matched_id(self, tmp_path: Path) -> None:
        """Verify compose() calls load() for every episode ID returned by query_index."""
        eps = _make_episodes([("ep-001", "pick", 0.9), ("ep-002", "pick", 0.85)])
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001", "ep-002"]),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]) as mock_load,
        ):
            compose(store_path=tmp_path)

        assert mock_load.call_count == 2
        called_ids = {call.args[0] for call in mock_load.call_args_list}
        assert called_ids == {"ep-001", "ep-002"}


# ── Recipe population ─────────────────────────────────────────────────────────

class TestRecipe:
    def test_all_filter_params_in_recipe(self, tmp_path: Path) -> None:
        eps = _make_episodes([("ep-001", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]),
            patch("torq.compose._compose.load", return_value=eps[0]),
        ):
            ds = compose(
                task="pick",
                quality_min=0.8,
                quality_max=1.0,
                embodiment="franka",
                sampling="none",
                limit=10,
                seed=42,
                name="v1",
                store_path=tmp_path,
            )

        r = ds.recipe
        assert r["task"] == "pick"
        assert r["quality_min"] == 0.8
        assert r["quality_max"] == 1.0
        assert r["embodiment"] == "franka"
        assert r["sampling"] == "none"
        assert r["limit"] == 10
        assert r["seed"] == 42
        assert r["name"] == "v1"

    def test_none_params_present_in_recipe(self, tmp_path: Path) -> None:
        """Even None-valued params must be in the recipe for full provenance."""
        eps = _make_episodes([("ep-001", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]),
            patch("torq.compose._compose.load", return_value=eps[0]),
        ):
            ds = compose(store_path=tmp_path)

        r = ds.recipe
        assert "task" in r
        assert r["task"] is None
        assert "quality_min" in r
        assert r["quality_min"] is None
        assert "embodiment" in r
        assert r["embodiment"] is None

    def test_source_episode_ids_pre_sampling(self, tmp_path: Path) -> None:
        eps = _make_episodes([("ep-001", "pick", 0.9), ("ep-002", "pick", 0.85)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001", "ep-002"]),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: {e.episode_id: e for e in eps}[eid]),
        ):
            ds = compose(sampling="none", limit=1, store_path=tmp_path)

        # source_episode_ids = all matched (before sampling)
        assert set(ds.recipe["source_episode_ids"]) == {"ep-001", "ep-002"}
        # sampled_episode_ids = only the 1 kept after limit
        assert len(ds.recipe["sampled_episode_ids"]) == 1

    def test_sampled_episode_ids_post_sampling(self, tmp_path: Path) -> None:
        eps = _make_episodes([
            ("ep-001", "pick", 0.9),
            ("ep-002", "pick", 0.85),
            ("ep-003", "pick", 0.7),
        ])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001", "ep-002", "ep-003"]),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: {e.episode_id: e for e in eps}[eid]),
        ):
            ds = compose(sampling="none", limit=2, store_path=tmp_path)

        assert len(ds.recipe["sampled_episode_ids"]) == 2
        assert len(ds.recipe["source_episode_ids"]) == 3


# ── Warnings ──────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_zero_episodes_after_query_returns_empty_dataset(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch("torq.compose._compose.query_index", return_value=[]),
            caplog.at_level(logging.WARNING, logger="torq.compose._compose"),
        ):
            ds = compose(task="nonexistent", store_path=tmp_path)

        assert len(ds) == 0
        assert any("0 episodes" in r.message or "filter" in r.message.lower() for r in caplog.records)

    def test_low_episode_count_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        eps = _make_episodes([("ep-001", "pick", 0.95), ("ep-002", "pick", 0.9)])
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001", "ep-002"]),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: {e.episode_id: e for e in eps}[eid]),
            caplog.at_level(logging.WARNING, logger="torq.compose._compose"),
        ):
            ds = compose(quality_min=0.9, store_path=tmp_path)

        # Dataset is returned (no exception)
        assert len(ds) == 2
        # Warning logged about low count
        assert any("episode" in r.message.lower() for r in caplog.records)

    def test_zero_episodes_after_sampling_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If query returns episodes but sampling reduces to 0, warn."""
        eps = _make_episodes([("ep-001", "pick", None)])  # no quality score
        with (
            patch("torq.compose._compose.query_index", return_value=["ep-001"]),
            patch("torq.compose._compose.load", return_value=eps[0]),
            caplog.at_level(logging.WARNING, logger="torq.compose._compose"),
        ):
            ds = compose(sampling="quality_weighted", store_path=tmp_path)

        assert len(ds) == 0
        assert any("sampling" in r.message.lower() for r in caplog.records)


# ── Sampling wired end-to-end ─────────────────────────────────────────────────

class TestSamplingWired:
    def test_stratified_wired(self, tmp_path: Path) -> None:
        eps = _make_episodes([
            ("ep-001", "pick", 0.8),
            ("ep-002", "pick", 0.8),
            ("ep-003", "place", 0.8),
            ("ep-004", "place", 0.8),
        ])
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=sorted(ep_map)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            ds = compose(sampling="stratified", limit=2, seed=0, store_path=tmp_path)

        tasks = [ep.metadata["task"] for ep in ds]
        assert tasks.count("pick") == 1
        assert tasks.count("place") == 1

    def test_quality_weighted_wired(self, tmp_path: Path) -> None:
        high = [_mock_episode(f"h{i}", "pick", 0.95) for i in range(8)]
        low = [_mock_episode(f"l{i}", "pick", 0.05) for i in range(8)]
        all_eps = high + low
        ep_map = {e.episode_id: e for e in all_eps}
        with (
            patch("torq.compose._compose.query_index", return_value=sorted(ep_map)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            ds = compose(sampling="quality_weighted", limit=4, seed=99, store_path=tmp_path)

        high_count = sum(1 for ep in ds if ep in high)
        assert high_count >= 3, f"Expected mostly high-quality, got {high_count}/4"


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_produces_same_order(self, tmp_path: Path) -> None:
        eps = _make_episodes([(f"ep-{i:03d}", "pick", 0.5 + i * 0.05) for i in range(10)])
        ep_map = {e.episode_id: e for e in eps}
        ids = sorted(ep_map)

        def _run():
            with (
                patch("torq.compose._compose.query_index", return_value=ids),
                patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
            ):
                return compose(sampling="quality_weighted", limit=5, seed=42, store_path=tmp_path)

        ds1 = _run()
        ds2 = _run()
        assert [ep.episode_id for ep in ds1] == [ep.episode_id for ep in ds2]
