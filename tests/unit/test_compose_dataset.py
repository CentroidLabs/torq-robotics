"""Unit tests for torq.compose.Dataset.

Covers:
    - __len__: returns episode count
    - __iter__: yields all episodes in order
    - __repr__: correct format with name, count, quality_avg
    - quality_avg with scored episodes
    - quality_avg with all-unscored episodes (N/A)
    - quality_avg with mixed scored/unscored (computed from scored only)
    - quality_avg with empty dataset (0 episodes, N/A)
    - recipe attribute returns exact dict passed at construction
    - name attribute returns exact string passed at construction
    - episodes attribute holds the passed list
    - __getitem__ for integer index and slice
    - __contains__ membership test
"""

from unittest.mock import MagicMock

import pytest

from torq.compose.dataset import Dataset
from torq.episode import Episode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(quality_overall: float | None = None) -> MagicMock:
    """Return a minimal Episode-like mock with optional quality.overall.

    Uses ``spec=Episode`` so that attribute typos in production code (e.g.
    ``ep.quality.overal``) are caught as ``AttributeError`` rather than
    silently returning a new MagicMock.
    """
    ep = MagicMock(spec=Episode)
    if quality_overall is None:
        ep.quality = None
    else:
        ep.quality = MagicMock()
        ep.quality.overall = quality_overall
    return ep


def _make_episodes(scores: list[float | None]) -> list:
    return [_make_episode(s) for s in scores]


# ── __len__ ───────────────────────────────────────────────────────────────────

class TestLen:
    def test_len_returns_episode_count(self) -> None:
        eps = _make_episodes([0.8, 0.9, 0.7])
        ds = Dataset(episodes=eps, name="pick_v1", recipe={})
        assert len(ds) == 3

    def test_len_empty_dataset(self) -> None:
        ds = Dataset(episodes=[], name="empty", recipe={})
        assert len(ds) == 0


# ── __iter__ ─────────────────────────────────────────────────────────────────

class TestIter:
    def test_iter_yields_all_episodes_in_order(self) -> None:
        eps = _make_episodes([0.8, 0.9, 0.7])
        ds = Dataset(episodes=eps, name="pick_v1", recipe={})
        assert list(iter(ds)) == eps

    def test_iter_empty_dataset_yields_nothing(self) -> None:
        ds = Dataset(episodes=[], name="empty", recipe={})
        assert list(iter(ds)) == []

    def test_iter_can_be_called_multiple_times(self) -> None:
        eps = _make_episodes([0.5, 0.6])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert list(ds) == list(ds)


# ── __getitem__ ───────────────────────────────────────────────────────────────

class TestGetitem:
    def test_getitem_returns_correct_episode(self) -> None:
        eps = _make_episodes([0.8, 0.9, 0.7])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds[0] is eps[0]
        assert ds[1] is eps[1]
        assert ds[2] is eps[2]

    def test_getitem_negative_index(self) -> None:
        eps = _make_episodes([0.8, 0.9])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds[-1] is eps[-1]

    def test_getitem_slice_returns_list(self) -> None:
        eps = _make_episodes([0.8, 0.9, 0.7, 0.6, 0.5])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds[0:3] == eps[0:3]

    def test_getitem_slice_empty(self) -> None:
        eps = _make_episodes([0.8, 0.9])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds[5:10] == []

    def test_getitem_slice_step(self) -> None:
        eps = _make_episodes([0.8, 0.9, 0.7, 0.6])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds[::2] == eps[::2]


# ── __repr__ ─────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_format_with_scored_episodes(self) -> None:
        """Exact format: Dataset('name', N episodes, quality_avg=X.XX)"""
        eps = _make_episodes([0.80, 0.82])
        ds = Dataset(episodes=eps, name="pick_v1", recipe={})
        assert repr(ds) == "Dataset('pick_v1', 2 episodes, quality_avg=0.81)"

    def test_repr_quality_avg_computed_correctly(self) -> None:
        eps = _make_episodes([0.60, 0.80, 1.00])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        # avg = (0.60 + 0.80 + 1.00) / 3 = 0.80
        assert repr(ds) == "Dataset('v1', 3 episodes, quality_avg=0.80)"

    def test_repr_with_all_unscored_shows_na(self) -> None:
        eps = _make_episodes([None, None])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert repr(ds) == "Dataset('v1', 2 episodes, quality_avg=N/A)"

    def test_repr_with_mixed_scored_and_unscored_uses_scored_only(self) -> None:
        eps = _make_episodes([0.70, None, 0.90])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        # avg = (0.70 + 0.90) / 2 = 0.80
        assert repr(ds) == "Dataset('v1', 3 episodes, quality_avg=0.80)"

    def test_repr_empty_dataset(self) -> None:
        ds = Dataset(episodes=[], name="empty_v1", recipe={})
        assert repr(ds) == "Dataset('empty_v1', 0 episodes, quality_avg=N/A)"

    def test_repr_contains_name(self) -> None:
        ds = Dataset(episodes=[], name="my_dataset", recipe={})
        assert "'my_dataset'" in repr(ds)

    def test_repr_contains_episode_count(self) -> None:
        eps = _make_episodes([0.5] * 31)
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert "31 episodes" in repr(ds)

    def test_repr_rounds_to_two_decimal_places(self) -> None:
        # 1/3 = 0.333... → rounds to 0.33
        eps = _make_episodes([0.0, 0.0, 1.0])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert repr(ds) == "Dataset('v1', 3 episodes, quality_avg=0.33)"


# ── recipe / name / episodes attributes ──────────────────────────────────────

class TestAttributes:
    def test_recipe_returns_exact_dict(self) -> None:
        recipe = {"task": "pick", "quality_min": 0.8, "seed": 42}
        ds = Dataset(episodes=[], name="v1", recipe=recipe)
        assert ds.recipe is recipe

    def test_name_returns_exact_string(self) -> None:
        ds = Dataset(episodes=[], name="pick_place_v3", recipe={})
        assert ds.name == "pick_place_v3"

    def test_episodes_holds_passed_list(self) -> None:
        eps = _make_episodes([0.7, 0.8])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert ds.episodes is eps

    def test_recipe_defaults_to_empty_dict(self) -> None:
        ds = Dataset(episodes=[], name="v1")
        assert ds.recipe == {}

    def test_dataset_importable_from_top_level(self) -> None:
        import torq as tq
        assert tq.Dataset is Dataset


# ── __contains__ ──────────────────────────────────────────────────────────────

class TestContains:
    def test_contains_episode_present(self) -> None:
        eps = _make_episodes([0.8, 0.9])
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert eps[0] in ds
        assert eps[1] in ds

    def test_contains_episode_absent(self) -> None:
        eps = _make_episodes([0.8])
        other = _make_episode(0.5)
        ds = Dataset(episodes=eps, name="v1", recipe={})
        assert other not in ds

    def test_contains_empty_dataset(self) -> None:
        ep = _make_episode(0.8)
        ds = Dataset(episodes=[], name="v1", recipe={})
        assert ep not in ds
