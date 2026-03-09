"""Unit tests for torq.compose.filters — filter predicate functions.

Covers:
    - normalise()
    - apply_task_filter() — single task, list of tasks (OR), None, unknown task
    - apply_embodiment_filter() — same patterns
    - apply_quality_filter() — min, max, range, None scores, combined
    - Set intersection: task ∩ embodiment ∩ quality applied simultaneously
"""

import pytest

from torq.compose.filters import (
    apply_embodiment_filter,
    apply_quality_filter,
    apply_task_filter,
    normalise,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def episode_ids() -> list[str]:
    return ["ep_0001", "ep_0002", "ep_0003", "ep_0004", "ep_0005"]


@pytest.fixture
def by_task() -> dict[str, list[str]]:
    return {
        "pick": ["ep_0001", "ep_0002"],
        "place": ["ep_0003", "ep_0004"],
        "pickandplace": ["ep_0005"],
    }


@pytest.fixture
def by_embodiment() -> dict[str, list[str]]:
    return {
        "aloha2": ["ep_0001", "ep_0003"],
        "franka": ["ep_0002", "ep_0004"],
        "ur5": ["ep_0005"],
    }


@pytest.fixture
def quality_list() -> list[list]:
    """Sorted ascending by score; ep_0005 has no quality score."""
    return [
        [0.60, "ep_0001"],
        [0.72, "ep_0002"],
        [0.80, "ep_0003"],
        [0.91, "ep_0004"],
        [None, "ep_0005"],
    ]


# ── normalise() ───────────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercase(self) -> None:
        assert normalise("Pick") == "pick"

    def test_strip_whitespace(self) -> None:
        assert normalise("  pick  ") == "pick"

    def test_remove_hyphens(self) -> None:
        assert normalise("pick-and-place") == "pickandplace"

    def test_remove_underscores(self) -> None:
        assert normalise("pick_place") == "pickplace"

    def test_remove_spaces(self) -> None:
        assert normalise("pick place") == "pickplace"

    def test_combined(self) -> None:
        assert normalise("  ALOHA-2  ") == "aloha2"


# ── apply_task_filter() ───────────────────────────────────────────────────────

class TestApplyTaskFilter:
    def test_single_task_string_returns_matching_ids(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        result = apply_task_filter(episode_ids, "pick", by_task)
        assert result == {"ep_0001", "ep_0002"}

    def test_list_of_tasks_returns_union(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        result = apply_task_filter(episode_ids, ["pick", "place"], by_task)
        assert result == {"ep_0001", "ep_0002", "ep_0003", "ep_0004"}

    def test_none_filter_returns_all_ids(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        result = apply_task_filter(episode_ids, None, by_task)
        assert result == set(episode_ids)

    def test_unknown_task_returns_empty_set(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        result = apply_task_filter(episode_ids, "grasp", by_task)
        assert result == set()

    def test_task_not_in_input_ids_is_excluded(self, by_task: dict) -> None:
        """If the matching episode is not in the input universe, it's excluded."""
        result = apply_task_filter(["ep_0001"], "pick", by_task)
        # ep_0002 is in by_task["pick"] but not in input — excluded
        assert result == {"ep_0001"}

    def test_normalisation_applied_to_query_task(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        """Task query is normalised before lookup."""
        result = apply_task_filter(episode_ids, "Pick", by_task)
        assert result == {"ep_0001", "ep_0002"}

    def test_list_with_single_task_same_as_string(
        self, episode_ids: list[str], by_task: dict
    ) -> None:
        result_list = apply_task_filter(episode_ids, ["pick"], by_task)
        result_str = apply_task_filter(episode_ids, "pick", by_task)
        assert result_list == result_str


# ── apply_embodiment_filter() ─────────────────────────────────────────────────

class TestApplyEmbodimentFilter:
    def test_single_embodiment_returns_matching_ids(
        self, episode_ids: list[str], by_embodiment: dict
    ) -> None:
        result = apply_embodiment_filter(episode_ids, "aloha2", by_embodiment)
        assert result == {"ep_0001", "ep_0003"}

    def test_list_of_embodiments_returns_union(
        self, episode_ids: list[str], by_embodiment: dict
    ) -> None:
        result = apply_embodiment_filter(episode_ids, ["aloha2", "franka"], by_embodiment)
        assert result == {"ep_0001", "ep_0002", "ep_0003", "ep_0004"}

    def test_none_filter_returns_all_ids(
        self, episode_ids: list[str], by_embodiment: dict
    ) -> None:
        result = apply_embodiment_filter(episode_ids, None, by_embodiment)
        assert result == set(episode_ids)

    def test_unknown_embodiment_returns_empty_set(
        self, episode_ids: list[str], by_embodiment: dict
    ) -> None:
        result = apply_embodiment_filter(episode_ids, "spot", by_embodiment)
        assert result == set()

    def test_normalisation_applied(
        self, episode_ids: list[str], by_embodiment: dict
    ) -> None:
        result = apply_embodiment_filter(episode_ids, "ALOHA-2", by_embodiment)
        assert result == {"ep_0001", "ep_0003"}


# ── apply_quality_filter() ────────────────────────────────────────────────────

class TestApplyQualityFilter:
    def test_min_only_excludes_below_threshold(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        result = apply_quality_filter(episode_ids, 0.75, None, quality_list)
        # ep_0001=0.60, ep_0002=0.72 excluded; ep_0003=0.80, ep_0004=0.91 included
        # ep_0005=None excluded
        assert result == {"ep_0003", "ep_0004"}

    def test_max_only_excludes_above_threshold(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        result = apply_quality_filter(episode_ids, None, 0.75, quality_list)
        # ep_0003=0.80, ep_0004=0.91 excluded; ep_0001=0.60, ep_0002=0.72 included
        # ep_0005=None excluded
        assert result == {"ep_0001", "ep_0002"}

    def test_range_returns_episodes_within_bounds(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        result = apply_quality_filter(episode_ids, 0.70, 0.85, quality_list)
        # ep_0002=0.72, ep_0003=0.80 in range; ep_0001=0.60 below, ep_0004=0.91 above
        assert result == {"ep_0002", "ep_0003"}

    def test_none_scores_excluded_when_filter_active(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        result = apply_quality_filter(episode_ids, 0.0, None, quality_list)
        # ep_0005 has None score — excluded even though 0.0 min would include all numeric scores
        assert "ep_0005" not in result
        assert {"ep_0001", "ep_0002", "ep_0003", "ep_0004"}.issubset(result)

    def test_no_filter_returns_all_ids(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        result = apply_quality_filter(episode_ids, None, None, quality_list)
        assert result == set(episode_ids)

    def test_boundary_inclusive(
        self, episode_ids: list[str], quality_list: list
    ) -> None:
        """min and max bounds are inclusive."""
        result = apply_quality_filter(episode_ids, 0.60, 0.60, quality_list)
        assert result == {"ep_0001"}

    def test_empty_input_ids_returns_empty(self, quality_list: list) -> None:
        result = apply_quality_filter([], 0.5, None, quality_list)
        assert result == set()


# ── Set intersection (combined filters) ───────────────────────────────────────

class TestCombinedFilters:
    def test_task_and_embodiment_intersection(
        self,
        episode_ids: list[str],
        by_task: dict,
        by_embodiment: dict,
    ) -> None:
        """task='pick' gives {ep_0001, ep_0002}; embodiment='aloha2' gives {ep_0001, ep_0003}.
        Intersection = {ep_0001}."""
        after_task = apply_task_filter(episode_ids, "pick", by_task)
        result = apply_embodiment_filter(list(after_task), "aloha2", by_embodiment)
        assert result == {"ep_0001"}

    def test_task_embodiment_quality_intersection(
        self,
        episode_ids: list[str],
        by_task: dict,
        by_embodiment: dict,
        quality_list: list,
    ) -> None:
        """task=['pick','place'] gives {ep_0001..ep_0004}; embodiment='franka' gives
        {ep_0002, ep_0004}; quality_min=0.80 gives {ep_0003, ep_0004}.
        Intersection = {ep_0004}."""
        after_task = apply_task_filter(episode_ids, ["pick", "place"], by_task)
        after_embodiment = apply_embodiment_filter(list(after_task), "franka", by_embodiment)
        result = apply_quality_filter(list(after_embodiment), 0.80, None, quality_list)
        assert result == {"ep_0004"}
