"""Unit tests for torq.compose.sampling — sample() function.

Covers:
    - strategy='none' with and without limit
    - strategy='stratified' with 3 unequal task groups, limit, no limit
    - strategy='stratified' total == limit (no rounding shortfall)
    - strategy='quality_weighted' favours high-quality episodes
    - strategy='quality_weighted' excludes None quality
    - Determinism: same seed → identical output
    - Different seeds → different outputs
    - limit > len(episodes): capped silently
    - Invalid strategy raises TorqComposeError
"""

import logging
from unittest.mock import MagicMock

import pytest

from torq.compose.sampling import sample
from torq.episode import Episode
from torq.errors import TorqComposeError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ep(task: str = "pick", quality: float | None = 0.8) -> MagicMock:
    ep = MagicMock(spec=Episode)
    ep.metadata = {"task": task}
    if quality is None:
        ep.quality = None
    else:
        ep.quality = MagicMock()
        ep.quality.overall = quality
    return ep


def _eps(task: str, n: int, quality: float = 0.8) -> list:
    return [_ep(task=task, quality=quality) for _ in range(n)]


# ── strategy='none' ───────────────────────────────────────────────────────────

class TestStrategyNone:
    def test_no_limit_returns_all(self) -> None:
        eps = _eps("pick", 5)
        result = sample(eps, strategy="none")
        assert result == eps

    def test_with_limit_returns_first_n(self) -> None:
        eps = _eps("pick", 10)
        result = sample(eps, strategy="none", limit=4)
        assert result == eps[:4]

    def test_preserves_order(self) -> None:
        eps = [_ep(task="pick", quality=float(i)) for i in range(5)]
        result = sample(eps, strategy="none", limit=3)
        assert result == eps[:3]

    def test_limit_exceeds_length_returns_all(self) -> None:
        eps = _eps("pick", 3)
        result = sample(eps, strategy="none", limit=100)
        assert result == eps

    def test_empty_input_returns_empty(self) -> None:
        assert sample([], strategy="none") == []


# ── strategy='stratified' ─────────────────────────────────────────────────────

class TestStrategyStratified:
    def test_three_equal_groups_with_limit(self) -> None:
        eps = _eps("pick", 50) + _eps("place", 20) + _eps("pour", 10)
        result = sample(eps, strategy="stratified", limit=30, seed=42)
        assert len(result) == 30

    def test_each_task_gets_roughly_equal_quota(self) -> None:
        """pick:50, place:20, pour:10 → each group gets ~10 (limit=30)."""
        eps = _eps("pick", 50) + _eps("place", 20) + _eps("pour", 10)
        result = sample(eps, strategy="stratified", limit=30, seed=42)
        tasks = [ep.metadata["task"] for ep in result]
        for task in ("pick", "place", "pour"):
            count = tasks.count(task)
            assert 9 <= count <= 11, f"task={task!r} got {count} (expected ~10)"

    def test_total_equals_limit_exactly(self) -> None:
        """No rounding shortfall — result length must equal limit."""
        eps = _eps("a", 20) + _eps("b", 20) + _eps("c", 20)
        result = sample(eps, strategy="stratified", limit=7, seed=0)
        assert len(result) == 7

    def test_no_limit_returns_all_grouped(self) -> None:
        eps = _eps("pick", 5) + _eps("place", 5)
        result = sample(eps, strategy="stratified", seed=1)
        assert len(result) == 10

    def test_limit_exceeds_total_capped(self) -> None:
        eps = _eps("pick", 3) + _eps("place", 3)
        result = sample(eps, strategy="stratified", limit=100, seed=0)
        assert len(result) == 6

    def test_empty_input_returns_empty(self) -> None:
        assert sample([], strategy="stratified", limit=10) == []

    def test_single_task_group_with_limit(self) -> None:
        """Single group — stratified should behave like _none with limit."""
        eps = _eps("pick", 20)
        result = sample(eps, strategy="stratified", limit=5, seed=0)
        assert len(result) == 5
        assert all(ep in eps for ep in result)

    def test_single_task_group_no_limit(self) -> None:
        """Single group without limit — all episodes returned."""
        eps = _eps("pick", 7)
        result = sample(eps, strategy="stratified", seed=0)
        assert len(result) == 7

    def test_small_group_slots_redistributed(self) -> None:
        """pick:50, place:20, pour:3, limit=30 — unallocated slots go to larger groups."""
        eps = _eps("pick", 50) + _eps("place", 20) + _eps("pour", 3)
        result = sample(eps, strategy="stratified", limit=30, seed=42)
        # Total must be exactly limit (not 23 as the naive algorithm would give)
        assert len(result) == 30
        tasks = [ep.metadata["task"] for ep in result]
        # pour contributes all 3; remaining 27 slots distributed to pick/place
        assert tasks.count("pour") == 3
        assert tasks.count("pick") + tasks.count("place") == 27


# ── strategy='quality_weighted' ──────────────────────────────────────────────

class TestStrategyQualityWeighted:
    def test_excludes_none_quality_episodes(self) -> None:
        eps_scored = _eps("pick", 5, quality=0.9)
        eps_unscored = [_ep(task="pick", quality=None) for _ in range(5)]
        result = sample(eps_scored + eps_unscored, strategy="quality_weighted", seed=0)
        assert all(ep in eps_scored for ep in result)
        assert not any(ep in eps_unscored for ep in result)

    def test_all_unscored_returns_empty_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        eps = [_ep(quality=None) for _ in range(5)]
        with caplog.at_level(logging.WARNING, logger="torq.compose.sampling"):
            result = sample(eps, strategy="quality_weighted")
        assert result == []
        assert any("quality score" in r.message for r in caplog.records)

    def test_high_quality_sampled_more_often(self) -> None:
        """Statistical test: across many trials, high-quality episodes win more."""
        high = [_ep(task="pick", quality=0.95) for _ in range(10)]
        low = [_ep(task="pick", quality=0.05) for _ in range(10)]
        eps = high + low

        high_counts = 0
        trials = 200
        for seed in range(trials):
            result = sample(eps, strategy="quality_weighted", limit=5, seed=seed)
            high_counts += sum(1 for ep in result if ep in high)

        high_rate = high_counts / (trials * 5)
        assert high_rate > 0.7, f"Expected high-quality to dominate (got {high_rate:.2f})"

    def test_with_limit(self) -> None:
        eps = _eps("pick", 20, quality=0.8)
        result = sample(eps, strategy="quality_weighted", limit=10, seed=7)
        assert len(result) == 10

    def test_limit_exceeds_scored_episodes_capped(self) -> None:
        eps = _eps("pick", 5, quality=0.8)
        result = sample(eps, strategy="quality_weighted", limit=100, seed=0)
        assert len(result) == 5

    def test_no_duplicates_in_result(self) -> None:
        eps = _eps("pick", 10, quality=0.8)
        result = sample(eps, strategy="quality_weighted", limit=10, seed=42)
        assert len(result) == len(set(id(ep) for ep in result))

    def test_empty_input_returns_empty(self) -> None:
        assert sample([], strategy="quality_weighted") == []

    def test_zero_quality_episodes_are_eligible(self) -> None:
        """Episodes with quality.overall=0.0 must still be selectable (not permanently excluded)."""
        zero_quality = [_ep(task="pick", quality=0.0) for _ in range(10)]
        # With only zero-quality episodes, all should be returned (no exclusion)
        result = sample(zero_quality, strategy="quality_weighted", limit=5, seed=0)
        assert len(result) == 5
        assert all(ep in zero_quality for ep in result)


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_seed_stratified_identical_output(self) -> None:
        eps = _eps("pick", 20) + _eps("place", 20)
        r1 = sample(eps, strategy="stratified", limit=10, seed=42)
        r2 = sample(eps, strategy="stratified", limit=10, seed=42)
        assert r1 == r2

    def test_same_seed_quality_weighted_identical_output(self) -> None:
        eps = [_ep(task="pick", quality=round(i * 0.1, 1)) for i in range(1, 11)]
        r1 = sample(eps, strategy="quality_weighted", limit=5, seed=99)
        r2 = sample(eps, strategy="quality_weighted", limit=5, seed=99)
        assert r1 == r2

    def test_different_seeds_produce_different_output(self) -> None:
        eps = _eps("pick", 20) + _eps("place", 20)
        r1 = sample(eps, strategy="stratified", limit=10, seed=1)
        r2 = sample(eps, strategy="stratified", limit=10, seed=2)
        # Same elements, different order (very high probability for n=20)
        assert r1 != r2


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_invalid_strategy_raises_compose_error(self) -> None:
        with pytest.raises(TorqComposeError, match="Unknown sampling strategy"):
            sample(_eps("pick", 5), strategy="random")

    def test_invalid_strategy_message_lists_valid_options(self) -> None:
        with pytest.raises(TorqComposeError, match="none.*quality_weighted.*stratified"):
            sample(_eps("pick", 5), strategy="bad")
