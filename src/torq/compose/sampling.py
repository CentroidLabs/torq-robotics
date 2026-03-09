"""Sampling strategies for dataset composition.

Provides the ``sample()`` entry point which applies one of three strategies
to a list of Episodes:

- ``'none'`` — pass-through, preserving insertion order
- ``'stratified'`` — balance across task groups (equal quota per task)
- ``'quality_weighted'`` — oversample high-quality episodes proportionally

All strategies are deterministic when ``seed`` is provided.  ``random.Random``
is used instead of the module-level ``random.seed()`` to avoid polluting the
caller's random state.
"""

from __future__ import annotations

import logging
import random as _random
from collections import defaultdict
from typing import TYPE_CHECKING

from torq.errors import TorqComposeError

if TYPE_CHECKING:
    from torq.episode import Episode

__all__ = ["sample"]

logger = logging.getLogger(__name__)

# Small epsilon added to quality weights so that episodes with quality=0.0 are
# still eligible for selection (though with very low probability relative to
# higher-quality episodes).  Without this, weight * rng.random() == 0.0
# always, meaning zero-quality episodes can never win the weighted shuffle.
_WEIGHT_EPS: float = 1e-9

VALID_STRATEGIES: frozenset[str] = frozenset({"none", "stratified", "quality_weighted"})


def sample(
    episodes: list[Episode],
    strategy: str,
    limit: int | None = None,
    seed: int | None = None,
) -> list[Episode]:
    """Apply a sampling strategy to a list of Episodes.

    Args:
        episodes: Input episode list to sample from.
        strategy: One of ``'none'``, ``'stratified'``, or ``'quality_weighted'``.
        limit: Maximum number of episodes to return.  When ``None``, all
            episodes are returned (subject to strategy constraints).
            When ``limit > len(episodes)``, it is silently capped at
            ``len(episodes)`` — never raises.
        seed: Random seed for deterministic output.  Uses ``random.Random(seed)``
            so the global random state is not affected.

    Returns:
        List of sampled Episodes.

    Raises:
        TorqComposeError: If ``strategy`` is not one of the valid options.
    """
    if strategy not in VALID_STRATEGIES:
        valid = ", ".join(f"'{s}'" for s in sorted(VALID_STRATEGIES))
        raise TorqComposeError(
            f"Unknown sampling strategy {strategy!r}. "
            f"Valid strategies are: {valid}. "
            f"Pass one of these as the strategy argument to sample()."
        )

    if not episodes:
        return []

    # Cap limit silently
    effective_limit = min(limit, len(episodes)) if limit is not None else None

    if strategy == "none":
        return _none(episodes, effective_limit)
    if strategy == "stratified":
        return _stratified(episodes, effective_limit, seed)
    # strategy == "quality_weighted"
    return _quality_weighted(episodes, effective_limit, seed)


# ── Strategy implementations ──────────────────────────────────────────────────

def _none(episodes: list[Episode], limit: int | None) -> list[Episode]:
    """Return episodes as-is, optionally truncated to limit."""
    return episodes[:limit] if limit is not None else episodes


def _stratified(
    episodes: list[Episode],
    limit: int | None,
    seed: int | None,
) -> list[Episode]:
    """Balance episodes across task groups with equal quota per group.

    Groups are sorted by task name for determinism.  Remainder episodes
    are distributed to the first N groups (round-robin).  When a group
    has fewer members than its quota, the unallocated slots are
    redistributed to groups that still have capacity (in sorted order).
    """
    rng = _random.Random(seed)

    groups: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        task = ep.metadata.get("task", "") if ep.metadata else ""
        groups[task].append(ep)

    n_groups = len(groups)
    if n_groups == 0:
        return []

    effective = limit if limit is not None else len(episodes)
    base = effective // n_groups
    remainder = effective % n_groups

    # Shuffle each group up front for determinism
    shuffled_groups: list[list[Episode]] = []
    for _task, group_eps in sorted(groups.items()):
        shuffled = list(group_eps)
        rng.shuffle(shuffled)
        shuffled_groups.append(shuffled)

    # First pass: assign initial quotas, capped by group size
    taken = [
        min(base + (1 if i < remainder else 0), len(g))
        for i, g in enumerate(shuffled_groups)
    ]

    # Second pass: redistribute any unallocated slots to groups with spare capacity
    slots_remaining = effective - sum(taken)
    for i, group in enumerate(shuffled_groups):
        if slots_remaining <= 0:
            break
        can_give = len(group) - taken[i]
        if can_give > 0:
            extra = min(can_give, slots_remaining)
            taken[i] += extra
            slots_remaining -= extra

    result: list[Episode] = []
    for group, n in zip(shuffled_groups, taken):
        result.extend(group[:n])
    return result


def _quality_weighted(
    episodes: list[Episode],
    limit: int | None,
    seed: int | None,
) -> list[Episode]:
    """Sample episodes without replacement, proportional to quality score.

    Episodes with ``None`` quality score are excluded entirely.  If no
    episodes have quality scores a warning is logged and an empty list is
    returned.

    Uses a weighted-shuffle approach: each episode is assigned a random
    priority scaled by its quality score, then the top-k are selected.
    This gives proportional sampling WITHOUT replacement.
    """
    rng = _random.Random(seed)

    scored = [
        (ep, ep.quality.overall)
        for ep in episodes
        if ep.quality is not None and ep.quality.overall is not None
    ]

    if not scored:
        logger.warning(
            "sample(strategy='quality_weighted') found 0 episodes with a quality score. "
            "Run tq.quality.score() on your episodes before composing a quality-weighted dataset."
        )
        return []

    k = min(limit, len(scored)) if limit is not None else len(scored)

    # Weighted shuffle: priority = (quality_score + eps) * uniform_random → sort descending.
    # The epsilon ensures episodes with quality=0.0 are still eligible (non-zero priority).
    weighted = sorted(
        (((weight + _WEIGHT_EPS) * rng.random(), ep) for ep, weight in scored),
        key=lambda x: x[0],
        reverse=True,
    )
    return [ep for _, ep in weighted[:k]]
