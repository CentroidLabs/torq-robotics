"""Filter predicate functions for the Torq compose layer.

Each function takes the current set of episode IDs and a filter argument,
returning the subset of IDs that satisfy the filter.  When no filter is
specified (argument is ``None``), all IDs in the input are returned unchanged.

String normalisation is delegated to ``storage.index._normalise`` to avoid
duplication — both modules apply the same transform:
    lowercase + strip + remove separator characters (hyphens, spaces, underscores).
"""

from __future__ import annotations

import bisect

from torq.storage.index import _normalise as normalise

__all__ = [
    "normalise",
    "apply_task_filter",
    "apply_embodiment_filter",
    "apply_quality_filter",
]


def apply_task_filter(
    episode_ids: list[str],
    task: str | list[str] | None,
    by_task: dict[str, list[str]],
) -> set[str]:
    """Filter episode IDs by task name(s).

    Args:
        episode_ids: Current universe of episode IDs to filter.
        task: Task name string (exact match) or list of names (OR logic).
            ``None`` returns all input IDs unchanged.
        by_task: Mapping of normalised task key → list of episode IDs,
            as loaded from ``by_task.json``.

    Returns:
        Subset of ``episode_ids`` whose task matches any of the specified tasks.
        Returns the full set if ``task`` is ``None``.
    """
    if task is None:
        return set(episode_ids)
    task_keys = [task] if isinstance(task, str) else list(task)
    matching: set[str] = set()
    for t in task_keys:
        matching |= set(by_task.get(normalise(t), []))
    return set(episode_ids) & matching


def apply_embodiment_filter(
    episode_ids: list[str],
    embodiment: str | list[str] | None,
    by_embodiment: dict[str, list[str]],
) -> set[str]:
    """Filter episode IDs by embodiment name(s).

    Args:
        episode_ids: Current universe of episode IDs to filter.
        embodiment: Embodiment name string or list of names (OR logic).
            ``None`` returns all input IDs unchanged.
        by_embodiment: Mapping of normalised embodiment key → list of episode IDs,
            as loaded from ``by_embodiment.json``.

    Returns:
        Subset of ``episode_ids`` whose embodiment matches any of the specified values.
        Returns the full set if ``embodiment`` is ``None``.
    """
    if embodiment is None:
        return set(episode_ids)
    embodiment_keys = [embodiment] if isinstance(embodiment, str) else list(embodiment)
    matching: set[str] = set()
    for e in embodiment_keys:
        matching |= set(by_embodiment.get(normalise(e), []))
    return set(episode_ids) & matching


def apply_quality_filter(
    episode_ids: list[str],
    quality_min: float | None,
    quality_max: float | None,
    quality_list: list[list],
) -> set[str]:
    """Filter episode IDs by quality score range using binary search.

    Uses ``bisect`` on the pre-sorted ``quality_list`` (ascending scores, nulls last)
    to locate the matching range in O(log n) instead of a full linear scan.

    Episodes with ``None`` quality score are excluded whenever any quality filter
    is active (i.e. ``quality_min`` or ``quality_max`` is not ``None``).

    Args:
        episode_ids: Current universe of episode IDs to filter.
        quality_min: Minimum quality score (inclusive). ``None`` = no lower bound.
        quality_max: Maximum quality score (inclusive). ``None`` = no upper bound.
        quality_list: Sorted list of ``[score_or_null, episode_id]`` pairs,
            as loaded from ``quality.json``. Must be sorted ascending with nulls last.

    Returns:
        Subset of ``episode_ids`` whose quality score falls within the specified range.
        Returns the full set if both ``quality_min`` and ``quality_max`` are ``None``.
    """
    if quality_min is None and quality_max is None:
        return set(episode_ids)

    # Separate scored entries from null-score entries (nulls are at the end).
    # Build a parallel list of scores for bisect operations.
    scored = [(entry[0], entry[1]) for entry in quality_list if entry[0] is not None]
    scores = [s for s, _ in scored]

    lo = bisect.bisect_left(scores, quality_min) if quality_min is not None else 0
    hi = bisect.bisect_right(scores, quality_max) if quality_max is not None else len(scores)

    valid_ids = {ep_id for _, ep_id in scored[lo:hi]}
    return set(episode_ids) & valid_ids
