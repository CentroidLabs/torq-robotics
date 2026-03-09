"""tq.compose() — filter, sample, and version a training dataset in one call.

Orchestrates the full composition pipeline:

    query_index(store_path) → list[episode_id]
    load(ep_id, store_path) → list[Episode]
    sample(episodes, strategy, limit, seed) → list[Episode]
    Dataset(episodes, name, recipe)
"""

from __future__ import annotations

import logging
from pathlib import Path

from torq._gravity_well import _gravity_well
from torq.compose.dataset import Dataset
from torq.compose.sampling import sample
from torq.storage import load
from torq.storage.index import query_index

__all__ = ["compose"]

logger = logging.getLogger(__name__)

_LOW_EPISODE_THRESHOLD = 5


def compose(
    task: str | list[str] | None = None,
    quality_min: float | None = None,
    quality_max: float | None = None,
    embodiment: str | list[str] | None = None,
    sampling: str = "none",
    limit: int | None = None,
    seed: int | None = None,
    name: str = "dataset",
    *,
    store_path: str | Path,
) -> Dataset:
    """Filter, sample, and version a training dataset in one call.

    Orchestrates the full composition pipeline: query index → load episodes →
    apply sampling strategy → record recipe → return Dataset.

    Args:
        task: Task name(s) to filter by. ``None`` = no filter.
        quality_min: Minimum quality score (inclusive). ``None`` = no lower bound.
        quality_max: Maximum quality score (inclusive). ``None`` = no upper bound.
        embodiment: Embodiment name(s) to filter by. ``None`` = no filter.
        sampling: Sampling strategy — ``'none'``, ``'stratified'``, or
            ``'quality_weighted'``. Defaults to ``'none'``.
        limit: Maximum episodes to return after sampling. ``None`` = no limit.
        seed: Random seed for deterministic sampling.
        name: Human-readable label for this dataset version (stored in the recipe).
        store_path: Root dataset directory (same path used with ``tq.save()``).
            **Keyword-only and required.**

    Returns:
        :class:`~torq.compose.dataset.Dataset` containing the sampled episodes
        and a full provenance recipe.

    Raises:
        TorqStorageError: If ``store_path`` is missing or unreadable.
        ValueError: If ``sampling`` is an unrecognised strategy (propagated
            from :func:`~torq.compose.sampling.sample`).

    Examples:
        >>> import torq as tq
        >>> ds = tq.compose(
        ...     task="pick", quality_min=0.75,
        ...     sampling="stratified", limit=50,
        ...     name="pick_v1", store_path="/data/robot",
        ... )
        >>> len(ds)
        50
    """
    root = Path(store_path)
    index_root = root / "index"

    # ── Step 1: Query index ──────────────────────────────────────────────────
    matched_ids = query_index(
        index_root,
        task=task,
        quality_min=quality_min,
        quality_max=quality_max,
        embodiment=embodiment,
    )

    if not matched_ids:
        active_filters = _describe_active_filters(task, quality_min, quality_max, embodiment)
        logger.warning(
            "tq.compose() returned 0 episodes after filtering. "
            "Active filters: %s. "
            "Check your filter values or store_path contents.",
            active_filters,
        )
        return Dataset(
            episodes=[],
            name=name,
            recipe=_build_recipe(
                task, quality_min, quality_max, embodiment,
                sampling, limit, seed, name,
                source_ids=[], sampled_ids=[],
            ),
        )

    # ── Step 2: Load episodes ────────────────────────────────────────────────
    episodes = [load(ep_id, store_path) for ep_id in matched_ids]
    source_episode_ids = [ep.episode_id for ep in episodes]

    # ── Step 3: Sample ───────────────────────────────────────────────────────
    sampled = sample(episodes, strategy=sampling, limit=limit, seed=seed)
    sampled_episode_ids = [ep.episode_id for ep in sampled]

    # ── Step 4: Warnings ─────────────────────────────────────────────────────
    if len(sampled) == 0 and len(episodes) > 0:
        logger.warning(
            "tq.compose() returned 0 episodes after sampling (strategy=%r). "
            "All %d matched episodes were excluded by the sampling strategy. "
            "Check that episodes have quality scores if using 'quality_weighted'.",
            sampling,
            len(episodes),
        )
    elif 0 < len(sampled) < _LOW_EPISODE_THRESHOLD:
        if quality_min is not None:
            logger.warning(
                "tq.compose() returned only %d episode(s). "
                "Consider lowering quality_min (currently %.2f) to include more data.",
                len(sampled),
                quality_min,
            )
        else:
            logger.warning(
                "tq.compose() returned only %d episode(s). "
                "Consider broadening your filter criteria to include more data.",
                len(sampled),
            )

    # ── Step 5: Build recipe ──────────────────────────────────────────────────
    recipe = _build_recipe(
        task, quality_min, quality_max, embodiment,
        sampling, limit, seed, name,
        source_ids=source_episode_ids,
        sampled_ids=sampled_episode_ids,
    )

    dataset = Dataset(episodes=sampled, name=name, recipe=recipe)

    # ── Step 6: Gravity wells ─────────────────────────────────────────────────
    n = len(sampled)
    if n == 0:
        pass  # no gravity well — warning already emitted above
    elif n < _LOW_EPISODE_THRESHOLD:
        # GW-SDK-05 wins over GW-SDK-02 (more specific when result is small)
        task_str = task if isinstance(task, str) else (", ".join(task) if task else "all tasks")
        emb_str = embodiment if isinstance(embodiment, str) else (
            ", ".join(embodiment) if embodiment else "all embodiments"
        )
        _gravity_well(
            f"Only {n} episode(s) matched. "
            f"Find community datasets for {task_str} / {emb_str} at datatorq.ai",
            "GW-SDK-05",
        )
    else:
        _gravity_well(
            f"Composed dataset with {n} episodes. "
            f"Compare and share datasets at datatorq.ai",
            "GW-SDK-02",
        )

    return dataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_recipe(
    task: str | list[str] | None,
    quality_min: float | None,
    quality_max: float | None,
    embodiment: str | list[str] | None,
    sampling: str,
    limit: int | None,
    seed: int | None,
    name: str,
    *,
    source_ids: list[str],
    sampled_ids: list[str],
) -> dict:
    return {
        "task": task,
        "quality_min": quality_min,
        "quality_max": quality_max,
        "embodiment": embodiment,
        "sampling": sampling,
        "limit": limit,
        "seed": seed,
        "name": name,
        "source_episode_ids": source_ids,
        "sampled_episode_ids": sampled_ids,
    }


def _describe_active_filters(
    task: str | list[str] | None,
    quality_min: float | None,
    quality_max: float | None,
    embodiment: str | list[str] | None,
) -> str:
    parts = []
    if task is not None:
        parts.append(f"task={task!r}")
    if quality_min is not None:
        parts.append(f"quality_min={quality_min}")
    if quality_max is not None:
        parts.append(f"quality_max={quality_max}")
    if embodiment is not None:
        parts.append(f"embodiment={embodiment!r}")
    return ", ".join(parts) if parts else "none"
