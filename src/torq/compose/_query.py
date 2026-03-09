"""tq.query() — lazy episode iterator using the sharded JSON index.

Returns an iterator of Episodes matching the given filter criteria.
Filtering is performed by ``compose.filters`` functions applied to the
in-memory index shards — no Parquet files are scanned until an Episode
is actually consumed from the iterator.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from torq._gravity_well import _gravity_well
from torq.compose.filters import apply_embodiment_filter, apply_quality_filter, apply_task_filter
from torq.storage import load

if TYPE_CHECKING:
    from torq.episode import Episode

__all__ = ["query"]

logger = logging.getLogger(__name__)


def query(
    task: str | list[str] | None = None,
    quality_min: float | None = None,
    quality_max: float | None = None,
    embodiment: str | list[str] | None = None,
    *,
    store_path: str | Path,
) -> Iterator[Episode]:
    """Return a lazy iterator of Episodes matching the given filter criteria.

    All filtering is performed on the sharded JSON index via ``compose.filters``
    — no Parquet files are scanned until an Episode is actually consumed.

    Args:
        task: Task name(s) to filter by. A single string matches one task; a list
            performs OR logic (union) across all specified tasks. ``None`` = no filter.
        quality_min: Minimum quality score (inclusive). Episodes without a quality
            score are excluded when this filter is active. ``None`` = no lower bound.
        quality_max: Maximum quality score (inclusive). ``None`` = no upper bound.
        embodiment: Embodiment name(s) to filter by. Same OR logic as ``task``.
            ``None`` = no filter.
        store_path: Root dataset directory (same path used with ``tq.save()``).
            Must be provided — there is no global default for this value.

    Yields:
        :class:`~torq.episode.Episode` objects for each matching episode,
        loaded lazily on demand.

    Examples:
        >>> import torq as tq
        >>> episodes = list(tq.query(task="pick", quality_min=0.8, store_path="/data/robot"))
    """
    root = Path(store_path)
    index_root = root / "index"
    quality_path = index_root / "quality.json"

    if not quality_path.exists():
        logger.warning(
            "tq.query() returned 0 episodes. "
            "Filters: task=%r, quality_min=%r, quality_max=%r, embodiment=%r",
            task,
            quality_min,
            quality_max,
            embodiment,
        )
        return

    quality_list: list[list] = json.loads(quality_path.read_text(encoding="utf-8"))
    universe = [entry[1] for entry in quality_list]

    if not universe:
        logger.warning(
            "tq.query() returned 0 episodes. "
            "Filters: task=%r, quality_min=%r, quality_max=%r, embodiment=%r",
            task,
            quality_min,
            quality_max,
            embodiment,
        )
        return

    # ── Apply filters via compose.filters ──
    current: set[str] = set(universe)

    if task is not None:
        by_task_path = index_root / "by_task.json"
        by_task: dict[str, list[str]] = (
            json.loads(by_task_path.read_text(encoding="utf-8")) if by_task_path.exists() else {}
        )
        current = apply_task_filter(list(current), task, by_task)

    if embodiment is not None:
        by_embodiment_path = index_root / "by_embodiment.json"
        by_embodiment: dict[str, list[str]] = (
            json.loads(by_embodiment_path.read_text(encoding="utf-8"))
            if by_embodiment_path.exists()
            else {}
        )
        current = apply_embodiment_filter(list(current), embodiment, by_embodiment)

    if quality_min is not None or quality_max is not None:
        current = apply_quality_filter(list(current), quality_min, quality_max, quality_list)

    episode_ids = sorted(current)

    if not episode_ids:
        logger.warning(
            "tq.query() returned 0 episodes. "
            "Filters: task=%r, quality_min=%r, quality_max=%r, embodiment=%r",
            task,
            quality_min,
            quality_max,
            embodiment,
        )
        return

    # Fire GW-SDK-05 before yielding when result count is low (0 < n < 5).
    # Fires before the first yield so the prompt appears even if the caller
    # never exhausts the iterator.
    if 0 < len(episode_ids) < 5:
        task_str = task if isinstance(task, str) else (", ".join(task) if task else "all tasks")
        emb_str = (
            embodiment if isinstance(embodiment, str)
            else (", ".join(embodiment) if embodiment else "all embodiments")
        )
        _gravity_well(
            f"Only {len(episode_ids)} episode(s) matched your query for {task_str} / {emb_str}. "
            f"Find community data at datatorq.ai",
            "GW-SDK-05",
        )

    for ep_id in episode_ids:
        yield load(ep_id, store_path)
