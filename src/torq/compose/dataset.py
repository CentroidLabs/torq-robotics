"""torq.compose.Dataset — versioned, inspectable episode collection.

A Dataset wraps a list of Episodes with a name and a recipe dict that records
the composition query that produced it (filters, sampling config, seed).  It is
the primary return type of ``tq.compose()`` and the input type expected by
``tq.DataLoader`` in the ``serve/`` layer.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torq.episode import Episode

__all__ = ["Dataset"]


@dataclass
class Dataset:
    """Versioned collection of Episodes produced by a composition query.

    Attributes:
        episodes: Ordered list of Episode objects in this dataset.
        name: Human-readable version label (e.g. ``'pick_place_v3'``).
        recipe: Dict recording the query parameters that created this dataset
            (filters, sampling strategy, seed, etc.).  Stored as-is; no
            validation is applied in R1 — the caller owns the contract.

    Examples:
        >>> ds = Dataset(episodes=eps, name='pick_v1', recipe={'task': 'pick'})
        >>> len(ds)
        31
        >>> repr(ds)
        "Dataset('pick_v1', 31 episodes, quality_avg=0.81)"
    """

    episodes: list[Episode]
    name: str
    recipe: dict = field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of episodes in this dataset."""
        return len(self.episodes)

    def __iter__(self) -> Iterator[Episode]:
        """Iterate over episodes in order."""
        return iter(self.episodes)

    def __getitem__(self, idx: int | slice) -> Episode | list[Episode]:
        """Return the episode at the given index or slice.

        Supports both integer indexing (``dataset[0]``) and slice indexing
        (``dataset[0:5]``), matching standard Python sequence behaviour.
        Integer indexing is required by PyTorch DataLoader (story 5.1).
        """
        return self.episodes[idx]

    def __contains__(self, episode: object) -> bool:
        """Return True if episode is present in this dataset (O(n) linear scan)."""
        return episode in self.episodes

    def __repr__(self) -> str:
        """Return a human-readable summary of this dataset.

        Format: ``Dataset('name', N episodes, quality_avg=X.XX)``

        ``quality_avg`` is computed from episodes that have a non-``None``
        ``quality.overall`` score.  Shows ``N/A`` when no episodes are scored.
        """
        n = len(self.episodes)
        scored = [
            ep.quality.overall
            for ep in self.episodes
            if ep.quality is not None and ep.quality.overall is not None
        ]
        avg = f"{sum(scored) / len(scored):.2f}" if scored else "N/A"
        return f"Dataset('{self.name}', {n} episodes, quality_avg={avg})"
