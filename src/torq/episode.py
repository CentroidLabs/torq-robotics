"""Canonical Episode representation for robot learning data.

Episode is the central data structure of the Torq SDK. All ingestion formats
(MCAP, HDF5, LeRobot) produce Episodes. All quality scoring, composition,
and serving operates on Episodes.

Episode is NOT designed for subclassing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from torq.errors import EpisodeImmutableFieldError

if TYPE_CHECKING:
    pass  # QualityReport forward reference handled by string annotation

__all__ = ["Episode"]

logger = logging.getLogger(__name__)

_IMMUTABLE_FIELDS: frozenset[str] = frozenset(
    {"episode_id", "observations", "actions", "timestamps"}
)


@dataclass
class Episode:
    """A single robot demonstration episode with aligned observations and actions.

    Fields are split into two categories:

    - **Immutable** (locked after init): episode_id, observations, actions, timestamps
    - **Mutable** (can be set after init): quality, metadata, tags

    Args:
        episode_id: Unique identifier in ``ep_{n:04d}`` format.
        observations: Dict mapping modality name → observation data.
            Values are ``np.ndarray`` for continuous data or ``FrameProvider``
            (e.g. ``ImageSequence``, ``_InMemoryFrames``) for image data.
        actions: Action array of shape [T, action_dim] (np.float32 or np.float64).
        timestamps: Nanosecond timestamps array of shape [T], dtype np.int64.
        source_path: Provenance — path to the original source file.
        metadata: Free-form dict for user tags, success flags, etc. Mutable.
        quality: QualityReport attached by tq.quality.score(). None until scored.
        tags: User labels (e.g. ["pick", "success"]). Mutable.
    """

    episode_id: str
    observations: dict[str, np.ndarray | object]  # np.ndarray or FrameProvider (duck-typed)
    actions: np.ndarray
    timestamps: np.ndarray  # np.int64 nanoseconds, shape [T]
    source_path: Path

    metadata: dict = field(default_factory=dict)
    quality: QualityReport | None = field(default=None)  # noqa: F821
    tags: list[str] = field(default_factory=list)

    # Derived fields — populated in __post_init__
    observation_keys: list[str] = field(init=False)
    action_keys: list[str] = field(init=False)
    duration_ns: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_keys", list(self.observations.keys()))
        object.__setattr__(self, "action_keys", ["actions"])  # standard key; multi-action in R2
        if len(self.timestamps) >= 2:
            object.__setattr__(self, "duration_ns", int(self.timestamps[-1] - self.timestamps[0]))
        else:
            object.__setattr__(self, "duration_ns", 0)
        # Convert source_path to Path if given as str
        if not isinstance(self.source_path, Path):
            object.__setattr__(self, "source_path", Path(self.source_path))

    def __setattr__(self, name: str, value: object) -> None:
        # Guard is for accidental external mutation, not deliberate internal use.
        # __post_init__ and storage layer bypass this guard via object.__setattr__()
        # when initialising derived fields or updating provenance after save.
        if name in _IMMUTABLE_FIELDS and hasattr(self, name):
            raise EpisodeImmutableFieldError(
                f"'{name}' cannot be changed after episode creation. "
                f"Create a new Episode with the updated value instead."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        duration_s = self.duration_ns / 1e9
        n_steps = len(self.timestamps)
        modalities = ", ".join(self.observation_keys) or "(none)"
        return (
            f"Episode(id={self.episode_id!r}, "
            f"steps={n_steps}, "
            f"duration={duration_s:.2f}s, "
            f"modalities=[{modalities}])"
        )
