"""Torq type aliases and protocols for domain concepts."""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = [
    "EpisodeID",
    "FrameProvider",
    "Timestamp",
    "QualityScore",
    "TaskName",
    "EmbodimentName",
]

EpisodeID = str  # format: "ep_0001"
Timestamp = np.int64  # nanoseconds since epoch
QualityScore = float | None  # None when episode too short to score
TaskName = str  # normalised: lowercase, stripped
EmbodimentName = str  # normalised: lowercase, stripped


class FrameProvider(Protocol):
    """Protocol for objects that provide image frame data.

    Satisfied by both ``ImageSequence`` (video-backed) and ``_InMemoryFrames``
    (HDF5-backed). Used as parameter type in storage/serving functions that
    need to read frames without coupling to a concrete class.
    """

    @property
    def frames(self) -> np.ndarray:
        """Return frames array of shape ``[T, H, W, C]``, dtype ``uint8``."""
        ...
