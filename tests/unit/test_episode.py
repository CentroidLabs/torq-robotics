"""Unit tests for torq.episode — Episode dataclass.

18 tests covering creation, immutability, repr, derived fields, and import compliance.
"""

from pathlib import Path

import numpy as np
import pytest

from torq.episode import Episode
from torq.errors import EpisodeImmutableFieldError


# ── Shared fixture ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_episode(tmp_path: Path) -> Episode:
    """Minimal Episode with 10 timesteps at 20ms spacing."""
    return Episode(
        episode_id="ep_0001",
        observations={"joint_pos": np.zeros((10, 6), dtype=np.float32)},
        actions=np.zeros((10, 6), dtype=np.float32),
        timestamps=np.arange(10, dtype=np.int64) * 20_000_000,  # 20ms spacing in ns
        source_path=tmp_path / "sample.mcap",
    )


# ── AC #1: Creation and field access ───────────────────────────────────────────


def test_episode_creation_and_field_access(sample_episode: Episode) -> None:
    """All fields are accessible after creation."""
    ep = sample_episode
    assert ep.episode_id == "ep_0001"
    assert "joint_pos" in ep.observations
    assert ep.actions.shape == (10, 6)
    assert ep.timestamps.shape == (10,)
    assert isinstance(ep.source_path, Path)


def test_repr_shows_duration(sample_episode: Episode) -> None:
    """repr() contains the duration string."""
    r = repr(sample_episode)
    assert "duration=" in r


def test_repr_shows_timestep_count(sample_episode: Episode) -> None:
    """repr() contains the timestep count."""
    r = repr(sample_episode)
    assert "steps=10" in r


def test_repr_shows_modality_list(sample_episode: Episode) -> None:
    """repr() contains observation key names."""
    r = repr(sample_episode)
    assert "joint_pos" in r


# ── AC #2: Immutable fields ────────────────────────────────────────────────────


def test_immutable_episode_id(sample_episode: Episode) -> None:
    """Setting episode_id post-init raises EpisodeImmutableFieldError."""
    with pytest.raises(EpisodeImmutableFieldError):
        sample_episode.episode_id = "ep_9999"


def test_immutable_observations(sample_episode: Episode) -> None:
    """Setting observations post-init raises EpisodeImmutableFieldError."""
    with pytest.raises(EpisodeImmutableFieldError):
        sample_episode.observations = {}


def test_immutable_actions(sample_episode: Episode) -> None:
    """Setting actions post-init raises EpisodeImmutableFieldError."""
    with pytest.raises(EpisodeImmutableFieldError):
        sample_episode.actions = np.zeros((5, 6))


def test_immutable_timestamps(sample_episode: Episode) -> None:
    """Setting timestamps post-init raises EpisodeImmutableFieldError."""
    with pytest.raises(EpisodeImmutableFieldError):
        sample_episode.timestamps = np.zeros(5, dtype=np.int64)


def test_immutable_error_message_quality(tmp_path: Path) -> None:
    """Error message names the field and tells user to create a new episode."""
    ep = Episode(
        episode_id="ep_0002",
        observations={"obs": np.zeros((5, 3))},
        actions=np.zeros((5, 3)),
        timestamps=np.arange(5, dtype=np.int64),
        source_path=tmp_path / "x.mcap",
    )
    with pytest.raises(EpisodeImmutableFieldError, match="episode_id") as exc_info:
        ep.episode_id = "changed"
    assert "new Episode" in str(exc_info.value) or "new episode" in str(exc_info.value).lower()


# ── AC #3: Mutable fields ──────────────────────────────────────────────────────


def test_mutable_quality(sample_episode: Episode) -> None:
    """episode.quality = value succeeds without error."""
    sample_episode.quality = object()  # any value


def test_mutable_metadata(sample_episode: Episode) -> None:
    """episode.metadata dict mutation succeeds."""
    sample_episode.metadata["key"] = "val"
    assert sample_episode.metadata["key"] == "val"


def test_mutable_tags(sample_episode: Episode) -> None:
    """episode.tags = list succeeds."""
    sample_episode.tags = ["pick", "success"]
    assert sample_episode.tags == ["pick", "success"]


# ── Derived fields ─────────────────────────────────────────────────────────────


def test_observation_keys_populated(sample_episode: Episode) -> None:
    """observation_keys matches observations.keys()."""
    assert sample_episode.observation_keys == list(sample_episode.observations.keys())


def test_action_keys_populated(sample_episode: Episode) -> None:
    """action_keys is a non-empty list."""
    assert isinstance(sample_episode.action_keys, list)
    assert len(sample_episode.action_keys) > 0


def test_duration_ns_calculated(sample_episode: Episode) -> None:
    """duration_ns == timestamps[-1] - timestamps[0]."""
    expected = int(sample_episode.timestamps[-1] - sample_episode.timestamps[0])
    assert sample_episode.duration_ns == expected


def test_source_path_tracking(tmp_path: Path) -> None:
    """source_path is stored as pathlib.Path."""
    ep = Episode(
        episode_id="ep_0003",
        observations={"obs": np.zeros((5, 3))},
        actions=np.zeros((5, 3)),
        timestamps=np.arange(5, dtype=np.int64),
        source_path=str(tmp_path / "raw_str.mcap"),  # pass as str
    )
    assert isinstance(ep.source_path, Path)


def test_quality_initializes_to_none(sample_episode: Episode) -> None:
    """Freshly created episode has quality == None."""
    assert sample_episode.quality is None


# ── Edge cases ─────────────────────────────────────────────────────────────────


def test_zero_length_timestamps_duration_is_zero(tmp_path: Path) -> None:
    """Episode with a single timestamp (0-length span) has duration_ns == 0."""
    ep = Episode(
        episode_id="ep_0000",
        observations={"obs": np.zeros((1, 3))},
        actions=np.zeros((1, 3)),
        timestamps=np.array([1_000_000_000], dtype=np.int64),  # single timestamp
        source_path=tmp_path / "x.mcap",
    )
    assert ep.duration_ns == 0


# ── Import compliance ──────────────────────────────────────────────────────────


def test_episode_imports_nothing_from_torq() -> None:
    """episode.py must not import from any torq module except torq.errors."""
    src = (Path(__file__).parent.parent.parent / "src" / "torq" / "episode.py").read_text()
    bad_imports = [
        line
        for line in src.splitlines()
        if ("from torq." in line or "import torq." in line) and "torq.errors" not in line
    ]
    assert bad_imports == [], f"Forbidden imports in episode.py: {bad_imports}"
