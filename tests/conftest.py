"""Shared pytest fixtures and configuration for the torq test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from torq.episode import Episode

FIXTURES_DATA = Path(__file__).parent / "fixtures" / "data"


@pytest.fixture
def make_quality_episode() -> Callable[..., Episode]:
    """Factory fixture for creating minimal Episodes used in quality scoring tests.

    Returns a callable with signature::

        make_quality_episode(
            actions=None,    # np.ndarray [T, D]; if None, zeros of shape [n_timesteps, 1]
            *,
            n_timesteps=30,  # used only when actions is None
            metadata=None,   # dict; may include 'success' key
            duration_seconds=None,  # if set, timestamps span this many seconds
        ) -> Episode

    Example::

        def test_foo(make_quality_episode):
            ep = make_quality_episode(np.zeros((20, 6)))
            ep2 = make_quality_episode(n_timesteps=5, metadata={"success": True})
            ep3 = make_quality_episode(duration_seconds=15.0)
    """

    def _make(
        actions: np.ndarray | None = None,
        *,
        n_timesteps: int = 30,
        metadata: dict | None = None,
        duration_seconds: float | None = None,
    ) -> Episode:
        if actions is None:
            actions = np.zeros((n_timesteps, 1))
        T = len(actions)
        if duration_seconds is not None:
            ts_end = int(duration_seconds * 1e9)
            timestamps = np.linspace(0, ts_end, T, dtype=np.int64)
        else:
            timestamps = np.arange(T, dtype=np.int64) * int(1e9 / 30)
        return Episode(
            episode_id="ep_test",
            observations={"joint_pos": np.zeros((T, 7))},
            actions=actions,
            timestamps=timestamps,
            source_path=Path("/test/fixture.mcap"),
            metadata=metadata or {},
        )

    return _make


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root (one level up from tests/)."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_root(project_root: Path) -> Path:
    """Absolute path to src/torq/ package root."""
    return project_root / "src" / "torq"


@pytest.fixture(scope="session")
def fixtures_dir(project_root: Path) -> Path:
    """Absolute path to tests/fixtures/data/ directory."""
    return project_root / "tests" / "fixtures" / "data"


# ── MCAP fixture paths (shared across unit and integration tests) ──


@pytest.fixture(scope="session")
def sample_mcap() -> Path:
    """Path to sample.mcap — 2-topic, 100-message MCAP fixture."""
    p = FIXTURES_DATA / "sample.mcap"
    if not p.exists():
        pytest.skip("sample.mcap not found — run: python tests/fixtures/generate_fixtures.py")
    return p


# ── HDF5 fixture paths (shared across unit and integration tests) ──


@pytest.fixture(scope="session")
def robomimic_hdf5() -> Path:
    """Path to robomimic_simple.hdf5 — 2 demos, no images."""
    p = FIXTURES_DATA / "robomimic_simple.hdf5"
    if not p.exists():
        pytest.skip(
            "robomimic_simple.hdf5 not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p


@pytest.fixture(scope="session")
def robomimic_images_hdf5() -> Path:
    """Path to robomimic_images.hdf5 — 1 demo with agentview_image."""
    p = FIXTURES_DATA / "robomimic_images.hdf5"
    if not p.exists():
        pytest.skip(
            "robomimic_images.hdf5 not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p


@pytest.fixture(scope="session")
def corrupt_hdf5() -> Path:
    """Path to corrupt.hdf5 — truncated file."""
    p = FIXTURES_DATA / "corrupt.hdf5"
    if not p.exists():
        pytest.skip("corrupt.hdf5 not found — run: python tests/fixtures/generate_fixtures.py")
    return p


# ── LeRobot fixture paths (shared across unit and integration tests) ──


@pytest.fixture(scope="session")
def lerobot_dataset() -> Path:
    """Path to lerobot/ — 2-episode LeRobot v3.0 dataset."""
    p = FIXTURES_DATA / "lerobot"
    if not p.exists():
        pytest.skip("lerobot/ fixture not found — run: python tests/fixtures/generate_fixtures.py")
    return p


@pytest.fixture(scope="session")
def lerobot_list_type() -> Path:
    """Path to lerobot_list_type/ — 2-episode LeRobot dataset with list-type columns."""
    p = FIXTURES_DATA / "lerobot_list_type"
    if not p.exists():
        pytest.skip(
            "lerobot_list_type/ not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p


@pytest.fixture(scope="session")
def lerobot_no_info() -> Path:
    """Path to lerobot_no_info/ — LeRobot structure without meta/info.json."""
    p = FIXTURES_DATA / "lerobot_no_info"
    if not p.exists():
        pytest.skip(
            "lerobot_no_info/ fixture not found — run: python tests/fixtures/generate_fixtures.py"
        )
    return p
