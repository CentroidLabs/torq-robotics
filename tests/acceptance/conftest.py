"""Shared fixtures for acceptance tests."""

from __future__ import annotations

import numpy as np
import pytest

from torq.episode import Episode


def _generate_and_save_scored_episodes(store_path, n: int = 5, seed: int = 42) -> list[str]:
    """Create n synthetic episodes, score them, then save to store_path.

    Quality scores must be attached BEFORE save() so they are written to the
    index (quality.json) and become queryable by tq.compose(quality_min=...).

    Returns list of episode IDs.
    """
    import torq as tq

    rng = np.random.default_rng(seed)
    episode_ids = []
    for _ in range(n):
        T = 50
        ep = Episode(
            episode_id="",  # assigned by save()
            observations={"joint_pos": rng.standard_normal((T, 7)).astype(np.float32)},
            actions=rng.standard_normal((T, 7)).astype(np.float32),
            timestamps=np.arange(T, dtype=np.int64) * 20_000_000,  # 50 Hz
            source_path=store_path,
            metadata={"task": "pick", "embodiment": "panda", "success": True},
        )
        # Score in-memory BEFORE saving so quality is written to the index
        original_quiet = tq.config.quiet
        try:
            tq.config.quiet = True
            tq.quality.score(ep)
            saved = tq.save(ep, path=store_path, quiet=True)
        finally:
            tq.config.quiet = original_quiet
        episode_ids.append(saved.episode_id)
    return episode_ids


@pytest.fixture()
def store_with_scored_episodes(tmp_path):
    """Provide a store_path pre-populated with 5 scored synthetic episodes."""
    _generate_and_save_scored_episodes(tmp_path, n=5)
    return tmp_path
