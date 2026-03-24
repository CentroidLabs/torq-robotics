"""Tests for torq.lineage.snapshot — immutable dataset snapshots."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from torq.compose.dataset import Dataset
from torq.errors import TorqError
from torq.lineage import Snapshot, snapshot
from torq.lineage.snapshot import _compute_content_hash
from torq.storage.index import read_snapshots, write_snapshots


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_episode(episode_id: str):
    from torq.episode import Episode

    return Episode(
        episode_id=episode_id,
        observations={},
        actions={},
        timestamps=[],
        source_path="/tmp/fake.mcap",
    )


def _make_dataset(episode_ids: list[str], recipe: dict | None = None) -> Dataset:
    episodes = [_make_episode(eid) for eid in episode_ids]
    return Dataset(episodes=episodes, name="test_ds", recipe=recipe or {"task": "pick"})


# ── Snapshot tests ────────────────────────────────────────────────────────────


def test_snapshot_returns_snapshot_object(tmp_path: Path) -> None:
    """AC#1, #3 — snapshot() returns a Snapshot with all fields populated."""
    ds = _make_dataset(["ep_0001", "ep_0002"])
    snap = snapshot(ds, name="pick_v1", project="manipulation", store_path=tmp_path)

    assert isinstance(snap, Snapshot)
    assert snap.name == "pick_v1"
    assert snap.project == "manipulation"
    assert set(snap.episode_ids) == {"ep_0001", "ep_0002"}
    assert snap.content_hash
    assert snap.snapshot_id == snap.content_hash
    assert snap.recipe == {"task": "pick"}
    assert snap.created_at  # ISO 8601


def test_snapshot_idempotent_dedup(tmp_path: Path) -> None:
    """AC#2 — second call returns existing record, no duplicate in JSON."""
    ds = _make_dataset(["ep_0001", "ep_0002"])
    snap1 = snapshot(ds, name="pick_v1", project="test", store_path=tmp_path)
    snap2 = snapshot(ds, name="pick_v1", project="test", store_path=tmp_path)

    assert snap1.snapshot_id == snap2.snapshot_id

    snapshots_path = tmp_path / "index" / "snapshots.json"
    data = json.loads(snapshots_path.read_text())
    assert len(data) == 1  # no duplicate


def test_snapshot_persisted_to_json(tmp_path: Path) -> None:
    """AC#4 — snapshots.json exists and contains the record."""
    ds = _make_dataset(["ep_0001"])
    snap = snapshot(ds, name="v1", project="proj", store_path=tmp_path)

    snapshots_path = tmp_path / "index" / "snapshots.json"
    assert snapshots_path.exists()
    data = json.loads(snapshots_path.read_text())
    assert snap.snapshot_id in data


def test_snapshot_atomic_write(tmp_path: Path) -> None:
    """AC#4 — write_snapshots (which wraps _atomic_write_json) is used."""
    ds = _make_dataset(["ep_0001"])
    snapshot(ds, name="v1", project="proj", store_path=tmp_path)

    # Verify the file is valid JSON and not a partial write
    snapshots_path = tmp_path / "index" / "snapshots.json"
    assert snapshots_path.exists()
    data = json.loads(snapshots_path.read_text())
    assert len(data) == 1


def test_snapshot_fields_complete(tmp_path: Path) -> None:
    """AC#3 — all required fields present."""
    ds = _make_dataset(["ep_0001", "ep_0002", "ep_0003"])
    snap = snapshot(ds, name="test", project="proj", store_path=tmp_path)

    assert snap.name == "test"
    assert snap.project == "proj"
    assert len(snap.episode_ids) == 3
    assert len(snap.content_hash) == 64  # SHA-256 hex
    assert snap.recipe
    assert "T" in snap.created_at  # ISO 8601 contains T separator
    assert snap.metadata == {}


def test_snapshot_repr(tmp_path: Path) -> None:
    """Snapshot repr is human-readable."""
    ds = _make_dataset(["ep_0001", "ep_0002"])
    snap = snapshot(ds, name="pick_v1", project="manipulation", store_path=tmp_path)
    r = repr(snap)

    assert "pick_v1" in r
    assert "manipulation" in r
    assert "2" in r  # episode count
    assert snap.content_hash[:12] in r


def test_snapshot_different_datasets_different_hash(tmp_path: Path) -> None:
    """Different episode sets → different content hash."""
    ds1 = _make_dataset(["ep_0001", "ep_0002"])
    ds2 = _make_dataset(["ep_0001", "ep_0003"])

    snap1 = snapshot(ds1, name="v1", project="proj", store_path=tmp_path)
    snap2 = snapshot(ds2, name="v2", project="proj", store_path=tmp_path)

    assert snap1.snapshot_id != snap2.snapshot_id


def test_snapshot_empty_dataset_raises(tmp_path: Path) -> None:
    """Empty dataset raises TorqError."""
    ds = _make_dataset([])
    with pytest.raises(TorqError, match="empty"):
        snapshot(ds, name="v1", project="proj", store_path=tmp_path)


def test_snapshot_order_independence() -> None:
    """Episode ID ordering doesn't affect the hash."""
    ep_ids_a = ["ep_0001", "ep_0002", "ep_0003"]
    ep_ids_b = ["ep_0003", "ep_0001", "ep_0002"]

    hash_a = _compute_content_hash(ep_ids_a, {"task": "pick"})
    hash_b = _compute_content_hash(ep_ids_b, {"task": "pick"})

    assert hash_a == hash_b


def test_snapshot_is_frozen(tmp_path: Path) -> None:
    """Snapshot is immutable — field assignment raises FrozenInstanceError."""
    ds = _make_dataset(["ep_0001"])
    snap = snapshot(ds, name="v1", project="proj", store_path=tmp_path)

    with pytest.raises(Exception):  # FrozenInstanceError (dataclasses)
        snap.name = "mutated"  # type: ignore[misc]


def test_snapshot_recipe_deep_copy(tmp_path: Path) -> None:
    """Mutating the original dataset recipe after snapshot does not affect the snapshot."""
    recipe = {"task": "pick", "nested": {"quality_min": 0.8}}
    ds = _make_dataset(["ep_0001"], recipe=recipe)
    snap = snapshot(ds, name="v1", project="proj", store_path=tmp_path)

    # Mutate the original recipe
    recipe["nested"]["quality_min"] = 9999.0

    # Snapshot recipe must be unaffected
    assert snap.recipe["nested"]["quality_min"] == 0.8  # type: ignore[index]


@pytest.mark.slow
def test_snapshot_performance_10k_episodes(tmp_path: Path) -> None:
    """AC#5 — snapshot() completes in under 1 second for 10k episodes."""
    episode_ids = [f"ep_{i:04d}" for i in range(1, 10001)]
    ds = _make_dataset(episode_ids)

    start = time.perf_counter()
    snapshot(ds, name="big_v1", project="perf_test", store_path=tmp_path)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"snapshot() took {elapsed:.3f}s (limit: 1.0s)"


# ── read_snapshots / write_snapshots tests ────────────────────────────────────


def test_read_snapshots_missing_file(tmp_path: Path) -> None:
    """read_snapshots returns empty dict when file does not exist."""
    index_root = tmp_path / "index"
    index_root.mkdir()
    assert read_snapshots(index_root) == {}


def test_write_then_read_snapshots(tmp_path: Path) -> None:
    """write_snapshots persists data that read_snapshots can reload."""
    index_root = tmp_path / "index"
    index_root.mkdir()

    data = {"abc123": {"snapshot_id": "abc123", "name": "v1"}}
    write_snapshots(data, index_root)

    loaded = read_snapshots(index_root)
    assert loaded == data


def test_write_snapshots_atomic(tmp_path: Path) -> None:
    """write_snapshots leaves no temp files after completion."""
    index_root = tmp_path / "index"
    index_root.mkdir()

    write_snapshots({"key": "value"}, index_root)

    assert not list(index_root.glob("*.tmp")), "temp files should be cleaned up after atomic write"
    assert (index_root / "snapshots.json").exists()
