"""Integration tests for Story 7.3: Lineage Storage.

Verifies:
- snapshots.json and experiments.json are co-located with manifest.json
- manifest.json includes snapshot_count and experiment_count fields
- Rapid sequential writes are all persisted correctly
- JSON files are valid after multiple writes
"""

from __future__ import annotations

import json
from pathlib import Path

from torq.compose.dataset import Dataset
from torq.episode import Episode
from torq.lineage import snapshot
from torq.lineage.experiment import experiment
from torq.storage.index import read_manifest

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_episode(episode_id: str) -> Episode:
    return Episode(
        episode_id=episode_id,
        observations={},
        actions={},
        timestamps=[],
        source_path="/tmp/fake.mcap",
    )


def _make_dataset(episode_ids: list[str]) -> Dataset:
    episodes = [_make_episode(eid) for eid in episode_ids]
    return Dataset(episodes=episodes, name="test_ds", recipe={"task": "pick"})


def _make_snapshot(store_path: Path, name: str = "snap", episode_ids: list[str] | None = None):
    if episode_ids is None:
        episode_ids = ["ep_0001", "ep_0002"]
    ds = _make_dataset(episode_ids)
    return snapshot(ds, name=name, project="test", store_path=store_path)


def _make_experiment(store_path: Path, snap, name: str = "exp"):
    return experiment(
        name,
        dataset=snap,
        project="test",
        code_commit=None,
        store_path=store_path,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_snapshots_json_created_on_first_snapshot(tmp_path: Path) -> None:
    """AC#1 — snapshots.json exists and contains valid JSON after first snapshot."""
    _make_snapshot(tmp_path)

    snapshots_path = tmp_path / "index" / "snapshots.json"
    assert snapshots_path.exists(), "snapshots.json must be created on first snapshot"
    data = json.loads(snapshots_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data) == 1


def test_experiments_json_created_on_first_experiment(tmp_path: Path) -> None:
    """AC#1 — experiments.json exists and contains valid JSON after first experiment."""
    snap = _make_snapshot(tmp_path)
    _make_experiment(tmp_path, snap)

    experiments_path = tmp_path / "index" / "experiments.json"
    assert experiments_path.exists(), "experiments.json must be created on first experiment"
    data = json.loads(experiments_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data) == 1


def test_manifest_includes_snapshot_count(tmp_path: Path) -> None:
    """AC#3 — manifest.json has snapshot_count field equal to number of snapshots created."""
    _make_snapshot(tmp_path, name="snap1", episode_ids=["ep_0001"])
    _make_snapshot(tmp_path, name="snap2", episode_ids=["ep_0002"])

    manifest = read_manifest(tmp_path / "index")
    assert "snapshot_count" in manifest, "manifest.json must include snapshot_count"
    assert manifest["snapshot_count"] == 2


def test_manifest_includes_experiment_count(tmp_path: Path) -> None:
    """AC#3 — manifest.json has experiment_count field equal to number of experiments created."""
    snap1 = _make_snapshot(tmp_path, name="snap1", episode_ids=["ep_0001"])
    _make_experiment(tmp_path, snap1, name="exp1")
    snap2 = _make_snapshot(tmp_path, name="snap2", episode_ids=["ep_0002"])
    _make_experiment(tmp_path, snap2, name="exp2")

    manifest = read_manifest(tmp_path / "index")
    assert "experiment_count" in manifest, "manifest.json must include experiment_count"
    assert manifest["experiment_count"] == 2


def test_rapid_sequential_snapshots_all_persisted(tmp_path: Path) -> None:
    """AC#2 — 5 snapshots created in rapid succession are all persisted."""
    for i in range(5):
        _make_snapshot(tmp_path, name=f"snap{i}", episode_ids=[f"ep_{i:04d}"])

    snapshots_path = tmp_path / "index" / "snapshots.json"
    data = json.loads(snapshots_path.read_text(encoding="utf-8"))
    assert len(data) == 5, f"Expected 5 snapshots, got {len(data)}"

    manifest = read_manifest(tmp_path / "index")
    assert manifest["snapshot_count"] == 5


def test_lineage_files_colocated_with_index(tmp_path: Path) -> None:
    """AC#1 — snapshots.json, experiments.json, and manifest.json are all in the same index/ dir."""
    snap = _make_snapshot(tmp_path)
    _make_experiment(tmp_path, snap)

    index_dir = tmp_path / "index"
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "snapshots.json").exists()
    assert (index_dir / "experiments.json").exists()


def test_valid_json_after_multiple_writes(tmp_path: Path) -> None:
    """AC#2 — Both JSON files parse correctly after a mix of snapshots and experiments."""
    snap1 = _make_snapshot(tmp_path, name="s1", episode_ids=["ep_0001"])
    _make_experiment(tmp_path, snap1, name="e1")
    snap2 = _make_snapshot(tmp_path, name="s2", episode_ids=["ep_0002"])
    _make_experiment(tmp_path, snap2, name="e2")
    _make_snapshot(tmp_path, name="s3", episode_ids=["ep_0003"])

    index_dir = tmp_path / "index"
    snaps = json.loads((index_dir / "snapshots.json").read_text(encoding="utf-8"))
    exps = json.loads((index_dir / "experiments.json").read_text(encoding="utf-8"))

    assert len(snaps) == 3
    assert len(exps) == 2

    manifest = read_manifest(index_dir)
    assert manifest["snapshot_count"] == 3
    assert manifest["experiment_count"] == 2
