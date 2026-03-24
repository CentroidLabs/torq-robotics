"""Tests for torq.lineage.trace — lineage() function and LineageGraph / LineageNode."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from torq.errors import TorqError
from torq.lineage.trace import LineageGraph, LineageNode, lineage


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_snapshot(index_root: Path, *, name: str, episode_ids: list[str]) -> dict:
    """Write a minimal snapshot record to snapshots.json and return the record."""
    snap_id = f"snap_{uuid.uuid4().hex[:12]}"
    record = {
        "snapshot_id": snap_id,
        "name": name,
        "project": "test",
        "episode_ids": episode_ids,
        "content_hash": snap_id,
        "recipe": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }
    snaps_path = index_root / "snapshots.json"
    existing = json.loads(snaps_path.read_text()) if snaps_path.exists() else {}
    existing[snap_id] = record
    snaps_path.write_text(json.dumps(existing))
    return record


def _make_experiment(
    index_root: Path,
    *,
    name: str,
    snapshot_id: str,
    parent_id: str | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
    metrics: dict | None = None,
) -> dict:
    """Write a minimal experiment record to experiments.json and return the record."""
    exp_id = f"exp_{uuid.uuid4().hex[:12]}"
    record = {
        "experiment_id": exp_id,
        "name": name,
        "project": "test",
        "dataset_snapshot": snapshot_id,
        "hypothesis": None,
        "assumptions": [],
        "code_commit": None,
        "parent_id": parent_id,
        "metrics": metrics or {},
        "config": {},
        "status": "completed",
        "created_at": created_at,
        "completed_at": None,
        "tags": [],
        "metadata": {},
    }
    exps_path = index_root / "experiments.json"
    existing = json.loads(exps_path.read_text()) if exps_path.exists() else {}
    existing[exp_id] = record
    exps_path.write_text(json.dumps(existing))
    return record


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """Return a fresh store_path with index directory created."""
    index_root = tmp_path / "index"
    index_root.mkdir(parents=True)
    return tmp_path


# ── tests ─────────────────────────────────────────────────────────────────────


def test_lineage_returns_graph(store: Path) -> None:
    """lineage() returns a LineageGraph instance."""
    index_root = store / "index"
    snap = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])
    _make_experiment(index_root, name="act_v1", snapshot_id=snap["snapshot_id"])

    result = lineage("act_v1", store_path=store)

    assert isinstance(result, LineageGraph)


def test_lineage_single_experiment_single_node(store: Path) -> None:
    """Root experiment with no parent yields a 1-node graph."""
    index_root = store / "index"
    snap = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001", "ep_0002"])
    _make_experiment(index_root, name="act_v1", snapshot_id=snap["snapshot_id"])

    graph = lineage("act_v1", store_path=store)

    assert len(graph.nodes) == 1
    assert graph.root is graph.leaf
    assert graph.root.experiment.name == "act_v1"


def test_lineage_chain_of_three(store: Path) -> None:
    """3 chained experiments → 3 nodes ordered root → leaf."""
    index_root = store / "index"
    snap1 = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])
    snap2 = _make_snapshot(index_root, name="pick_v2", episode_ids=["ep_0001", "ep_0002"])
    snap3 = _make_snapshot(index_root, name="pick_v3", episode_ids=["ep_0001", "ep_0002", "ep_0003"])

    rec1 = _make_experiment(
        index_root, name="act_v1", snapshot_id=snap1["snapshot_id"], created_at="2026-01-01T00:00:00+00:00"
    )
    rec2 = _make_experiment(
        index_root,
        name="act_v2",
        snapshot_id=snap2["snapshot_id"],
        parent_id=rec1["experiment_id"],
        created_at="2026-01-02T00:00:00+00:00",
    )
    _make_experiment(
        index_root,
        name="act_v3",
        snapshot_id=snap3["snapshot_id"],
        parent_id=rec2["experiment_id"],
        created_at="2026-01-03T00:00:00+00:00",
    )

    graph = lineage("act_v3", store_path=store)

    assert len(graph.nodes) == 3
    names = [n.experiment.name for n in graph.nodes]
    assert names == ["act_v1", "act_v2", "act_v3"]
    assert graph.root.experiment.name == "act_v1"
    assert graph.leaf.experiment.name == "act_v3"


def test_lineage_includes_snapshots(store: Path) -> None:
    """Each node in the graph has its associated Snapshot loaded."""
    index_root = store / "index"
    snap1 = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])
    snap2 = _make_snapshot(index_root, name="pick_v2", episode_ids=["ep_0001", "ep_0002"])

    rec1 = _make_experiment(index_root, name="act_v1", snapshot_id=snap1["snapshot_id"])
    _make_experiment(
        index_root,
        name="act_v2",
        snapshot_id=snap2["snapshot_id"],
        parent_id=rec1["experiment_id"],
        created_at="2026-01-02T00:00:00+00:00",
    )

    graph = lineage("act_v2", store_path=store)

    assert graph.nodes[0].snapshot is not None
    assert graph.nodes[0].snapshot.name == "pick_v1"
    assert graph.nodes[0].episode_count == 1

    assert graph.nodes[1].snapshot is not None
    assert graph.nodes[1].snapshot.name == "pick_v2"
    assert graph.nodes[1].episode_count == 2


def test_lineage_repr_readable(store: Path) -> None:
    """repr(graph) contains experiment names and tree structure."""
    index_root = store / "index"
    snap1 = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])
    snap2 = _make_snapshot(index_root, name="pick_v2", episode_ids=["ep_0001", "ep_0002"])

    rec1 = _make_experiment(
        index_root,
        name="act_v1",
        snapshot_id=snap1["snapshot_id"],
        metrics={"success_rate": 0.72},
    )
    _make_experiment(
        index_root,
        name="act_v2",
        snapshot_id=snap2["snapshot_id"],
        parent_id=rec1["experiment_id"],
        created_at="2026-01-02T00:00:00+00:00",
        metrics={"success_rate": 0.85},
    )

    graph = lineage("act_v2", store_path=store)
    text = repr(graph)

    assert "act_v1" in text
    assert "act_v2" in text
    assert "pick_v1" in text
    assert "pick_v2" in text
    assert "Lineage:" in text


def test_lineage_experiment_not_found(store: Path) -> None:
    """lineage() raises TorqError when experiment name does not exist."""
    with pytest.raises(TorqError, match="not found"):
        lineage("nonexistent_experiment", store_path=store)


def test_lineage_handles_missing_snapshot(store: Path) -> None:
    """Experiment that references a non-existent snapshot gets node.snapshot = None."""
    index_root = store / "index"
    # Write experiment without writing its snapshot
    _make_experiment(index_root, name="act_v1", snapshot_id="snap_deadbeef1234")

    graph = lineage("act_v1", store_path=store)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].snapshot is None
    assert graph.nodes[0].episode_count == 0


def test_lineage_cycle_detection(store: Path) -> None:
    """Cycle in parent chain terminates without infinite loop."""
    index_root = store / "index"
    snap = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])

    # Create two experiments that form a cycle: A → B → A
    rec_a = _make_experiment(
        index_root, name="act_v1", snapshot_id=snap["snapshot_id"],
        created_at="2026-01-01T00:00:00+00:00",
    )
    rec_b = _make_experiment(
        index_root, name="act_v2", snapshot_id=snap["snapshot_id"],
        parent_id=rec_a["experiment_id"],
        created_at="2026-01-02T00:00:00+00:00",
    )
    # Manually create the cycle: point A's parent_id at B
    exps_path = index_root / "experiments.json"
    data = json.loads(exps_path.read_text())
    data[rec_a["experiment_id"]]["parent_id"] = rec_b["experiment_id"]
    exps_path.write_text(json.dumps(data))

    graph = lineage("act_v2", store_path=store)

    # Should terminate with 2 nodes (cycle broken), not hang
    assert len(graph.nodes) == 2


def test_lineage_repr_no_duplicate_names(store: Path) -> None:
    """repr does not duplicate non-root node names."""
    index_root = store / "index"
    snap1 = _make_snapshot(index_root, name="pick_v1", episode_ids=["ep_0001"])
    snap2 = _make_snapshot(index_root, name="pick_v2", episode_ids=["ep_0001", "ep_0002"])

    rec1 = _make_experiment(
        index_root, name="act_v1", snapshot_id=snap1["snapshot_id"],
    )
    _make_experiment(
        index_root, name="act_v2", snapshot_id=snap2["snapshot_id"],
        parent_id=rec1["experiment_id"],
        created_at="2026-01-02T00:00:00+00:00",
    )

    graph = lineage("act_v2", store_path=store)
    text = repr(graph)

    # Each experiment name should appear exactly twice:
    # once in the tree and once in the "Lineage: act_v2" header (for leaf only)
    # act_v1: once in "act_v1 (root)", once in no header
    assert text.count("act_v1") == 1
    # act_v2: once in "Lineage: act_v2", once in "└─→ act_v2"
    assert text.count("act_v2") == 2
