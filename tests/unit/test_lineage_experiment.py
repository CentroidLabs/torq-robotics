"""Tests for torq.lineage.experiment — experiment tracking."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from torq.lineage import Experiment, Snapshot, experiment, snapshot
from torq.storage.index import read_experiments, write_experiments


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


def _make_snapshot(tmp_path: Path, episode_ids: list[str] | None = None) -> Snapshot:
    from torq.compose.dataset import Dataset

    ids = episode_ids or ["ep_0001", "ep_0002"]
    episodes = [_make_episode(eid) for eid in ids]
    ds = Dataset(episodes=episodes, name="ds", recipe={"task": "pick"})
    return snapshot(ds, name="snap_v1", project="proj", store_path=tmp_path)


# ── Experiment tests ──────────────────────────────────────────────────────────


def test_experiment_creates_and_returns_object(tmp_path: Path) -> None:
    """AC#1 — experiment() returns Experiment with all fields populated."""
    snap = _make_snapshot(tmp_path)
    exp = experiment(
        "act_v2",
        dataset=snap,
        hypothesis="wrist_cam improves grasp",
        project="pick_place",
        store_path=tmp_path,
    )

    assert isinstance(exp, Experiment)
    assert exp.name == "act_v2"
    assert exp.project == "pick_place"
    assert exp.dataset_snapshot == snap.snapshot_id
    assert exp.hypothesis == "wrist_cam improves grasp"
    assert exp.status == "running"
    assert exp.experiment_id.startswith("exp_")
    assert exp.created_at


def test_experiment_persisted_to_json(tmp_path: Path) -> None:
    """AC#1 — experiments.json contains the record after creation."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    experiments_path = tmp_path / "index" / "experiments.json"
    assert experiments_path.exists()
    data = json.loads(experiments_path.read_text())
    assert exp.experiment_id in data


def test_experiment_root_node_no_parent(tmp_path: Path) -> None:
    """AC#5 — first experiment in a project has parent_id = None."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("first", dataset=snap, project="new_project", store_path=tmp_path)

    assert exp.parent_id is None


def test_experiment_auto_links_to_latest_in_project(tmp_path: Path) -> None:
    """AC#6 — third experiment links to the second (most recent) in project."""
    snap = _make_snapshot(tmp_path)

    exp1 = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)
    exp2 = experiment("run2", dataset=snap, project="proj", store_path=tmp_path)
    exp3 = experiment("run3", dataset=snap, project="proj", store_path=tmp_path)

    assert exp1.parent_id is None
    assert exp2.parent_id == exp1.experiment_id
    assert exp3.parent_id == exp2.experiment_id


def test_experiment_git_commit_auto_in_repo(tmp_path: Path) -> None:
    """AC#2 — code_commit captured from git HEAD when in a repo."""
    snap = _make_snapshot(tmp_path)
    fake_hash = "abc1234567890abcdef1234567890abcdef123456"

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = fake_hash + "\n"

    with patch("torq.lineage.experiment.subprocess.run", return_value=mock_result):
        exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    assert exp.code_commit == fake_hash


def test_experiment_git_commit_auto_no_repo(tmp_path: Path) -> None:
    """AC#3 — code_commit = None when not in a git repo; no exception raised."""
    snap = _make_snapshot(tmp_path)

    with patch(
        "torq.lineage.experiment.subprocess.run",
        side_effect=FileNotFoundError("git not found"),
    ):
        exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    assert exp.code_commit is None


def test_experiment_log_merges_metrics(tmp_path: Path) -> None:
    """AC#4 — logging twice with different keys merges all metrics."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    exp.log(metrics={"success_rate": 0.91})
    exp.log(metrics={"avg_return": 0.73})

    assert exp.metrics["success_rate"] == 0.91
    assert exp.metrics["avg_return"] == 0.73


def test_experiment_log_persists_to_json(tmp_path: Path) -> None:
    """AC#4 — experiments.json updated after exp.log()."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    exp.log(metrics={"success_rate": 0.91})

    experiments_path = tmp_path / "index" / "experiments.json"
    data = json.loads(experiments_path.read_text())
    assert data[exp.experiment_id]["metrics"]["success_rate"] == 0.91


def test_experiment_log_status_transition(tmp_path: Path) -> None:
    """AC#4 — logging status='completed' sets completed_at."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    assert exp.completed_at is None
    exp.log(status="completed")

    assert exp.status == "completed"
    assert exp.completed_at is not None


def test_experiment_different_projects_independent(tmp_path: Path) -> None:
    """AC#5 — experiments in different projects don't cross-link."""
    snap = _make_snapshot(tmp_path)

    exp_a = experiment("run1", dataset=snap, project="proj_a", store_path=tmp_path)
    exp_b = experiment("run1", dataset=snap, project="proj_b", store_path=tmp_path)

    assert exp_a.parent_id is None
    assert exp_b.parent_id is None


def test_experiment_repr(tmp_path: Path) -> None:
    """Experiment repr is human-readable."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("act_v2", dataset=snap, project="manipulation", store_path=tmp_path)
    r = repr(exp)

    assert "act_v2" in r
    assert "manipulation" in r
    assert "running" in r
    assert snap.snapshot_id[:12] in r


# ── read_experiments / write_experiments tests ────────────────────────────────


def test_read_experiments_missing_file(tmp_path: Path) -> None:
    """read_experiments returns empty dict when file does not exist."""
    index_root = tmp_path / "index"
    index_root.mkdir()
    assert read_experiments(index_root) == {}


def test_write_then_read_experiments(tmp_path: Path) -> None:
    """write_experiments persists data that read_experiments can reload."""
    index_root = tmp_path / "index"
    index_root.mkdir()

    data = {"exp_abc": {"experiment_id": "exp_abc", "name": "run1"}}
    write_experiments(data, index_root)

    loaded = read_experiments(index_root)
    assert loaded == data


def test_write_experiments_atomic(tmp_path: Path) -> None:
    """write_experiments leaves no temp files after completion."""
    index_root = tmp_path / "index"
    index_root.mkdir()

    write_experiments({"key": "value"}, index_root)

    assert not list(index_root.glob("*.tmp")), "temp files should be cleaned up after atomic write"
    assert (index_root / "experiments.json").exists()


# ── Review fix tests ─────────────────────────────────────────────────────────


def test_experiment_log_status_failed(tmp_path: Path) -> None:
    """log(status='failed') sets status but does NOT set completed_at."""
    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    exp.log(status="failed")

    assert exp.status == "failed"
    assert exp.completed_at is None


def test_experiment_log_invalid_status_raises(tmp_path: Path) -> None:
    """log(status='banana') raises TorqError."""
    from torq.errors import TorqError

    snap = _make_snapshot(tmp_path)
    exp = experiment("run1", dataset=snap, project="proj", store_path=tmp_path)

    with pytest.raises(TorqError, match="Invalid experiment status"):
        exp.log(status="banana")


def test_experiment_invalid_dataset_raises(tmp_path: Path) -> None:
    """Passing a non-Snapshot as dataset raises TorqError."""
    from torq.errors import TorqError

    with pytest.raises(TorqError, match="Expected a Snapshot"):
        experiment("run1", dataset="not_a_snapshot", project="proj", store_path=tmp_path)  # type: ignore[arg-type]


def test_experiment_equality_ignores_store_path(tmp_path: Path) -> None:
    """Two Experiments with same data but different _store_path are equal."""
    from torq.lineage.experiment import Experiment

    kwargs = dict(
        experiment_id="exp_abc",
        name="test",
        project="proj",
        dataset_snapshot="snap_hash",
        hypothesis=None,
        assumptions=[],
        code_commit=None,
        parent_id=None,
        metrics={},
        config={},
        status="running",
        created_at="2026-01-01",
        completed_at=None,
        tags=[],
        metadata={},
    )
    exp1 = Experiment(**kwargs, _store_path=Path("/path/a"))
    exp2 = Experiment(**kwargs, _store_path=Path("/path/b"))

    assert exp1 == exp2
