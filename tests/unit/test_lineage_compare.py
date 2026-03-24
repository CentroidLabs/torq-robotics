"""Tests for Story 7.4: Experiment Comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from torq.compose.dataset import Dataset
from torq.episode import Episode
from torq.errors import TorqError
from torq.lineage import Experiment, Snapshot, experiment, snapshot
from torq.lineage.compare import ExperimentDiff, compare

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_episode(episode_id: str) -> Episode:
    return Episode(
        episode_id=episode_id,
        observations={},
        actions={},
        timestamps=[],
        source_path="/tmp/fake.mcap",
    )


def _make_dataset(episode_ids: list[str]) -> Dataset:
    return Dataset(
        episodes=[_make_episode(eid) for eid in episode_ids],
        name="test_ds",
        recipe={"task": "pick"},
    )


def _make_snap(store_path: Path, name: str, episode_ids: list[str]) -> Snapshot:
    return snapshot(_make_dataset(episode_ids), name=name, project="test", store_path=store_path)


def _make_exp(
    store_path: Path,
    name: str,
    snap: Snapshot,
    *,
    hypothesis: str | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
) -> Experiment:
    exp = experiment(
        name,
        dataset=snap,
        hypothesis=hypothesis,
        project="test",
        code_commit=None,
        config=config or {},
        store_path=store_path,
    )
    if metrics:
        exp.log(metrics=metrics)
    return exp


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_compare_returns_diff_object(tmp_path: Path) -> None:
    """AC#1 — compare() returns an ExperimentDiff with all required fields."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001", "ep_0002"])
    _make_exp(tmp_path, "act_v1", snap, metrics={"success_rate": 0.85})
    _make_exp(tmp_path, "act_v2", snap, metrics={"success_rate": 0.91})

    diff = compare("act_v1", "act_v2", store_path=tmp_path)

    assert isinstance(diff, ExperimentDiff)
    assert diff.experiment_a == "act_v1"
    assert diff.experiment_b == "act_v2"
    assert diff.dataset_diff.identical is True
    assert isinstance(diff.metric_deltas, dict)
    assert isinstance(diff.config_changes, dict)
    assert isinstance(diff.hypothesis_comparison, tuple)


def test_compare_same_dataset_shows_identical(tmp_path: Path) -> None:
    """AC#2 — same snapshot → dataset_diff.identical == True."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001", "ep_0002"])
    _make_exp(tmp_path, "exp_a", snap)
    _make_exp(tmp_path, "exp_b", snap)

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    assert diff.dataset_diff.identical is True
    assert diff.dataset_diff.episodes_added == ()
    assert diff.dataset_diff.episodes_removed == ()


def test_compare_different_datasets_shows_episodes(tmp_path: Path) -> None:
    """AC#1 — different snapshots → added/removed episode lists populated."""
    snap_a = _make_snap(tmp_path, "snap_a", ["ep_0001", "ep_0002"])
    snap_b = _make_snap(tmp_path, "snap_b", ["ep_0002", "ep_0003"])
    _make_exp(tmp_path, "exp_a", snap_a)
    _make_exp(tmp_path, "exp_b", snap_b)

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    assert diff.dataset_diff.identical is False
    assert "ep_0003" in diff.dataset_diff.episodes_added
    assert "ep_0001" in diff.dataset_diff.episodes_removed


def test_compare_metric_deltas_computed(tmp_path: Path) -> None:
    """AC#1 — metric deltas = b_value - a_value for shared keys."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    _make_exp(tmp_path, "exp_a", snap, metrics={"success_rate": 0.80, "avg_return": 0.60})
    _make_exp(tmp_path, "exp_b", snap, metrics={"success_rate": 0.90, "avg_return": 0.70})

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    assert pytest.approx(diff.metric_deltas["success_rate"]) == 0.10
    assert pytest.approx(diff.metric_deltas["avg_return"]) == 0.10


def test_compare_config_changes_detected(tmp_path: Path) -> None:
    """AC#1 — config changes includes added, removed, and changed keys."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    _make_exp(tmp_path, "exp_a", snap, config={"lr": 0.001, "batch_size": 32, "dropout": 0.1})
    _make_exp(tmp_path, "exp_b", snap, config={"lr": 0.001, "batch_size": 64, "epochs": 100})

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    # batch_size changed 32→64
    assert "batch_size" in diff.config_changes.get("changed", {})
    # dropout removed in b
    assert "dropout" in diff.config_changes.get("removed", [])
    # epochs added in b
    assert "epochs" in diff.config_changes.get("added", {})


def test_compare_hypothesis_comparison(tmp_path: Path) -> None:
    """AC#1 — hypothesis_comparison tuple holds both hypothesis strings."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    _make_exp(tmp_path, "exp_a", snap, hypothesis="baseline grasp policy")
    _make_exp(tmp_path, "exp_b", snap, hypothesis="wrist_cam improves grasp")

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    assert diff.hypothesis_comparison == ("baseline grasp policy", "wrist_cam improves grasp")


def test_compare_experiment_not_found(tmp_path: Path) -> None:
    """AC#1 — TorqError raised with helpful message when experiment name not found."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    _make_exp(tmp_path, "exp_a", snap)

    with pytest.raises(TorqError, match="exp_missing"):
        compare("exp_a", "exp_missing", store_path=tmp_path)


def test_compare_repr_readable(tmp_path: Path) -> None:
    """AC#3 — repr contains experiment names, dataset info, metrics."""
    snap = _make_snap(tmp_path, "snap1", ["ep_0001", "ep_0002"])
    _make_exp(tmp_path, "act_v1", snap, metrics={"success_rate": 0.85})
    _make_exp(tmp_path, "act_v2", snap, metrics={"success_rate": 0.91})

    diff = compare("act_v1", "act_v2", store_path=tmp_path)
    r = repr(diff)

    assert "act_v1" in r
    assert "act_v2" in r
    assert "success_rate" in r


def test_compare_missing_snapshot_raises(tmp_path: Path) -> None:
    """Referenced snapshot deleted from store → TorqError raised."""
    import json

    snap_a = _make_snap(tmp_path, "snap_a", ["ep_0001"])
    snap_b = _make_snap(tmp_path, "snap_b", ["ep_0002"])
    _make_exp(tmp_path, "exp_a", snap_a)
    _make_exp(tmp_path, "exp_b", snap_b)

    # Corrupt: delete one snapshot from the store
    snapshots_path = tmp_path / "index" / "snapshots.json"
    data = json.loads(snapshots_path.read_text())
    first_key = next(iter(data))
    del data[first_key]
    snapshots_path.write_text(json.dumps(data))

    with pytest.raises(TorqError, match="not found in snapshots.json"):
        compare("exp_a", "exp_b", store_path=tmp_path)


def test_compare_ambiguous_name_raises(tmp_path: Path) -> None:
    """Same experiment name in different projects → TorqError without project kwarg."""
    from torq.lineage.experiment import experiment as create_exp

    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    create_exp("shared_name", dataset=snap, project="proj_a", code_commit=None, store_path=tmp_path)
    create_exp("shared_name", dataset=snap, project="proj_b", code_commit=None, store_path=tmp_path)

    with pytest.raises(TorqError, match="ambiguous"):
        compare("shared_name", "shared_name", store_path=tmp_path)


def test_compare_with_project_disambiguates(tmp_path: Path) -> None:
    """project kwarg resolves ambiguous names."""
    from torq.lineage.experiment import experiment as create_exp

    snap = _make_snap(tmp_path, "snap1", ["ep_0001"])
    create_exp("run1", dataset=snap, project="proj_a", code_commit=None, store_path=tmp_path)
    create_exp("run2", dataset=snap, project="proj_a", code_commit=None, store_path=tmp_path)
    create_exp("run1", dataset=snap, project="proj_b", code_commit=None, store_path=tmp_path)
    create_exp("run2", dataset=snap, project="proj_b", code_commit=None, store_path=tmp_path)

    diff = compare("run1", "run2", project="proj_a", store_path=tmp_path)
    assert diff.experiment_a == "run1"
    assert diff.experiment_b == "run2"


def test_dataset_diff_episodes_immutable(tmp_path: Path) -> None:
    """DatasetDiff episode tuples cannot be mutated."""
    snap_a = _make_snap(tmp_path, "snap_a", ["ep_0001", "ep_0002"])
    snap_b = _make_snap(tmp_path, "snap_b", ["ep_0002", "ep_0003"])
    _make_exp(tmp_path, "exp_a", snap_a)
    _make_exp(tmp_path, "exp_b", snap_b)

    diff = compare("exp_a", "exp_b", store_path=tmp_path)

    assert isinstance(diff.dataset_diff.episodes_added, tuple)
    assert isinstance(diff.dataset_diff.episodes_removed, tuple)
