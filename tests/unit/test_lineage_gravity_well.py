"""Tests for gravity well integration in Experiment.log() — Story 7.6."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import torq
from torq.lineage import Experiment, Snapshot, experiment, snapshot


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_episode(episode_id: str):
    from torq.episode import Episode

    return Episode(
        episode_id=episode_id,
        observations={},
        actions={},
        timestamps=[],
        source_path="/tmp/fake.mcap",
    )


def _make_snapshot_obj(tmp_path: Path) -> Snapshot:
    from torq.compose.dataset import Dataset

    ep = _make_episode("ep_0001")
    ds = Dataset(episodes=[ep], name="ds", recipe=None)
    return snapshot(ds, name="pick_v1", store_path=tmp_path)


def _make_experiment_obj(tmp_path: Path, snap: Snapshot) -> Experiment:
    return experiment(
        "act_v1",
        dataset=snap,
        project="test",
        code_commit=None,
        store_path=tmp_path,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_gravity_well_fires_after_log(tmp_path: Path) -> None:
    """_gravity_well is called after exp.log() succeeds."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    with patch("torq.lineage.experiment._gravity_well") as mock_gw:
        exp.log(metrics={"success_rate": 0.91})

    mock_gw.assert_called_once()


def test_gravity_well_includes_experiment_name(tmp_path: Path) -> None:
    """Gravity well message contains the experiment name."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    with patch("torq.lineage.experiment._gravity_well") as mock_gw:
        exp.log(metrics={"success_rate": 0.85})

    call_kwargs = mock_gw.call_args
    message = call_kwargs.kwargs["message"]
    assert "act_v1" in message


def test_gravity_well_feature_code(tmp_path: Path) -> None:
    """`feature='GW-SDK-08'` is passed to _gravity_well."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    with patch("torq.lineage.experiment._gravity_well") as mock_gw:
        exp.log(metrics={"loss": 0.1})

    call_kwargs = mock_gw.call_args
    assert call_kwargs.kwargs["feature"] == "GW-SDK-08"


def test_gravity_well_quiet_mode(tmp_path: Path, monkeypatch, capsys) -> None:
    """No output is printed when tq.config.quiet = True."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    monkeypatch.setattr(torq.config, "quiet", True)
    exp.log(metrics={"success_rate": 0.9})

    captured = capsys.readouterr()
    assert captured.out == ""


def test_gravity_well_not_fired_on_exception(tmp_path: Path) -> None:
    """_gravity_well is NOT called if write_experiments raises."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    with patch("torq.lineage.experiment.write_experiments", side_effect=OSError("disk full")):
        with patch("torq.lineage.experiment._gravity_well") as mock_gw:
            with pytest.raises(OSError, match="disk full"):
                exp.log(metrics={"success_rate": 0.5})

    mock_gw.assert_not_called()


def test_gravity_well_not_fired_on_config_only_log(tmp_path: Path) -> None:
    """_gravity_well does NOT fire for config-only or status-only log calls."""
    snap = _make_snapshot_obj(tmp_path)
    exp = _make_experiment_obj(tmp_path, snap)

    with patch("torq.lineage.experiment._gravity_well") as mock_gw:
        exp.log(config={"lr": 0.001})

    mock_gw.assert_not_called()

    with patch("torq.lineage.experiment._gravity_well") as mock_gw:
        exp.log(status="completed")

    mock_gw.assert_not_called()
