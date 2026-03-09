"""Unit tests for torq.integrations (Story 5.2).

Covers:
    - import torq.integrations raises no exception without wandb/mlflow
    - wandb.init() raises TorqImportError when wandb not installed
    - mlflow.init() raises TorqImportError when mlflow not installed
    - wandb.notify() is silent when wandb not installed
    - wandb.notify() is silent when no active wandb run
    - wandb.notify() calls wandb.run.config.update with correct keys (mock)
    - quality stats: correct mean/std/min/max from scored episodes only
    - unscored episodes excluded from quality stats
    - mlflow.notify() calls mlflow.log_params with correct keys (mock)
    - _notify_integrations() calls both wandb and mlflow notify
    - config.quiet does not suppress logging calls (only suppresses print/gravity-well)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(ep_id: str, quality_overall: float | None = None) -> MagicMock:
    ep = MagicMock()
    ep.episode_id = ep_id
    if quality_overall is not None:
        ep.quality = MagicMock()
        ep.quality.overall = quality_overall
    else:
        ep.quality = None
    return ep


def _make_dataset(n: int = 3, quality_scores: list[float | None] | None = None) -> MagicMock:
    from torq.compose.dataset import Dataset

    if quality_scores is None:
        quality_scores = [None] * n
    eps = [_make_episode(f"ep_{i:03d}", q) for i, q in enumerate(quality_scores)]
    ds = MagicMock(spec=Dataset)
    ds.name = "test_dataset"
    ds.episodes = eps
    ds.recipe = {"task": "pick", "sampling": "none"}
    ds.__len__ = lambda self: len(eps)
    return ds


# ── Import isolation ──────────────────────────────────────────────────────────

class TestImportIsolation:
    def test_import_integrations_no_wandb_no_mlflow(self) -> None:
        """torq.integrations must be importable even when wandb and mlflow are absent."""
        with patch.dict(sys.modules, {"wandb": None, "mlflow": None}):
            # Remove cached integration modules to force re-evaluation
            mods = [k for k in sys.modules if k.startswith("torq.integrations")]
            saved = {k: sys.modules.pop(k) for k in mods}
            try:
                import torq.integrations  # noqa: F401 — must not raise
            finally:
                sys.modules.update(saved)

    def test_import_torq_serve_no_wandb(self) -> None:
        """import torq.serve must not raise when wandb is absent."""
        with patch.dict(sys.modules, {"wandb": None}):
            import torq.serve  # noqa: F401 — must not raise


# ── wandb.init() ──────────────────────────────────────────────────────────────

class TestWandbInit:
    def test_init_raises_import_error_when_wandb_missing(self) -> None:
        from torq.errors import TorqImportError
        from torq.integrations import wandb as torq_wandb

        ds = _make_dataset()
        with patch.dict(sys.modules, {"wandb": None}):
            with pytest.raises(TorqImportError, match="wandb is required"):
                torq_wandb.init(ds)

    def test_init_calls_notify_when_wandb_present(self) -> None:
        from torq.integrations import wandb as torq_wandb

        ds = _make_dataset()
        with patch.object(torq_wandb, "notify") as mock_notify:
            with patch.dict(sys.modules, {"wandb": MagicMock()}):
                torq_wandb.init(ds, {"batch_size": 4})

        mock_notify.assert_called_once_with(ds, {"batch_size": 4})


# ── wandb.notify() ────────────────────────────────────────────────────────────

class TestWandbNotify:
    def test_notify_silent_when_wandb_not_installed(self) -> None:
        from torq.integrations import wandb as torq_wandb

        ds = _make_dataset()
        with patch.dict(sys.modules, {"wandb": None}):
            torq_wandb.notify(ds, {})  # must not raise

    def test_notify_silent_when_no_active_run(self) -> None:
        from torq.integrations import wandb as torq_wandb

        mock_wandb = MagicMock()
        mock_wandb.run = None  # no active run

        ds = _make_dataset()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            torq_wandb.notify(ds, {})  # must not raise

    def test_notify_calls_config_update_with_correct_keys(self) -> None:
        from torq.integrations import wandb as torq_wandb

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "run-abc"

        ds = _make_dataset(3, quality_scores=[0.8, 0.6, 0.9])
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            torq_wandb.notify(ds, {"batch_size": 32})

        mock_wandb.run.config.update.assert_called_once()
        logged = mock_wandb.run.config.update.call_args[0][0]

        assert logged["torq_dataset_name"] == "test_dataset"
        assert logged["torq_episode_count"] == 3
        assert isinstance(logged["torq_recipe"], str)
        assert logged["torq_quality_n_scored"] == 3
        assert logged["torq_quality_min"] == 0.6
        assert logged["torq_quality_max"] == 0.9
        assert abs(logged["torq_quality_mean"] - round((0.8 + 0.6 + 0.9) / 3, 4)) < 1e-6
        assert logged["torq_quality_std"] is not None
        assert logged["torq_batch_size"] == 32  # config dict is propagated

    def test_notify_does_not_call_print(self, capsys) -> None:
        from torq.integrations import wandb as torq_wandb

        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        ds = _make_dataset()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            torq_wandb.notify(ds, {})

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ── Quality stats ─────────────────────────────────────────────────────────────

class TestQualityStats:
    def test_correct_stats_from_scored_episodes(self) -> None:
        from torq.integrations._utils import _quality_stats

        ds = _make_dataset(3, quality_scores=[0.8, 0.6, 1.0])
        stats = _quality_stats(ds)

        assert stats["torq_quality_n_scored"] == 3
        assert abs(stats["torq_quality_mean"] - round((0.8 + 0.6 + 1.0) / 3, 4)) < 1e-6
        assert stats["torq_quality_min"] == 0.6
        assert stats["torq_quality_max"] == 1.0
        assert stats["torq_quality_std"] is not None

    def test_unscored_episodes_excluded(self) -> None:
        from torq.integrations._utils import _quality_stats

        # Mix of scored and unscored
        ds = _make_dataset(4, quality_scores=[0.8, None, 0.6, None])
        stats = _quality_stats(ds)

        assert stats["torq_quality_n_scored"] == 2
        assert abs(stats["torq_quality_mean"] - round((0.8 + 0.6) / 2, 4)) < 1e-6

    def test_no_scored_episodes_returns_none_stats(self) -> None:
        from torq.integrations._utils import _quality_stats

        ds = _make_dataset(3, quality_scores=[None, None, None])
        stats = _quality_stats(ds)

        assert stats["torq_quality_n_scored"] == 0
        assert stats["torq_quality_mean"] is None
        assert stats["torq_quality_std"] is None
        assert stats["torq_quality_min"] is None
        assert stats["torq_quality_max"] is None

    def test_single_scored_episode_std_is_zero(self) -> None:
        from torq.integrations._utils import _quality_stats

        ds = _make_dataset(1, quality_scores=[0.75])
        stats = _quality_stats(ds)

        assert stats["torq_quality_n_scored"] == 1
        assert stats["torq_quality_std"] == 0.0


# ── mlflow.init() ────────────────────────────────────────────────────────────

class TestMlflowInit:
    def test_init_raises_import_error_when_mlflow_missing(self) -> None:
        from torq.errors import TorqImportError
        from torq.integrations import mlflow as torq_mlflow

        ds = _make_dataset()
        with patch.dict(sys.modules, {"mlflow": None}):
            with pytest.raises(TorqImportError, match="mlflow is required"):
                torq_mlflow.init(ds)


# ── mlflow.notify() ──────────────────────────────────────────────────────────

class TestMlflowNotify:
    def test_notify_silent_when_mlflow_not_installed(self) -> None:
        from torq.integrations import mlflow as torq_mlflow

        ds = _make_dataset()
        with patch.dict(sys.modules, {"mlflow": None}):
            torq_mlflow.notify(ds, {})  # must not raise

    def test_notify_silent_when_no_active_run(self) -> None:
        from torq.integrations import mlflow as torq_mlflow

        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        ds = _make_dataset()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            torq_mlflow.notify(ds, {})  # must not raise

    def test_notify_calls_log_params_with_correct_keys(self) -> None:
        from torq.integrations import mlflow as torq_mlflow

        mock_run = MagicMock()
        mock_run.info.run_id = "run-xyz"
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        ds = _make_dataset(2, quality_scores=[0.7, 0.9])
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            torq_mlflow.notify(ds, {"batch_size": 8})

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]

        assert params["torq_dataset_name"] == "test_dataset"
        assert params["torq_episode_count"] == 2
        assert isinstance(params["torq_recipe"], str)
        assert params["torq_quality_n_scored"] == 2
        assert params["torq_quality_min"] == 0.7
        assert params["torq_quality_max"] == 0.9
        assert abs(params["torq_quality_mean"] - round((0.7 + 0.9) / 2, 4)) < 1e-6
        assert params["torq_batch_size"] == 8

    def test_notify_calls_set_tags(self) -> None:
        from torq.integrations import mlflow as torq_mlflow

        mock_run = MagicMock()
        mock_run.info.run_id = "run-xyz"
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        ds = _make_dataset(2, quality_scores=[0.7, 0.9])
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            torq_mlflow.notify(ds, {})

        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]

        assert tags["torq_dataset_name"] == "test_dataset"
        assert tags["torq_episode_count"] == "2"
        assert tags["torq_quality_n_scored"] == "2"

    def test_notify_does_not_call_print(self, capsys) -> None:
        from torq.integrations import mlflow as torq_mlflow

        mock_run = MagicMock()
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = mock_run

        ds = _make_dataset()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            torq_mlflow.notify(ds, {})

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ── _notify_integrations() ────────────────────────────────────────────────────

class TestNotifyIntegrations:
    def test_calls_both_wandb_and_mlflow_notify(self) -> None:
        from torq.integrations import _notify_integrations
        from torq.integrations import wandb as torq_wandb
        from torq.integrations import mlflow as torq_mlflow

        ds = _make_dataset()
        config = {"batch_size": 16}

        with patch.object(torq_wandb, "notify") as mock_w, \
             patch.object(torq_mlflow, "notify") as mock_m:
            _notify_integrations(ds, config)

        mock_w.assert_called_once_with(ds, config)
        mock_m.assert_called_once_with(ds, config)

    def test_wandb_failure_does_not_prevent_mlflow(self) -> None:
        from torq.integrations import _notify_integrations
        from torq.integrations import wandb as torq_wandb
        from torq.integrations import mlflow as torq_mlflow

        ds = _make_dataset()

        with patch.object(torq_wandb, "notify", side_effect=RuntimeError("boom")), \
             patch.object(torq_mlflow, "notify") as mock_m:
            _notify_integrations(ds, {})  # must not raise

        mock_m.assert_called_once()

    def test_mlflow_failure_does_not_raise(self) -> None:
        from torq.integrations import _notify_integrations
        from torq.integrations import wandb as torq_wandb
        from torq.integrations import mlflow as torq_mlflow

        ds = _make_dataset()

        with patch.object(torq_wandb, "notify"), \
             patch.object(torq_mlflow, "notify", side_effect=RuntimeError("mlflow down")):
            _notify_integrations(ds, {})  # must not raise


# ── DataLoader auto-hook (AC #5) ─────────────────────────────────────────────

class TestDataLoaderAutoHook:
    def test_dataloader_calls_notify_integrations(self) -> None:
        """Creating a DataLoader triggers _notify_integrations (→ real wandb/mlflow calls)."""
        import importlib
        import importlib.util
        import numpy as np
        torch_available = importlib.util.find_spec("torch") is not None
        if not torch_available:
            pytest.skip("torch not installed")

        from torq.compose.dataset import Dataset
        from torq.episode import Episode
        from torq.serve import DataLoader
        from torq.integrations import wandb as torq_wandb
        from torq.integrations import mlflow as torq_mlflow

        # Inline dataset builder — avoids cross-test-module import
        def _make_torch_episode(ep_id: str) -> MagicMock:
            ep = MagicMock(spec=Episode)
            ep.episode_id = ep_id
            ep.observations = {"joint_pos": np.zeros((10, 4), dtype=np.float32)}
            ep.observation_keys = ["joint_pos"]
            ep.actions = np.zeros((10, 2), dtype=np.float32)
            return ep

        eps = [_make_torch_episode(f"ep_{i:03d}") for i in range(4)]
        ds = MagicMock(spec=Dataset)
        ds.name = "hook_test"
        ds.episodes = eps
        ds.__len__ = lambda self: len(eps)
        ds.__getitem__ = lambda self, idx: eps[idx]

        with patch.object(torq_wandb, "notify") as mock_w, \
             patch.object(torq_mlflow, "notify") as mock_m:
            DataLoader(ds, batch_size=2)

        mock_w.assert_called_once()
        mock_m.assert_called_once()


# ── config.quiet (AC #4) ─────────────────────────────────────────────────────

class TestConfigQuiet:
    def test_quiet_mode_produces_no_console_output(self, capsys) -> None:
        """With config.quiet=True, integration logging produces no console output."""
        from torq.integrations import wandb as torq_wandb
        from torq.integrations import mlflow as torq_mlflow
        import torq

        original_quiet = torq.config.quiet
        try:
            torq.config.quiet = True

            mock_wandb = MagicMock()
            mock_wandb.run = MagicMock()
            ds = _make_dataset(2, quality_scores=[0.8, 0.9])

            with patch.dict(sys.modules, {"wandb": mock_wandb}):
                torq_wandb.notify(ds, {})

            mock_run = MagicMock()
            mock_mlflow = MagicMock()
            mock_mlflow.active_run.return_value = mock_run
            with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
                torq_mlflow.notify(ds, {})

            captured = capsys.readouterr()
            assert captured.out == "", "Integration produced console output with config.quiet=True"
            assert captured.err == "", "Integration produced stderr with config.quiet=True"
        finally:
            torq.config.quiet = original_quiet
