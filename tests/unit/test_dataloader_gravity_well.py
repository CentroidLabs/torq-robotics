"""Unit tests for gravity well GW-SDK-03 in TorqDataLoader (Story 5.3).

Tests that the gravity well fires when store_path points to a directory
whose total on-disk size exceeds 50 GB, and is suppressed in all cases
where it shouldn't fire.

Covers:
    - >50 GB → GW-SDK-03 fires; message contains dataset size and "datatorq.ai"
    - <50 GB → no gravity well
    - exactly 50 GB → no gravity well (threshold is strictly >)
    - store_path=None → no gravity well, no error
    - config.quiet=True → no output even when >50 GB
    - gravity well message contains the dataset size in GB (e.g. "51.0 GB")
    - gravity well message contains "datatorq.ai"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from torq._config import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(ep_id: str, T: int = 5, obs_dim: int = 2, action_dim: int = 2) -> MagicMock:
    from torq.episode import Episode

    ep = MagicMock(spec=Episode)
    ep.episode_id = ep_id
    ep.observations = {"joint_pos": np.zeros((T, obs_dim), dtype=np.float32)}
    ep.observation_keys = ["joint_pos"]
    ep.actions = np.zeros((T, action_dim), dtype=np.float32)
    return ep


def _make_dataset(n: int = 2) -> MagicMock:
    from torq.compose.dataset import Dataset

    eps = [_make_episode(f"ep_{i:03d}") for i in range(n)]
    ds = MagicMock(spec=Dataset)
    ds.episodes = eps
    ds.name = "test_dataset"
    ds.__len__ = lambda self: len(eps)
    ds.__getitem__ = lambda self, idx: eps[idx]
    return ds


def _mock_torch():
    """Return a MagicMock that satisfies TorqDataLoader's internal torch usage."""
    torch = MagicMock()
    # torch.utils.data.Dataset base class — must be a real type for subclassing
    torch.utils.data.Dataset = object
    # torch.utils.data.DataLoader returns a sentinel loader
    torch.utils.data.DataLoader.return_value = MagicMock(name="loader")
    # torch.from_numpy returns a tensor-like
    torch.from_numpy.side_effect = lambda arr: MagicMock(name="tensor")
    return torch


def _make_mock_path(total_bytes: int):
    """Return a mock Path object whose rglob yields one file with the given total size."""
    mock_stat = MagicMock()
    mock_stat.st_size = total_bytes

    mock_file = MagicMock(spec=Path)
    mock_file.is_file.return_value = True
    mock_file.stat.return_value = mock_stat

    mock_store = MagicMock(spec=Path)
    mock_store.rglob.return_value = [mock_file]

    # MockPath(any_arg) → mock_store
    MockPath = MagicMock(return_value=mock_store)
    return MockPath


@pytest.fixture(autouse=True)
def reset_quiet(monkeypatch):
    monkeypatch.setattr(config, "quiet", False)


# ── GW-SDK-03 fires when >50 GB ───────────────────────────────────────────────

class TestGravityWellFires:
    def test_51gb_fires_gravity_well(self, capsys) -> None:
        """Dataset >50 GB → GW-SDK-03 fires, message contains 'datatorq.ai'."""
        torch = _mock_torch()
        MockPath = _make_mock_path(51 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        out = capsys.readouterr().out
        assert "datatorq.ai" in out

    def test_51gb_message_contains_size_in_gb(self, capsys) -> None:
        """GW-SDK-03 message must include the dataset size (e.g. '51.0 GB')."""
        torch = _mock_torch()
        MockPath = _make_mock_path(51 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        out = capsys.readouterr().out
        assert "51.0 GB" in out

    def test_51gb_message_contains_full_url(self, capsys) -> None:
        """GW-SDK-03 message must contain the full datatorq.ai URL with arrow prefix."""
        torch = _mock_torch()
        MockPath = _make_mock_path(51 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        out = capsys.readouterr().out
        assert "https://www.datatorq.ai" in out


# ── GW-SDK-03 does NOT fire ───────────────────────────────────────────────────

class TestGravityWellSilent:
    def test_10gb_no_gravity_well(self, capsys) -> None:
        """Dataset <50 GB → no gravity well."""
        torch = _mock_torch()
        MockPath = _make_mock_path(10 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_exactly_50gb_no_gravity_well(self, capsys) -> None:
        """Exactly 50 GB → no gravity well (threshold is strictly >50 GB)."""
        torch = _mock_torch()
        MockPath = _make_mock_path(50 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_store_path_none_no_gravity_well(self, capsys) -> None:
        """store_path=None → no gravity well fired, no error raised."""
        torch = _mock_torch()
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path=None)  # must not raise

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_store_path_omitted_no_gravity_well(self, capsys) -> None:
        """Omitting store_path entirely (default=None) → no gravity well, no error."""
        torch = _mock_torch()
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset())  # store_path not passed at all

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_quiet_true_suppresses_gravity_well(self, capsys, monkeypatch) -> None:
        """config.quiet=True suppresses GW-SDK-03 even when >50 GB."""
        monkeypatch.setattr(config, "quiet", True)
        torch = _mock_torch()
        MockPath = _make_mock_path(51 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        assert capsys.readouterr().out == ""


# ── Ordering and resilience ───────────────────────────────────────────────────

class TestGravityWellOrdering:
    def test_gravity_well_fires_before_notify_integrations(self, capsys) -> None:
        """GW-SDK-03 must fire strictly before _notify_integrations() is called."""
        call_order: list[str] = []

        from torq._gravity_well import _gravity_well as real_gw

        def tracking_gw(msg: str, feature: str) -> None:
            call_order.append("gw")
            real_gw(msg, feature)

        def tracking_notify(dataset, config):
            call_order.append("notify")

        torch = _mock_torch()
        MockPath = _make_mock_path(51 * 1024 ** 3)
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq._gravity_well._gravity_well", side_effect=tracking_gw),
            patch("torq.serve._notify_integrations", side_effect=tracking_notify),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/fake/store")

        assert "gw" in call_order, "Gravity well was not called"
        assert "notify" in call_order, "_notify_integrations was not called"
        assert call_order.index("gw") < call_order.index("notify"), (
            f"Expected GW before notify_integrations, got order: {call_order}"
        )

    def test_oserror_during_stat_does_not_crash(self, capsys) -> None:
        """OSError/PermissionError during size scan must not crash DataLoader init."""
        mock_store = MagicMock(spec=Path)
        mock_store.rglob.side_effect = PermissionError("access denied")
        MockPath = MagicMock(return_value=mock_store)

        torch = _mock_torch()
        with (
            patch("torq.serve.torch_loader._require_torch", return_value=torch),
            patch("torq.serve.torch_loader.Path", MockPath),
            patch("torq.serve._notify_integrations"),
        ):
            from torq.serve import DataLoader
            DataLoader(_make_dataset(), store_path="/restricted/store")  # must not raise

        # no gravity well fires when scan fails
        assert "datatorq.ai" not in capsys.readouterr().out
