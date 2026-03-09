"""Unit tests for torq.serve.DataLoader (Story 5.1).

Tests that require torch are guarded with @pytest.mark.skipif so the suite
passes in CI environments without torch installed.

Covers:
    - DataLoader NOT importable from top-level torq
    - from torq.serve import DataLoader works without loading torch
    - torch not installed → TorqImportError with correct message
    - batch keys: exactly 'observations' and 'actions'
    - batch shape: [B, T, obs_dim] and [B, T, action_dim]
    - variable-length episodes padded to max T in batch
    - zero-padding: shorter episodes padded with zeros
    - len(loader) == ceil(len(dataset) / batch_size)
    - shuffle=False → deterministic episode order
    - _notify_integrations() called at init
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch_available = importlib.util.find_spec("torch") is not None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episode(ep_id: str, T: int = 10, obs_dim: int = 4, action_dim: int = 2) -> MagicMock:
    """Build a mock Episode with np.ndarray obs and actions."""
    from torq.episode import Episode

    ep = MagicMock(spec=Episode)
    ep.episode_id = ep_id
    ep.observations = {"joint_pos": np.zeros((T, obs_dim), dtype=np.float32)}
    ep.observation_keys = ["joint_pos"]
    ep.actions = np.zeros((T, action_dim), dtype=np.float32)
    return ep


def _make_dataset(n: int, T: int = 10, obs_dim: int = 4, action_dim: int = 2) -> MagicMock:
    from torq.compose.dataset import Dataset

    eps = [_make_episode(f"ep_{i:03d}", T=T, obs_dim=obs_dim, action_dim=action_dim)
           for i in range(n)]
    ds = MagicMock(spec=Dataset)
    ds.episodes = eps
    ds.name = "test_dataset"
    ds.__len__ = lambda self: len(eps)
    ds.__getitem__ = lambda self, idx: eps[idx]
    return ds


# ── Import isolation (no torch required) ─────────────────────────────────────

class TestImportIsolation:
    def test_dataloader_not_in_top_level_torq(self) -> None:
        import torq
        assert not hasattr(torq, "DataLoader"), (
            "DataLoader must NOT be exported from top-level torq — "
            "use 'from torq.serve import DataLoader' instead"
        )

    def test_from_torq_serve_import_dataloader_works(self) -> None:
        """Importing DataLoader from torq.serve must NOT trigger torch import."""
        from torq.serve import DataLoader  # noqa: F401  # must not raise

    def test_serve_init_does_not_import_torch(self) -> None:
        """Importing torq.serve must not bring torch into sys.modules."""
        import sys
        # Remove serve from sys.modules to force re-import
        mods_to_remove = [k for k in sys.modules if k.startswith("torq.serve")]
        for m in mods_to_remove:
            del sys.modules[m]

        # torch should not be imported as a side effect
        torch_before = "torch" in sys.modules
        import torq.serve  # noqa: F401
        torch_after = "torch" in sys.modules

        assert torch_before == torch_after, (
            "Importing torq.serve pulled torch into sys.modules"
        )

    def test_torch_not_installed_raises_import_error(self) -> None:
        """If torch is not available, TorqImportError must be raised on DataLoader init."""
        import sys

        from torq.errors import TorqImportError
        from torq.serve import DataLoader

        ds = _make_dataset(4)
        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises(TorqImportError, match="PyTorch is required"):
                DataLoader(ds, batch_size=2)


# ── Batch content (torch required) ───────────────────────────────────────────

@pytest.mark.skipif(not torch_available, reason="torch not installed")
class TestBatchContent:
    def test_batch_keys(self) -> None:
        from torq.serve import DataLoader

        ds = _make_dataset(4, T=10, obs_dim=4, action_dim=2)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=4)

        batch = next(iter(loader))
        assert set(batch.keys()) == {"observations", "actions", "episode_ids"}

    def test_batch_episode_ids(self) -> None:
        """episode_ids in batch must be a list of str matching input episodes."""
        from torq.serve import DataLoader

        ds = _make_dataset(4, T=10, obs_dim=4, action_dim=2)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        assert isinstance(batch["episode_ids"], list)
        assert all(isinstance(eid, str) for eid in batch["episode_ids"])
        assert batch["episode_ids"] == [f"ep_{i:03d}" for i in range(4)]

    def test_batch_obs_shape(self) -> None:
        import torch
        from torq.serve import DataLoader

        B, T, obs_dim = 4, 10, 4
        ds = _make_dataset(B, T=T, obs_dim=obs_dim)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=B)

        batch = next(iter(loader))
        assert isinstance(batch["observations"], torch.Tensor)
        assert batch["observations"].shape == (B, T, obs_dim)

    def test_batch_actions_shape(self) -> None:
        import torch
        from torq.serve import DataLoader

        B, T, action_dim = 4, 10, 2
        ds = _make_dataset(B, T=T, action_dim=action_dim)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=B)

        batch = next(iter(loader))
        assert isinstance(batch["actions"], torch.Tensor)
        assert batch["actions"].shape == (B, T, action_dim)

    def test_variable_length_padding(self) -> None:
        """Episodes with T=5 and T=10 in same batch → padded to T=10."""
        import torch
        from torq.compose.dataset import Dataset
        from torq.serve import DataLoader

        ep_short = _make_episode("ep_short", T=5, obs_dim=4, action_dim=2)
        ep_long = _make_episode("ep_long", T=10, obs_dim=4, action_dim=2)

        eps = [ep_short, ep_long]
        ds = MagicMock(spec=Dataset)
        ds.episodes = eps
        ds.name = "mixed"
        ds.__len__ = lambda self: 2
        ds.__getitem__ = lambda self, idx: eps[idx]

        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=2)

        batch = next(iter(loader))
        # Both padded to T_max=10
        assert batch["observations"].shape == (2, 10, 4)
        assert batch["actions"].shape == (2, 10, 2)

    def test_zero_padding_at_end(self) -> None:
        """Short episode gets zero-padded at the END (not the beginning)."""
        import torch
        from torq.compose.dataset import Dataset
        from torq.serve import DataLoader

        obs_val = np.ones((5, 2), dtype=np.float32)
        ep_short = MagicMock()
        ep_short.episode_id = "ep_short"
        ep_short.observations = {"joint_pos": obs_val}
        ep_short.observation_keys = ["joint_pos"]
        ep_short.actions = np.ones((5, 2), dtype=np.float32)

        ep_long = _make_episode("ep_long", T=10, obs_dim=2, action_dim=2)

        eps = [ep_short, ep_long]
        ds = MagicMock(spec=Dataset)
        ds.episodes = eps
        ds.name = "pad_test"
        ds.__len__ = lambda self: 2
        ds.__getitem__ = lambda self, idx: eps[idx]

        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=2)

        batch = next(iter(loader))
        obs = batch["observations"]
        # First episode (ep_short) was index 0: rows 5-9 should be zeros
        short_obs = obs[0]  # [10, 2]
        assert torch.all(short_obs[:5] == 1.0), "Non-padded rows should be 1.0"
        assert torch.all(short_obs[5:] == 0.0), "Padded rows should be 0.0"

    def test_len_equals_ceil_n_over_batch_size(self) -> None:
        from torq.serve import DataLoader

        ds = _make_dataset(10)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=3)

        assert len(loader) == math.ceil(10 / 3)

    def test_shuffle_false_deterministic_order(self) -> None:
        """shuffle=False produces same episode order across two iterations."""
        from torq.serve import DataLoader

        ds = _make_dataset(8, T=5, obs_dim=2, action_dim=2)
        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=4, shuffle=False)

        run1 = [b["observations"].numpy().copy() for b in loader]
        run2 = [b["observations"].numpy().copy() for b in loader]
        for a, b in zip(run1, run2):
            np.testing.assert_array_equal(a, b)

    def test_obs_dim_mismatch_raises_ingest_error(self) -> None:
        """Episodes with different obs_dim in the same batch → TorqIngestError naming episode ID."""
        from torq.compose.dataset import Dataset
        from torq.errors import TorqIngestError
        from torq.serve import DataLoader

        ep_a = _make_episode("ep_a", T=5, obs_dim=4, action_dim=2)
        ep_b = _make_episode("ep_b", T=5, obs_dim=8, action_dim=2)  # different obs_dim

        eps = [ep_a, ep_b]
        ds = MagicMock(spec=Dataset)
        ds.episodes = eps
        ds.name = "mismatch_test"
        ds.__len__ = lambda self: 2
        ds.__getitem__ = lambda self, idx: eps[idx]

        with patch("torq.serve._notify_integrations"):
            loader = DataLoader(ds, batch_size=2)

        with pytest.raises(TorqIngestError, match="ep_b"):
            next(iter(loader))

    def test_notify_integrations_called_at_init(self) -> None:
        from torq.serve import DataLoader

        ds = _make_dataset(4)
        with patch("torq.serve._notify_integrations") as mock_notify:
            DataLoader(ds, batch_size=2)

        mock_notify.assert_called_once()
        args = mock_notify.call_args
        assert args[0][0] is ds  # first positional arg is dataset
