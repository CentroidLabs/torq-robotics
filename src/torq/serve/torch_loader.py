"""TorqDataLoader — PyTorch DataLoader for Torq Datasets.

``torch`` is NEVER imported at module level.  All torch usage lives inside
factory functions gated by ``_require_torch()``, which raises
``TorqImportError`` with install instructions when torch is absent.

Usage::

    from torq.serve import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        obs = batch['observations']    # Tensor [B, T, obs_dim]
        act = batch['actions']         # Tensor [B, T, action_dim]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from torq.errors import TorqIngestError, _require_torch

if TYPE_CHECKING:
    import torch
    from torq.compose.dataset import Dataset

__all__ = ["TorqDataLoader", "DataLoader"]


# ── Collation (no torch at definition time — torch imported inside) ───────────

def _torq_collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length episodes to T_max and stack into tensors.

    Args:
        batch: List of dicts from ``__getitem__``, each with keys
               ``'obs'``, ``'actions'``, ``'episode_id'``.

    Returns:
        Dict with ``'observations'`` and ``'actions'`` float tensors of shape
        ``[B, T_max, D]``, and ``'episode_ids'`` as a list of str for provenance.

    Raises:
        TorqIngestError: If feature dimension D differs across episodes.
    """
    torch = _require_torch()

    t_max = max(item["obs"].shape[0] for item in batch)

    obs_dim = batch[0]["obs"].shape[-1] if batch[0]["obs"].ndim > 1 else 0
    act_dim = batch[0]["actions"].shape[-1] if batch[0]["actions"].ndim > 1 else 0

    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []

    for item in batch:
        ep_obs = item["obs"]
        ep_act = item["actions"]
        ep_id = item["episode_id"]

        ep_obs_dim = ep_obs.shape[-1] if ep_obs.ndim > 1 else 0
        if ep_obs_dim != obs_dim:
            raise TorqIngestError(
                f"Observation feature dimension mismatch in episode {ep_id!r}: "
                f"expected {obs_dim}, got {ep_obs_dim}. "
                f"All episodes in a batch must have the same observation dimensionality."
            )

        ep_act_dim = ep_act.shape[-1] if ep_act.ndim > 1 else 0
        if ep_act_dim != act_dim:
            raise TorqIngestError(
                f"Action feature dimension mismatch in episode {ep_id!r}: "
                f"expected {act_dim}, got {ep_act_dim}. "
                f"All episodes in a batch must have the same action dimensionality."
            )

        t = ep_obs.shape[0]
        pad = t_max - t
        if pad > 0:
            ep_obs = np.pad(ep_obs, [(0, pad), (0, 0)])
            ep_act = np.pad(ep_act, [(0, pad), (0, 0)])

        obs_list.append(ep_obs)
        act_list.append(ep_act)

    return {
        "observations": torch.from_numpy(np.stack(obs_list)).float(),
        "actions": torch.from_numpy(np.stack(act_list)).float(),
        "episode_ids": [item["episode_id"] for item in batch],
    }


# ── Factory — creates torch subclasses lazily ─────────────────────────────────

def TorqDataLoader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    store_path: str | Path | None = None,
    **kwargs,
) -> "torch.utils.data.DataLoader":
    """Create a PyTorch DataLoader streaming batches from a Torq Dataset.

    This is a factory function, not a class.  It returns a
    ``torch.utils.data.DataLoader`` instance, which means callers can pass it
    to ``torch.utils.data.distributed.DistributedSampler`` and other
    standard torch tooling without any extra wrapping.

    Args:
        dataset: Torq :class:`~torq.compose.dataset.Dataset` to load from.
        batch_size: Episodes per batch. Defaults to ``32``.
        shuffle: Shuffle episode order each epoch. Defaults to ``False``.
        num_workers: Worker processes for loading. Defaults to ``0``.
        pin_memory: Pin CPU tensors for GPU transfer. Defaults to ``False``.
        store_path: Optional path to the dataset store directory. When provided,
            the total on-disk size is computed and a gravity well fires if it
            exceeds 50 GB (GW-SDK-03). Defaults to ``None`` (no size check).
        **kwargs: Forwarded to ``torch.utils.data.DataLoader``.

    Returns:
        A ``torch.utils.data.DataLoader`` instance.

    Raises:
        TorqImportError: If PyTorch is not installed.

    Examples:
        >>> from torq.serve import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     obs = batch['observations']   # Tensor [B, T, obs_dim]
        ...     act = batch['actions']        # Tensor [B, T, action_dim]
    """
    torch = _require_torch()

    # Define the internal torch Dataset class here so torch is never imported
    # at module level.
    class _TorqTorchDataset(torch.utils.data.Dataset):
        def __init__(self, ds: Dataset) -> None:
            super().__init__()
            self._dataset = ds

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, idx: int) -> dict:
            episode = self._dataset[idx]

            if episode.observation_keys and episode.observations:
                obs_arrays = [episode.observations[k] for k in episode.observation_keys]
                obs = np.concatenate(obs_arrays, axis=-1).astype(np.float32)
            else:
                T = episode.actions.shape[0] if hasattr(episode.actions, "shape") else 0
                obs = np.zeros((T, 0), dtype=np.float32)

            if episode.actions is None:
                T_act = obs.shape[0] if obs.ndim >= 1 else 0
                actions = np.zeros((T_act, 0), dtype=np.float32)
            else:
                actions = np.asarray(episode.actions, dtype=np.float32)

            return {
                "obs": obs,
                "actions": actions,
                "episode_id": episode.episode_id,
            }

    torch_dataset = _TorqTorchDataset(dataset)

    loader = torch.utils.data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_torq_collate_fn,
        **kwargs,
    )

    # GW-SDK-03: fire gravity well if store_path points to >50 GB of data.
    if store_path is not None:
        from torq._gravity_well import _gravity_well  # noqa: PLC0415

        _50GB = 50 * 1024 ** 3
        store = Path(store_path)
        try:
            total_bytes = sum(f.stat().st_size for f in store.rglob("*") if f.is_file())
        except (PermissionError, OSError):
            total_bytes = 0  # skip gravity well if store is unreadable
        if total_bytes > _50GB:
            size_gb = total_bytes / (1024 ** 3)
            _gravity_well(
                f"Dataset is {size_gb:.1f} GB. Stream at scale from datatorq.ai",
                "GW-SDK-03",
            )

    # Lazy import to avoid circular import — torq.serve.__init__ imports us,
    # so we must not import torq.serve at module load time.
    from torq.serve import _notify_integrations  # noqa: PLC0415

    _notify_integrations(dataset, {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
    })

    return loader


# Public alias
DataLoader = TorqDataLoader
