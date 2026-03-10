"""Torq W&B integration — logs dataset lineage to an active W&B run.

``wandb`` is NEVER imported at module level.  All wandb usage lives inside
function bodies so that ``import torq.integrations`` succeeds in environments
without wandb installed.

Usage (explicit)::

    from torq.integrations.wandb import init
    init(dataset)

Usage (automatic)::

    # Called automatically by _notify_integrations() when DataLoader is created.
    # No user action required if wandb.init() has already been called.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torq.errors import TorqImportError
from torq.integrations._utils import _quality_stats

if TYPE_CHECKING:
    from torq.compose.dataset import Dataset

__all__ = ["init", "notify"]

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def notify(dataset: Dataset, config: dict) -> None:
    """Log dataset metadata to the active W&B run (silent on missing dep/run).

    Called automatically by ``_notify_integrations()`` when a DataLoader is
    created.  Failures are logged at DEBUG and never propagate — this function
    must not disrupt training.

    Args:
        dataset: Torq Dataset whose metadata to log.
        config: DataLoader configuration dict (batch_size, num_workers, etc.).
    """
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        logger.debug("torq.integrations.wandb: wandb not installed — skipping")
        return

    if wandb.run is None:
        logger.debug("torq.integrations.wandb: no active wandb run — skipping")
        return

    metadata = {
        "torq_dataset_name": dataset.name,
        "torq_episode_count": len(dataset),
        "torq_recipe": str(dataset.recipe),  # serialise for consistent scalar handling
        **_quality_stats(dataset),
        **{f"torq_{k}": v for k, v in config.items()},
    }
    wandb.run.config.update(metadata)
    logger.debug("torq.integrations.wandb: logged dataset metadata to run %s", wandb.run.id)


def init(dataset: Dataset, config: dict | None = None) -> None:
    """Explicitly log dataset metadata to the active W&B run.

    This is the user-facing entry point.  Unlike ``notify()``, it raises
    ``TorqImportError`` when wandb is not installed so the user gets a clear
    actionable message.

    Args:
        dataset: Torq Dataset whose metadata to log.
        config: Optional DataLoader configuration dict.

    Raises:
        TorqImportError: If wandb is not installed.
    """
    try:
        import wandb  # noqa: F401, PLC0415
    except ImportError:
        raise TorqImportError(
            "wandb is required for torq W&B integration. Install it with: pip install wandb"
        ) from None
    notify(dataset, config or {})
