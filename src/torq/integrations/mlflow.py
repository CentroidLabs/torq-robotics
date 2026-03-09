"""Torq MLflow integration — logs dataset lineage to an active MLflow run.

``mlflow`` is NEVER imported at module level.  All mlflow usage lives inside
function bodies so that ``import torq.integrations`` succeeds in environments
without mlflow installed.

Usage (explicit)::

    from torq.integrations.mlflow import init
    init(dataset)

Usage (automatic)::

    # Called automatically by _notify_integrations() when DataLoader is created.
    # No user action required if an active MLflow run exists.
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
    """Log dataset metadata to the active MLflow run (silent on missing dep/run).

    Called automatically by ``_notify_integrations()`` when a DataLoader is
    created.  Failures are logged at DEBUG and never propagate.

    Args:
        dataset: Torq Dataset whose metadata to log.
        config: DataLoader configuration dict (batch_size, num_workers, etc.).
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        logger.debug("torq.integrations.mlflow: mlflow not installed — skipping")
        return

    if mlflow.active_run() is None:
        logger.debug("torq.integrations.mlflow: no active mlflow run — skipping")
        return

    quality = _quality_stats(dataset)
    params: dict[str, object] = {
        "torq_dataset_name": dataset.name,
        "torq_episode_count": len(dataset),
        "torq_recipe": str(dataset.recipe),  # mlflow.log_params requires scalars
        **quality,
        **{f"torq_{k}": v for k, v in config.items()},
    }
    mlflow.log_params(params)
    mlflow.set_tags({
        "torq_dataset_name": dataset.name,
        "torq_episode_count": str(len(dataset)),
        "torq_quality_n_scored": str(quality["torq_quality_n_scored"]),
    })
    logger.debug(
        "torq.integrations.mlflow: logged dataset metadata to run %s",
        mlflow.active_run().info.run_id,
    )


def init(dataset: Dataset, config: dict | None = None) -> None:
    """Explicitly log dataset metadata to the active MLflow run.

    This is the user-facing entry point.  Unlike ``notify()``, it raises
    ``TorqImportError`` when mlflow is not installed.

    Args:
        dataset: Torq Dataset whose metadata to log.
        config: Optional DataLoader configuration dict.

    Raises:
        TorqImportError: If mlflow is not installed.
    """
    try:
        import mlflow  # noqa: F401, PLC0415
    except ImportError:
        raise TorqImportError(
            "mlflow is required for torq MLflow integration. "
            "Install it with: pip install mlflow"
        ) from None
    notify(dataset, config or {})
