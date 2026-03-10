# torq.integrations — ML experiment tracking (W&B, MLflow, TensorBoard)
# Usage: from torq.integrations.wandb import init; init(dataset)
#
# All framework imports (wandb, mlflow) are INSIDE function bodies.
# Importing this module succeeds even when no tracker is installed.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torq.compose.dataset import Dataset

__all__ = ["_notify_integrations"]

logger = logging.getLogger(__name__)


def _notify_integrations(dataset: Dataset, config: dict) -> None:
    """Call all registered ML integration hooks.

    Invoked automatically by ``TorqDataLoader`` at initialisation.  Errors
    from individual integrations are caught, logged as warnings, and never
    re-raised — this function must not disrupt training.

    Args:
        dataset: The Torq Dataset being loaded.
        config: DataLoader configuration dict (batch_size, num_workers, etc.).
    """
    from torq.integrations import mlflow as _mlflow  # noqa: PLC0415
    from torq.integrations import wandb as _wandb  # noqa: PLC0415

    for name, mod in (("wandb", _wandb), ("mlflow", _mlflow)):
        try:
            mod.notify(dataset, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "torq.integrations: unexpected error in %s integration: %s",
                name,
                exc,
            )
