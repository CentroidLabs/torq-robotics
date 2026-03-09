# torq.serve — ML framework data loaders (NOT in torq.__init__ — explicit import only)
# Usage: from torq.serve import DataLoader
#
# torch is NOT imported here — only when DataLoader.__init__ is called.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torq.serve.torch_loader import TorqDataLoader as DataLoader

if TYPE_CHECKING:
    from torq.compose.dataset import Dataset

__all__ = ["DataLoader"]

logger = logging.getLogger(__name__)


def _notify_integrations(dataset: Dataset, config: dict) -> None:
    """Delegate to torq.integrations to notify all ML experiment trackers.

    Lazy import avoids circular imports — ``torq.integrations`` does not
    import from ``torq.serve``.

    Args:
        dataset: The Torq Dataset being loaded.
        config: DataLoader configuration dict (batch_size, num_workers, etc.).
    """
    from torq.integrations import _notify_integrations as _real_notify  # noqa: PLC0415

    _real_notify(dataset, config)
