"""Immutable, content-addressed dataset snapshots for Torq.

A Snapshot captures the exact set of episode IDs and composition recipe used
to produce a Dataset, addressed by the SHA-256 hash of that content.  Two calls
with identical data produce the same ``snapshot_id`` (idempotent).
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from torq.errors import TorqError
from torq.storage.index import read_snapshots, update_manifest_lineage_counts, write_snapshots

if TYPE_CHECKING:
    from torq.compose.dataset import Dataset

__all__ = ["Snapshot", "snapshot"]


def _compute_content_hash(episode_ids: list[str], recipe: dict | None) -> str:
    """Compute SHA-256 content hash for a set of episode IDs and recipe.

    The hash input is a deterministic JSON string: episode IDs are sorted so
    insertion order does not affect the result; ``sort_keys=True`` on the recipe
    ensures dict key ordering is also normalised.

    Args:
        episode_ids: List of episode ID strings (order-independent).
        recipe: Composition recipe dict, or ``None``.

    Returns:
        64-character lowercase hexadecimal SHA-256 digest.
    """
    payload = json.dumps(
        {"episode_ids": sorted(episode_ids), "recipe": recipe},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """Immutable, content-addressed record of a Dataset at a point in time.

    ``snapshot_id`` and ``content_hash`` hold the same value — the SHA-256
    digest of the sorted episode ID list and recipe JSON.  This means the same
    data always produces the same identifier regardless of call order.

    Attributes:
        snapshot_id: Content-addressed identifier (SHA-256 hex, 64 chars).
        name: Human-readable label given by the caller (e.g. ``'pick_v1'``).
        project: Project namespace (e.g. ``'manipulation'``).
        episode_ids: Ordered list of episode ID strings in this snapshot.
        content_hash: Same value as ``snapshot_id``.
        recipe: Composition recipe dict that produced the dataset.
        created_at: ISO 8601 timestamp of snapshot creation (UTC).
        metadata: Arbitrary caller-supplied metadata dict.
    """

    snapshot_id: str
    name: str
    project: str
    episode_ids: tuple[str, ...]
    content_hash: str
    recipe: dict | None
    created_at: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n = len(self.episode_ids)
        return (
            f"Snapshot('{self.name}', project='{self.project}', "
            f"{n} episodes, {self.content_hash[:12]})"
        )


def snapshot(
    dataset: Dataset,
    *,
    name: str,
    project: str = "default",
    store_path: Path | str = Path("./torq_data"),
    metadata: dict | None = None,
) -> Snapshot:
    """Create an immutable, content-addressed snapshot of a Dataset.

    If a snapshot with the same content hash already exists in
    ``snapshots.json``, the existing record is returned unchanged (idempotent).

    Args:
        dataset: Composed Dataset to snapshot.  Must have at least one episode.
        name: Human-readable label for this snapshot (e.g. ``'pick_v1'``).
        project: Project namespace for grouping snapshots.
        store_path: Root directory of the Torq data store.
        metadata: Optional caller-supplied metadata dict.

    Returns:
        ``Snapshot`` object with content-addressed ``snapshot_id``.

    Raises:
        TorqError: If ``dataset`` contains no episodes.
    """
    episode_ids = [ep.episode_id for ep in dataset.episodes]
    if not episode_ids:
        raise TorqError(
            "Cannot snapshot an empty dataset. Compose a dataset with at least one episode first."
        )

    recipe = copy.deepcopy(dataset.recipe) if dataset.recipe else None
    content_hash = _compute_content_hash(episode_ids, recipe)

    store_path = Path(store_path)
    index_root = store_path / "index"
    index_root.mkdir(parents=True, exist_ok=True)

    existing = read_snapshots(index_root)

    # Idempotency: return existing record if hash already present
    if content_hash in existing:
        record = existing[content_hash]
        return Snapshot(
            snapshot_id=record["snapshot_id"],
            name=record["name"],
            project=record["project"],
            episode_ids=tuple(record["episode_ids"]),
            content_hash=record["content_hash"],
            recipe=record["recipe"],
            created_at=record["created_at"],
            metadata=record.get("metadata", {}),
        )

    created_at = datetime.datetime.now(datetime.UTC).isoformat()
    snap = Snapshot(
        snapshot_id=content_hash,
        name=name,
        project=project,
        episode_ids=tuple(episode_ids),
        content_hash=content_hash,
        recipe=recipe,
        created_at=created_at,
        metadata=metadata or {},
    )

    existing[content_hash] = {
        "snapshot_id": snap.snapshot_id,
        "name": snap.name,
        "project": snap.project,
        "episode_ids": list(snap.episode_ids),
        "content_hash": snap.content_hash,
        "recipe": snap.recipe,
        "created_at": snap.created_at,
        "metadata": snap.metadata,
    }
    write_snapshots(existing, index_root)
    update_manifest_lineage_counts(index_root)

    return snap
