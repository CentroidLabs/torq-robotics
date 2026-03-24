"""Lineage tracing for Torq — full provenance chain from experiment back to source data.

Given an experiment name, ``lineage()`` walks the ``parent_id`` chain and returns a
``LineageGraph`` describing the complete DAG from root ancestor to the queried experiment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from torq.errors import TorqError
from torq.storage.index import read_experiments, read_snapshots

if TYPE_CHECKING:
    from torq.lineage.experiment import Experiment
    from torq.lineage.snapshot import Snapshot

__all__ = ["LineageGraph", "LineageNode", "lineage"]

logger = logging.getLogger(__name__)


def _build_snapshot(record: dict) -> Snapshot:
    """Reconstruct a Snapshot object from a JSON record dict."""
    from torq.lineage.snapshot import Snapshot

    return Snapshot(
        snapshot_id=record["snapshot_id"],
        name=record["name"],
        project=record["project"],
        episode_ids=tuple(record["episode_ids"]),
        content_hash=record["content_hash"],
        recipe=record.get("recipe"),
        created_at=record["created_at"],
        metadata=record.get("metadata", {}),
    )


def _build_experiment(record: dict, store_path: Path) -> Experiment:
    """Reconstruct an Experiment object from a JSON record dict."""
    from torq.lineage.experiment import Experiment

    return Experiment(
        experiment_id=record["experiment_id"],
        name=record["name"],
        project=record.get("project", "default"),
        dataset_snapshot=record["dataset_snapshot"],
        hypothesis=record.get("hypothesis"),
        assumptions=record.get("assumptions", []),
        code_commit=record.get("code_commit"),
        parent_id=record.get("parent_id"),
        metrics=record.get("metrics", {}),
        config=record.get("config", {}),
        status=record.get("status", "running"),
        created_at=record["created_at"],
        completed_at=record.get("completed_at"),
        tags=record.get("tags", []),
        metadata=record.get("metadata", {}),
        _store_path=store_path,
    )


@dataclass
class LineageNode:
    """A single node in a lineage graph.

    Attributes:
        experiment: The Experiment at this node.
        snapshot: The Snapshot associated with this experiment, or ``None`` if not found.
        episode_count: Number of episodes in the snapshot (0 if snapshot is None).
    """

    experiment: Experiment
    snapshot: Snapshot | None
    episode_count: int


@dataclass
class LineageGraph:
    """A DAG of experiments from root ancestor to queried experiment.

    In R1, the graph is always a simple chain (each experiment has at most one parent).
    ``nodes`` is ordered root → leaf (oldest ancestor first, queried experiment last).

    Attributes:
        nodes: Ordered list of LineageNodes from root to leaf.
        root: The oldest ancestor node (no parent).
        leaf: The queried experiment node.
    """

    nodes: list[LineageNode]
    root: LineageNode
    leaf: LineageNode

    def __repr__(self) -> str:
        leaf_name = self.leaf.experiment.name
        lines = [f"Lineage: {leaf_name}", "═" * (len("Lineage: ") + len(leaf_name))]

        for i, node in enumerate(self.nodes):
            exp = node.experiment
            is_root = i == 0
            is_leaf = i == len(self.nodes) - 1
            indent = "    " * i

            # Root prints its own name; non-root names are already printed
            # by the previous node's └─→ line
            if is_root:
                lines.append(f"{indent}{exp.name} (root)")

            snap_name = node.snapshot.name if node.snapshot else "<missing>"
            lines.append(f"{indent}├── snapshot: {snap_name} ({node.episode_count} episodes)")

            metrics = exp.metrics
            if metrics:
                lines.append(f"{indent}├── metrics: {metrics}")

            if not is_leaf:
                lines.append(f"{indent}│")
                next_node = self.nodes[i + 1]
                next_name = next_node.experiment.name
                suffix = " ← (queried)" if i + 1 == len(self.nodes) - 1 else ""
                lines.append(f"{indent}└─→ {next_name}{suffix}")

        return "\n".join(lines)


def lineage(
    experiment_name: str,
    *,
    store_path: Path | str = Path("./torq_data"),
) -> LineageGraph:
    """Return the full provenance chain for the given experiment.

    Walks the ``parent_id`` chain from the named experiment back to the root
    ancestor and returns a ``LineageGraph`` with nodes ordered root → leaf.

    Args:
        experiment_name: Human-readable name of the experiment to trace.
        store_path: Root directory of the Torq data store.

    Returns:
        ``LineageGraph`` with all ancestors from root to the named experiment.

    Raises:
        TorqError: If no experiment with ``experiment_name`` is found.
    """
    store_path = Path(store_path)
    index_root = store_path / "index"

    all_experiments = read_experiments(index_root)
    all_snapshots = read_snapshots(index_root)

    # Find the experiment by name (names are not unique keys; use the most recent if duplicates)
    matching = [r for r in all_experiments.values() if r["name"] == experiment_name]
    if not matching:
        raise TorqError(
            f"Experiment '{experiment_name}' not found. "
            "Check the experiment name or store_path and try again."
        )
    # If multiple with same name, pick the most recently created
    start_record = max(matching, key=lambda r: r["created_at"])

    # Build id-keyed lookup for fast parent traversal
    by_id: dict[str, dict] = {r["experiment_id"]: r for r in all_experiments.values()}

    # Walk parent chain (leaf → root order, will be reversed)
    walk: list[LineageNode] = []
    visited: set[str] = set()
    current: dict | None = start_record

    while current is not None:
        eid = current["experiment_id"]
        if eid in visited:
            logger.warning(
                "Cycle detected in lineage at experiment '%s' (%s)", current["name"], eid
            )
            break
        visited.add(eid)

        snap_id = current.get("dataset_snapshot")
        snap_record = all_snapshots.get(snap_id) if snap_id else None
        snap_obj = _build_snapshot(snap_record) if snap_record else None
        ep_count = len(snap_record["episode_ids"]) if snap_record else 0

        exp_obj = _build_experiment(current, store_path)
        walk.append(LineageNode(experiment=exp_obj, snapshot=snap_obj, episode_count=ep_count))

        parent_id = current.get("parent_id")
        current = by_id.get(parent_id) if parent_id else None

    # Reverse so nodes are root → leaf
    nodes = list(reversed(walk))

    return LineageGraph(nodes=nodes, root=nodes[0], leaf=nodes[-1])
