"""Experiment tracking for Torq — captures hypothesis, dataset, and training results.

Unlike Snapshots (immutable, content-addressed), Experiments are mutable: metrics
and status are updated over time via ``exp.log()``.
"""

from __future__ import annotations

import datetime
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from torq._gravity_well import _gravity_well
from torq.errors import TorqError
from torq.storage.index import read_experiments, update_manifest_lineage_counts, write_experiments

if TYPE_CHECKING:
    from torq.lineage.snapshot import Snapshot

__all__ = ["Experiment", "experiment"]

logger = logging.getLogger(__name__)


def _detect_git_commit() -> str | None:
    """Return the current HEAD commit hash, or None if detection fails."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug("Git rev-parse returned non-zero: %s", result.stderr.strip())
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.debug("Git commit auto-detection failed: %s", exc)
        return None


def _experiment_to_dict(exp: Experiment) -> dict:
    """Serialize an Experiment to a JSON-compatible dict (excludes _store_path)."""
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "project": exp.project,
        "dataset_snapshot": exp.dataset_snapshot,
        "hypothesis": exp.hypothesis,
        "assumptions": exp.assumptions,
        "code_commit": exp.code_commit,
        "parent_id": exp.parent_id,
        "metrics": exp.metrics,
        "config": exp.config,
        "status": exp.status,
        "created_at": exp.created_at,
        "completed_at": exp.completed_at,
        "tags": exp.tags,
        "metadata": exp.metadata,
    }


@dataclass
class Experiment:
    """Mutable experiment record capturing hypothesis, dataset, and training results.

    Attributes:
        experiment_id: Unique identifier (``exp_<12 hex chars>``).
        name: Human-readable experiment name.
        project: Project namespace for grouping related experiments.
        dataset_snapshot: ``snapshot_id`` of the Dataset used for training.
        hypothesis: Free-text hypothesis being tested.
        assumptions: List of explicit assumptions.
        code_commit: Git commit hash at experiment creation time, or ``None``.
        parent_id: ``experiment_id`` of the previous experiment in this project.
        metrics: Dict of logged metrics (updated via ``log()``).
        config: Dict of hyperparameters / training config.
        status: One of ``"running"``, ``"completed"``, ``"failed"``.
        created_at: ISO 8601 UTC timestamp of creation.
        completed_at: ISO 8601 UTC timestamp of completion, or ``None``.
        tags: List of string tags.
        metadata: Arbitrary caller-supplied metadata.
    """

    experiment_id: str
    name: str
    project: str
    dataset_snapshot: str
    hypothesis: str | None
    assumptions: list[str]
    code_commit: str | None
    parent_id: str | None
    metrics: dict
    config: dict
    status: str
    created_at: str
    completed_at: str | None
    tags: list[str]
    metadata: dict
    _store_path: Path = field(repr=False, compare=False, default=Path("./torq_data"))

    def __repr__(self) -> str:
        return (
            f"Experiment('{self.name}', project='{self.project}', "
            f"status='{self.status}', snap={self.dataset_snapshot[:12]})"
        )

    def log(
        self,
        *,
        metrics: dict | None = None,
        config: dict | None = None,
        status: str | None = None,
    ) -> None:
        """Update and persist experiment metrics, config, and/or status.

        Merges new values into existing dicts — does not replace them.

        Args:
            metrics: New metric key/value pairs to merge in.
            config: New config key/value pairs to merge in.
            status: New status string.  If ``"completed"``, sets ``completed_at``.
        """
        if metrics:
            self.metrics.update(metrics)
        if config:
            self.config.update(config)
        if status is not None:
            _VALID_STATUSES = {"running", "completed", "failed"}
            if status not in _VALID_STATUSES:
                raise TorqError(
                    f"Invalid experiment status '{status}'. "
                    f"Must be one of: {', '.join(sorted(_VALID_STATUSES))}"
                )
            self.status = status
            if status == "completed":
                self.completed_at = datetime.datetime.now(datetime.UTC).isoformat()

        index_root = self._store_path / "index"
        existing = read_experiments(index_root)
        existing[self.experiment_id] = _experiment_to_dict(self)
        write_experiments(existing, index_root)
        if metrics:
            _gravity_well(
                message=f"Experiment '{self.name}' logged."
                " Share your research journey with your lab",
                feature="GW-SDK-08",
            )


def experiment(
    name: str,
    *,
    dataset: Snapshot,
    hypothesis: str | None = None,
    assumptions: list[str] | None = None,
    project: str = "default",
    code_commit: str = "auto",
    config: dict | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    store_path: Path | str = Path("./torq_data"),
) -> Experiment:
    """Create and persist a new experiment record.

    The new experiment is auto-linked to the most recent experiment in the same
    project as its ``parent_id``.  If no prior experiments exist in the project,
    ``parent_id`` is ``None`` (root node).

    Args:
        name: Human-readable experiment name.
        dataset: ``Snapshot`` object for the dataset used in this experiment.
        hypothesis: Free-text hypothesis being tested.
        assumptions: List of explicit assumptions.
        project: Project namespace for grouping experiments.
        code_commit: Git commit hash, or ``"auto"`` to detect from HEAD.
        config: Hyperparameters / training config dict.
        tags: List of string tags.
        metadata: Arbitrary caller-supplied metadata.
        store_path: Root directory of the Torq data store.

    Returns:
        ``Experiment`` object with ``status="running"``.
    """
    from torq.lineage.snapshot import Snapshot

    if not isinstance(dataset, Snapshot):
        raise TorqError(
            f"Expected a Snapshot object for 'dataset', got {type(dataset).__name__}. "
            "Create one with tq.snapshot(dataset, name=...) first."
        )

    store_path = Path(store_path)
    index_root = store_path / "index"
    index_root.mkdir(parents=True, exist_ok=True)

    # Git auto-detection
    resolved_commit: str | None
    if code_commit == "auto":
        resolved_commit = _detect_git_commit()
    else:
        resolved_commit = code_commit

    # Project auto-linking — find most recent experiment in this project
    existing = read_experiments(index_root)
    project_experiments = [rec for rec in existing.values() if rec.get("project") == project]
    parent_id: str | None = None
    if project_experiments:
        most_recent = max(project_experiments, key=lambda r: r["created_at"])
        parent_id = most_recent["experiment_id"]

    experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
    created_at = datetime.datetime.now(datetime.UTC).isoformat()

    exp = Experiment(
        experiment_id=experiment_id,
        name=name,
        project=project,
        dataset_snapshot=dataset.snapshot_id,
        hypothesis=hypothesis,
        assumptions=assumptions or [],
        code_commit=resolved_commit,
        parent_id=parent_id,
        metrics={},
        config=config or {},
        status="running",
        created_at=created_at,
        completed_at=None,
        tags=tags or [],
        metadata=metadata or {},
        _store_path=store_path,
    )

    existing[experiment_id] = _experiment_to_dict(exp)
    write_experiments(existing, index_root)
    update_manifest_lineage_counts(index_root)

    return exp
