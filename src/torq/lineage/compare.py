"""Experiment comparison for Torq — side-by-side diff of two experiment runs.

``compare()`` accepts two experiment names, loads their records and dataset
snapshots from the store, and returns an ``ExperimentDiff`` with structured
information about dataset differences, metric deltas, config changes, and
hypothesis comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torq.errors import TorqError
from torq.storage.index import read_experiments, read_snapshots

__all__ = ["DatasetDiff", "ExperimentDiff", "compare"]


@dataclass(frozen=True)
class DatasetDiff:
    """Structured diff between two dataset snapshots.

    Attributes:
        identical: True if both experiments used the exact same snapshot.
        episodes_added: Episode IDs present in B but not A.
        episodes_removed: Episode IDs present in A but not B.
        snapshot_a: ``snapshot_id`` of experiment A's dataset.
        snapshot_b: ``snapshot_id`` of experiment B's dataset.
    """

    identical: bool
    episodes_added: tuple[str, ...]
    episodes_removed: tuple[str, ...]
    snapshot_a: str
    snapshot_b: str


@dataclass(frozen=True)
class ExperimentDiff:
    """Structured comparison between two experiments.

    Attributes:
        experiment_a: Name of the first experiment.
        experiment_b: Name of the second experiment.
        dataset_diff: Snapshot-level diff (identical flag, added/removed episodes).
        metric_deltas: Dict mapping metric name → (b_value - a_value).
            ``None`` when the metric exists in only one experiment.
        config_changes: Dict with keys ``"added"``, ``"removed"``, ``"changed"``.
            ``"added"`` and ``"changed"`` map key → value; ``"removed"`` is a list.
        hypothesis_comparison: Tuple ``(hypothesis_a, hypothesis_b)`` — either
            can be ``None`` if the experiment has no hypothesis.
        summary: One-line human-readable summary string.
    """

    experiment_a: str
    experiment_b: str
    dataset_diff: DatasetDiff
    metric_deltas: dict[str, float | None]
    config_changes: dict[str, object]
    hypothesis_comparison: tuple[str | None, str | None]
    summary: str = field(default="")

    def __repr__(self) -> str:  # noqa: D105
        sep = "─" * 45
        lines = [
            f"Experiment Comparison: {self.experiment_a} vs {self.experiment_b}",
            sep,
        ]

        # Dataset section
        dd = self.dataset_diff
        if dd.identical:
            lines.append(f"Dataset:  identical  (snapshot {dd.snapshot_a[:12]})")
        else:
            lines.append("Dataset:  different")
            if dd.episodes_added:
                sample = dd.episodes_added[:5]
                lines.append(f"  + {len(dd.episodes_added)} episodes added: {sample}")
            if dd.episodes_removed:
                lines.append(
                    f"  - {len(dd.episodes_removed)} episodes removed: {dd.episodes_removed[:5]}"
                )

        # Metrics section
        if self.metric_deltas:
            lines.append("Metrics:")
            for key, delta in sorted(self.metric_deltas.items()):
                if delta is None:
                    lines.append(f"  {key}: (only in one experiment)")
                else:
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"  {key}: {sign}{delta:.4g}")
        else:
            lines.append("Metrics:  (none logged)")

        # Config section
        cc = self.config_changes
        added = cc.get("added", {})
        removed = cc.get("removed", [])
        changed = cc.get("changed", {})
        if added or removed or changed:
            lines.append("Config changes:")
            for k, v in sorted((added or {}).items()):
                lines.append(f"  + {k}: {v}")
            for k in sorted(removed or []):
                lines.append(f"  - {k}")
            for k, (old, new) in sorted((changed or {}).items()):
                lines.append(f"  ~ {k}: {old} → {new}")
        else:
            lines.append("Config:   identical")

        # Hypothesis section
        hyp_a, hyp_b = self.hypothesis_comparison
        lines.append("Hypothesis:")
        lines.append(f'  A: "{hyp_a}"')
        lines.append(f'  B: "{hyp_b}"')

        return "\n".join(lines)


def _find_experiment_by_name(name: str, all_experiments: dict, project: str | None = None) -> dict:
    """Find the most recent experiment record matching the given name.

    Args:
        name: Experiment name to search for.
        all_experiments: Dict of all experiment records from experiments.json.
        project: Optional project filter. If ``None`` and multiple projects
            contain experiments with the same name, raises ``TorqError``.

    Returns:
        The matching experiment record dict.

    Raises:
        TorqError: If no experiment with that name exists, or if the name
            is ambiguous across projects and no project filter is given.
    """
    matches = [rec for rec in all_experiments.values() if rec.get("name") == name]
    if project is not None:
        matches = [rec for rec in matches if rec.get("project") == project]
    if not matches:
        msg = f"Experiment '{name}' not found"
        if project:
            msg += f" in project '{project}'"
        raise TorqError(
            f"{msg}. Use tq.experiment() to create experiments first, "
            "or check available names in your store's experiments.json."
        )
    # Warn on ambiguity across projects
    projects = {rec.get("project") for rec in matches}
    if project is None and len(projects) > 1:
        raise TorqError(
            f"Experiment name '{name}' is ambiguous — found in projects: "
            f"{sorted(projects)}. Pass project='...' to disambiguate."
        )
    # Return most recent if multiple matches (same project, different runs)
    return max(matches, key=lambda r: r["created_at"])


def _compute_dataset_diff(
    snap_a_id: str,
    snap_b_id: str,
    all_snapshots: dict,
) -> DatasetDiff:
    """Compute the dataset-level diff between two snapshot IDs."""
    if snap_a_id == snap_b_id:
        return DatasetDiff(
            identical=True,
            episodes_added=(),
            episodes_removed=(),
            snapshot_a=snap_a_id,
            snapshot_b=snap_b_id,
        )

    for label, sid in [("A", snap_a_id), ("B", snap_b_id)]:
        if sid not in all_snapshots:
            raise TorqError(
                f"Snapshot '{sid[:16]}...' referenced by experiment {label} "
                "not found in snapshots.json. The store may be inconsistent."
            )

    eps_a = set(all_snapshots[snap_a_id].get("episode_ids", []))
    eps_b = set(all_snapshots[snap_b_id].get("episode_ids", []))

    return DatasetDiff(
        identical=False,
        episodes_added=tuple(sorted(eps_b - eps_a)),
        episodes_removed=tuple(sorted(eps_a - eps_b)),
        snapshot_a=snap_a_id,
        snapshot_b=snap_b_id,
    )


def _compute_metric_deltas(metrics_a: dict, metrics_b: dict) -> dict[str, float | None]:
    """Compute b - a for each metric key in the union of both dicts."""
    all_keys = set(metrics_a) | set(metrics_b)
    deltas: dict[str, float | None] = {}
    for key in sorted(all_keys):
        va = metrics_a.get(key)
        vb = metrics_b.get(key)
        if va is None or vb is None:
            deltas[key] = None
        else:
            try:
                deltas[key] = float(vb) - float(va)
            except (TypeError, ValueError):
                deltas[key] = None
    return deltas


def _compute_config_changes(config_a: dict, config_b: dict) -> dict[str, object]:
    """Return added, removed, and changed config keys between a and b."""
    keys_a = set(config_a)
    keys_b = set(config_b)

    added = {k: config_b[k] for k in keys_b - keys_a}
    removed = sorted(keys_a - keys_b)
    changed = {k: (config_a[k], config_b[k]) for k in keys_a & keys_b if config_a[k] != config_b[k]}
    return {"added": added, "removed": removed, "changed": changed}


def compare(
    exp_a: str,
    exp_b: str,
    *,
    project: str | None = None,
    store_path: Path | str = Path("./torq_data"),
) -> ExperimentDiff:
    """Compare two experiments side-by-side and return a structured diff.

    Looks up experiments by **name** (most recent if duplicate names exist
    within the same project).  If the name appears in multiple projects
    and ``project`` is not specified, raises ``TorqError``.

    Args:
        exp_a: Name of the first experiment.
        exp_b: Name of the second experiment.
        project: Optional project filter to disambiguate duplicate names.
        store_path: Root directory of the Torq data store.

    Returns:
        ``ExperimentDiff`` with dataset diff, metric deltas, config changes,
        and hypothesis comparison.

    Raises:
        TorqError: If either experiment name is not found, or if a name
            is ambiguous across projects and ``project`` is not specified.
    """
    store_path = Path(store_path)
    index_root = store_path / "index"

    all_experiments = read_experiments(index_root)
    rec_a = _find_experiment_by_name(exp_a, all_experiments, project)
    rec_b = _find_experiment_by_name(exp_b, all_experiments, project)

    all_snapshots = read_snapshots(index_root)

    dataset_diff = _compute_dataset_diff(
        rec_a["dataset_snapshot"],
        rec_b["dataset_snapshot"],
        all_snapshots,
    )
    metric_deltas = _compute_metric_deltas(rec_a.get("metrics", {}), rec_b.get("metrics", {}))
    config_changes = _compute_config_changes(rec_a.get("config", {}), rec_b.get("config", {}))
    hypothesis_comparison = (rec_a.get("hypothesis"), rec_b.get("hypothesis"))

    summary = (
        f"{exp_a} vs {exp_b}: "
        f"{'identical dataset' if dataset_diff.identical else 'different datasets'}, "
        f"{len(metric_deltas)} metrics compared"
    )

    return ExperimentDiff(
        experiment_a=exp_a,
        experiment_b=exp_b,
        dataset_diff=dataset_diff,
        metric_deltas=metric_deltas,
        config_changes=config_changes,
        hypothesis_comparison=hypothesis_comparison,
        summary=summary,
    )
