"""Torq CLI — entry point.

Commands:
    tq ingest <source>       — Ingest robot recordings from a file or directory.
    tq list                  — List all indexed episodes in the local dataset.
    tq info <id>             — Display full metadata for a single episode.
    tq export <name> -o DIR  — Export a composed dataset to Parquet + MP4 format.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import typer

app = typer.Typer(
    name="tq",
    help="Torq — Robot Learning Data Infrastructure SDK",
    no_args_is_help=False,
)

_DEFAULT_STORE = Path("./torq_data")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _is_headless() -> bool:
    """Return True when running in a non-interactive (headless/CI) environment."""
    return not sys.stdout.isatty()


def _make_error_json(
    *,
    episodes_ingested: int = 0,
    files_processed: int = 0,
    files_failed: int = 1,
    duration_seconds: float = 0.0,
) -> str:
    """Serialise an error or failure result to a JSON string."""
    return json.dumps(
        {
            "episodes_ingested": episodes_ingested,
            "files_processed": files_processed,
            "files_failed": files_failed,
            "duration_seconds": round(duration_seconds, 3),
        }
    )


def _load_index_shards(store: Path) -> tuple[list, dict, dict]:
    """Read quality.json, by_task.json, by_embodiment.json from the index directory."""
    index_dir = store / "index"
    quality_path = index_dir / "quality.json"
    by_task_path = index_dir / "by_task.json"
    by_embodiment_path = index_dir / "by_embodiment.json"

    quality_list: list = json.loads(quality_path.read_text()) if quality_path.exists() else []
    by_task: dict = json.loads(by_task_path.read_text()) if by_task_path.exists() else {}
    by_embodiment: dict = (
        json.loads(by_embodiment_path.read_text()) if by_embodiment_path.exists() else {}
    )
    return quality_list, by_task, by_embodiment


def _invert_index(shard: dict[str, list[str]]) -> dict[str, str]:
    """Invert a task/embodiment shard: bucket → [ep_ids] becomes ep_id → bucket."""
    result: dict[str, str] = {}
    for bucket, ep_ids in shard.items():
        for ep_id in ep_ids:
            result[ep_id] = bucket
    return result


def _read_duration_s(episodes_dir: Path, episode_id: str) -> float | None:
    """Read episode duration by loading only the timestamp_ns column from Parquet."""
    import pyarrow.parquet as pq

    parquet_path = episodes_dir / f"{episode_id}.parquet"
    if not parquet_path.exists():
        return None
    try:
        table = pq.read_table(str(parquet_path), columns=["timestamp_ns"])
        ts = table.column("timestamp_ns").to_pylist()
        if len(ts) < 2:
            return 0.0
        return (ts[-1] - ts[0]) / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Torq CLI — robot learning data infrastructure."""
    if ctx.invoked_subcommand is None:
        from torq._version import __version__

        typer.echo(f"torq {__version__} — available commands: ingest, list, info, export")
        typer.echo("Use the Python SDK: import torq as tq")
        raise typer.Exit(0)


# ---------------------------------------------------------------------------
# tq ingest
# ---------------------------------------------------------------------------


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Directory or file to ingest"),
    json_output: bool = typer.Option(
        False, "--json", help="Output machine-readable JSON to stdout"
    ),
) -> None:
    """Ingest robot recordings from SOURCE into the local dataset.

    Supports MCAP (ROS 2), HDF5 (robomimic), and LeRobot v3.0 formats.
    When SOURCE is a directory, all supported files are ingested recursively.
    Corrupt or unrecognised files in a directory are skipped; exit code 1
    is returned if any files failed.
    """
    import torq as tq
    from torq.errors import TorqError

    if _is_headless():
        tq.config.quiet = True

    if not source.exists():
        msg = f"Source path does not exist: {source}"
        if json_output:
            typer.echo(_make_error_json())
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    start = time.monotonic()
    _errors: list[dict] = []
    _stats: dict[str, int] = {}

    try:
        episodes = tq.ingest(source, errors=_errors, stats=_stats)
    except TorqError as exc:
        duration = time.monotonic() - start
        if json_output:
            typer.echo(_make_error_json(duration_seconds=duration))
        else:
            typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        duration = time.monotonic() - start
        if json_output:
            typer.echo(_make_error_json(duration_seconds=duration))
        else:
            typer.echo(f"Unexpected error: {exc}", err=True)
        raise typer.Exit(1) from exc

    duration = time.monotonic() - start
    episodes_ingested = len(episodes)
    files_failed = len(_errors)
    files_succeeded = _stats.get("files_succeeded", episodes_ingested)
    files_processed = files_succeeded + files_failed

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "episodes_ingested": episodes_ingested,
                    "files_processed": files_processed,
                    "files_failed": files_failed,
                    "duration_seconds": round(duration, 3),
                }
            )
        )
    else:
        typer.echo(f"Ingested {episodes_ingested} episodes in {duration:.1f}s")
        if _errors:
            for err in _errors:
                typer.echo(f"  FAILED {err['path']}: {err['reason']}", err=True)

    raise typer.Exit(1 if files_failed else 0)


# ---------------------------------------------------------------------------
# tq list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_episodes(
    store: Path = typer.Option(_DEFAULT_STORE, "--store", help="Dataset root directory"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """List all indexed episodes in the local dataset.

    Reads the JSON index shards — does not load full Parquet files.
    Duration is computed from a fast column-only Parquet read.
    """
    import torq as tq

    if _is_headless():
        tq.config.quiet = True

    index_dir = store / "index"
    quality_path = index_dir / "quality.json"

    if not quality_path.exists():
        if json_output:
            typer.echo("[]")
        else:
            typer.echo(f"No episodes found in {store}")
        raise typer.Exit(0)

    quality_list, by_task, by_embodiment = _load_index_shards(store)

    if not quality_list:
        if json_output:
            typer.echo("[]")
        else:
            typer.echo(f"No episodes found in {store}")
        raise typer.Exit(0)

    ep_to_task = _invert_index(by_task)
    ep_to_embodiment = _invert_index(by_embodiment)
    episodes_dir = store / "episodes"

    rows = []
    for score, ep_id in quality_list:
        duration_s = _read_duration_s(episodes_dir, ep_id)
        rows.append(
            {
                "id": ep_id,
                "task": ep_to_task.get(ep_id),
                "embodiment": ep_to_embodiment.get(ep_id),
                "duration_s": duration_s,
                "quality": score,
            }
        )

    if json_output:
        typer.echo(json.dumps(rows))
        raise typer.Exit(0)

    # Human-readable table
    def _trunc(value: str, width: int) -> str:
        """Truncate value to fit column width, appending ellipsis if needed."""
        return value if len(value) <= width else value[: width - 1] + "…"

    typer.echo(f"\nEpisodes in {store} ({len(rows)} total):\n")
    header = f"{'ID':<12}{'TASK':<15}{'EMBODIMENT':<14}{'DURATION':<12}{'QUALITY':<8}"
    typer.echo(header)
    typer.echo("-" * 61)
    for row in rows:
        quality_str = f"{row['quality']:.2f}" if row["quality"] is not None else "N/A"
        duration_str = f"{row['duration_s']:.2f}s" if row["duration_s"] is not None else "N/A"
        line = (
            f"{_trunc(row['id'], 12):<12}"
            f"{_trunc(row['task'] or '', 15):<15}"
            f"{_trunc(row['embodiment'] or '', 14):<14}"
            f"{duration_str:<12}"
            f"{quality_str:<8}"
        )
        typer.echo(line)

    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# tq info
# ---------------------------------------------------------------------------


@app.command()
def info(
    episode_id: str = typer.Argument(..., help="Episode ID to inspect (e.g. ep_0001)"),
    store: Path = typer.Option(_DEFAULT_STORE, "--store", help="Dataset root directory"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Display full metadata for a single episode.

    Loads the complete Episode from the Parquet store including quality scores,
    modalities, and provenance.
    """
    import torq as tq
    from torq.storage import load as storage_load

    if _is_headless():
        tq.config.quiet = True

    parquet_path = store / "episodes" / f"{episode_id}.parquet"
    if not parquet_path.exists():
        msg = (
            f"Episode '{episode_id}' not found in store '{store}'.\n"
            f"Use 'tq list --store {store}' to see available episodes."
        )
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    try:
        episode = storage_load(episode_id, store)
    except Exception as exc:
        typer.echo(f"Failed to load episode '{episode_id}': {exc}", err=True)
        raise typer.Exit(1) from exc

    # Build quality dict
    if episode.quality is not None:
        quality_dict: dict | None = {
            "overall": episode.quality.overall,
            "smoothness": getattr(episode.quality, "smoothness", None),
            "consistency": getattr(episode.quality, "consistency", None),
            "completeness": getattr(episode.quality, "completeness", None),
        }
    else:
        quality_dict = None

    duration_s = episode.duration_ns / 1e9
    timesteps = len(episode.timestamps)
    task = episode.metadata.get("task") if episode.metadata else None
    embodiment = episode.metadata.get("embodiment") if episode.metadata else None

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "id": episode.episode_id,
                    "source_path": str(episode.source_path),
                    "timesteps": timesteps,
                    "duration_s": round(duration_s, 3),
                    "modalities": list(episode.observation_keys),
                    "action_keys": list(episode.action_keys),
                    "task": task,
                    "embodiment": embodiment,
                    "tags": list(episode.tags),
                    "quality": quality_dict,
                }
            )
        )
        raise typer.Exit(0)

    # Human-readable output
    typer.echo(f"\nEpisode: {episode.episode_id}")
    typer.echo(f"  Source:      {episode.source_path}")
    typer.echo(f"  Timesteps:   {timesteps}")
    typer.echo(f"  Duration:    {duration_s:.2f}s")
    typer.echo(f"  Modalities:  {', '.join(episode.observation_keys) or '(none)'}")
    typer.echo(f"  Task:        {task or '(unknown)'}")
    typer.echo(f"  Embodiment:  {embodiment or '(unknown)'}")
    typer.echo(f"  Tags:        {', '.join(episode.tags) if episode.tags else '(none)'}")

    if quality_dict is not None:
        typer.echo("  Quality:")
        typer.echo(f"    Overall:       {quality_dict['overall']:.2f}")
        if quality_dict["smoothness"] is not None:
            typer.echo(f"    Smoothness:    {quality_dict['smoothness']:.2f}")
        if quality_dict["consistency"] is not None:
            typer.echo(f"    Consistency:   {quality_dict['consistency']:.2f}")
        if quality_dict["completeness"] is not None:
            typer.echo(f"    Completeness:  {quality_dict['completeness']:.2f}")
    else:
        typer.echo("  Quality:     (not scored)")

    raise typer.Exit(0)


# ---------------------------------------------------------------------------
# tq export
# ---------------------------------------------------------------------------


@app.command()
def export(
    name: str = typer.Argument(..., help="Dataset name to export"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory path"),
    store: Path = typer.Option(_DEFAULT_STORE, "--store", help="Dataset store root"),
    json_output: bool = typer.Option(False, "--json", help="Output machine-readable JSON"),
) -> None:
    """Export a composed dataset to a portable Parquet + MP4 format.

    The dataset must have been created via tq.compose() with the same store_path.
    Episode files and a recipe.json are written to the OUTPUT directory.
    """
    import torq as tq

    if _is_headless():
        tq.config.quiet = True

    recipe_path = store / "datasets" / name / "recipe.json"
    if not recipe_path.exists():
        msg = (
            f"Dataset '{name}' not found in store '{store}'.\n"
            f"Use 'tq list --store {store}' to see available episodes."
        )
        if json_output:
            typer.echo(json.dumps({"error": msg}))
        else:
            typer.echo(msg, err=True)
        raise typer.Exit(1)

    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))

    # Use sampled_episode_ids (post-sampling list); fall back to source IDs if empty/absent
    ep_ids: list[str] = recipe.get("sampled_episode_ids") or recipe.get("source_episode_ids", [])

    # Create output directories
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(exist_ok=True)

    episodes_copied = 0
    episodes_skipped: list[str] = []
    size_bytes = 0
    src_videos = store / "videos"

    for ep_id in ep_ids:
        # Copy Parquet
        src_parquet = store / "episodes" / f"{ep_id}.parquet"
        dst_parquet = output / "episodes" / f"{ep_id}.parquet"
        if src_parquet.exists():
            shutil.copy2(src_parquet, dst_parquet)
            size_bytes += dst_parquet.stat().st_size
            episodes_copied += 1
        else:
            episodes_skipped.append(ep_id)

        # Copy MP4 files (optional — skip silently if videos dir doesn't exist)
        if src_videos.exists():
            mp4_files = sorted(src_videos.glob(f"{ep_id}_*.mp4"))
            if mp4_files:
                (output / "videos").mkdir(exist_ok=True)
            for mp4 in mp4_files:
                dst_mp4 = output / "videos" / mp4.name
                shutil.copy2(mp4, dst_mp4)
                size_bytes += dst_mp4.stat().st_size

    # Write recipe.json to output
    recipe_out = output / "recipe.json"
    recipe_out.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    size_bytes += recipe_out.stat().st_size

    if episodes_skipped:
        skipped_str = ", ".join(episodes_skipped)
        typer.echo(
            f"WARNING: {len(episodes_skipped)} episode(s) missing from store and were skipped: "
            f"{skipped_str}",
            err=True,
        )

    if json_output:
        result_dict: dict = {
            "episodes_exported": episodes_copied,
            "output_path": str(output.resolve()),
            "size_bytes": size_bytes,
        }
        if episodes_skipped:
            result_dict["episodes_skipped"] = episodes_skipped
        typer.echo(json.dumps(result_dict))
    else:
        typer.echo(
            f"Exported {episodes_copied} episodes to '{output}' ({size_bytes / 1024:.1f} KB)"
        )

    raise typer.Exit(1 if episodes_skipped else 0)
