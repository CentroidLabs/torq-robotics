"""Unit tests for `tq export` CLI command.

Covers:
    - AC #1: export copies Parquet files and writes recipe.json
    - AC #2: --json output with episodes_exported, output_path, size_bytes
    - AC #3: output directory created automatically
    - AC #4: missing dataset → exit 1, name in output, suggests tq list
    - AC #5: headless → tq.config.quiet = True
    - bonus: MP4 files copied when videos/ dir exists
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from torq.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(
    tmp_path: Path,
    name: str = "pick_v1",
    episode_ids: list[str] | None = None,
    use_sampled_ids: bool = True,
) -> tuple[Path, dict]:
    """Create a minimal store with a recipe file and Parquet stubs."""
    if episode_ids is None:
        episode_ids = ["ep_0001", "ep_0002"]

    store = tmp_path / "store"
    datasets_dir = store / "datasets" / name
    datasets_dir.mkdir(parents=True)

    recipe: dict = {
        "name": name,
        "task": "pick",
        "quality_min": None,
        "quality_max": None,
        "embodiment": None,
        "sampling": "none",
        "limit": None,
        "seed": None,
        "source_episode_ids": episode_ids,
        "sampled_episode_ids": episode_ids if use_sampled_ids else [],
    }
    (datasets_dir / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")

    episodes_dir = store / "episodes"
    episodes_dir.mkdir()
    for ep_id in episode_ids:
        (episodes_dir / f"{ep_id}.parquet").write_bytes(b"PAR1fake")

    return store, recipe


# ---------------------------------------------------------------------------
# AC #1: happy path — Parquet files copied, recipe.json written, exit 0
# ---------------------------------------------------------------------------


class TestExportHappyPath:
    def test_exit_zero_on_success(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        output = tmp_path / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 0, f"Output: {result.output!r}"

    def test_parquet_files_copied(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        output = tmp_path / "export"

        runner.invoke(app, ["export", "pick_v1", "--store", str(store), "--output", str(output)])

        assert (output / "episodes" / "ep_0001.parquet").exists()
        assert (output / "episodes" / "ep_0002.parquet").exists()

    def test_recipe_json_written_to_output(self, tmp_path: Path) -> None:
        store, original_recipe = _make_store(tmp_path)
        output = tmp_path / "export"

        runner.invoke(app, ["export", "pick_v1", "--store", str(store), "--output", str(output)])

        recipe_out = output / "recipe.json"
        assert recipe_out.exists(), "recipe.json must be written to output dir"
        saved = json.loads(recipe_out.read_text())
        assert saved == original_recipe, f"Recipe mismatch: {saved} != {original_recipe}"

    def test_human_readable_output_shows_episode_count(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        output = tmp_path / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert "Exported 2" in result.output or "2 episodes" in result.output


# ---------------------------------------------------------------------------
# AC #2: --json output
# ---------------------------------------------------------------------------


class TestExportJson:
    def test_json_output_is_valid(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        assert result.exit_code == 0, f"Output: {result.output!r}"
        data = json.loads(result.output.strip())
        assert isinstance(data, dict)

    def test_json_has_required_keys(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        data = json.loads(result.output.strip())
        for key in ("episodes_exported", "output_path", "size_bytes"):
            assert key in data, f"Missing key '{key}' in: {data}"

    def test_json_episodes_exported_count(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        data = json.loads(result.output.strip())
        assert data["episodes_exported"] == 2

    def test_json_output_path_is_absolute(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        data = json.loads(result.output.strip())
        assert Path(data["output_path"]).is_absolute(), (
            f"output_path should be absolute: {data['output_path']}"
        )

    def test_json_size_bytes_accurate(self, tmp_path: Path) -> None:
        """size_bytes must equal the sum of all written file sizes."""
        parquet_content = b"PAR1fake"
        store, recipe = _make_store(tmp_path, episode_ids=["ep_0001"])
        # Overwrite with known content so we can predict exact size
        (store / "episodes" / "ep_0001.parquet").write_bytes(parquet_content)
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        data = json.loads(result.output.strip())
        # Manually compute expected size: parquet + recipe.json
        parquet_size = len(parquet_content)
        recipe_json_size = len(json.dumps(recipe, indent=2).encode("utf-8"))
        expected = parquet_size + recipe_json_size
        assert data["size_bytes"] == expected, (
            f"Expected size_bytes={expected}, got {data['size_bytes']}"
        )


# ---------------------------------------------------------------------------
# AC #3: output directory created automatically
# ---------------------------------------------------------------------------


class TestExportCreatesOutputDir:
    def test_creates_nested_output_dir(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path)
        output = tmp_path / "nested" / "deep" / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 0
        assert output.exists(), "Output directory must be created automatically"


# ---------------------------------------------------------------------------
# AC #4: missing dataset → exit 1, name in stderr, suggests tq list
# ---------------------------------------------------------------------------


class TestExportMissingDataset:
    def test_exit_one_when_dataset_missing(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "unknown_dataset", "--store", str(store), "--output", str(output)],
        )

        assert result.exit_code == 1

    def test_output_names_missing_dataset(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "unknown_dataset", "--store", str(store), "--output", str(output)],
        )

        assert "unknown_dataset" in result.output, f"Got: {result.output!r}"

    def test_output_suggests_tq_list(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "unknown_dataset", "--store", str(store), "--output", str(output)],
        )

        assert "tq list" in result.output or "list" in result.output.lower(), (
            f"Expected 'tq list' suggestion. Got: {result.output!r}"
        )

    def test_json_missing_dataset_returns_error_key(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            [
                "export",
                "unknown_dataset",
                "--store",
                str(store),
                "--output",
                str(output),
                "--json",
            ],
        )

        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert "error" in data
        assert "unknown_dataset" in data["error"]


# ---------------------------------------------------------------------------
# AC #5: headless → tq.config.quiet = True
# ---------------------------------------------------------------------------


class TestExportHeadless:
    def test_headless_sets_config_quiet(self, tmp_path: Path) -> None:
        """When _is_headless() returns True, tq.config.quiet must be True before SDK calls."""
        import torq as tq

        store, _ = _make_store(tmp_path)
        output = tmp_path / "export"

        quiet_captured: list[bool] = []

        original_json_loads = json.loads

        def capturing_json_loads(s: str, **kwargs):
            quiet_captured.append(tq.config.quiet)
            return original_json_loads(s, **kwargs)

        with (
            patch("torq.cli.main._is_headless", return_value=True),
            patch("torq.cli.main.json.loads", side_effect=capturing_json_loads),
        ):
            tq.config.quiet = False
            runner.invoke(
                app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
            )

        assert quiet_captured, "json.loads was never called during export"
        assert quiet_captured[0] is True


# ---------------------------------------------------------------------------
# MP4 copying
# ---------------------------------------------------------------------------


class TestExportMp4Files:
    def test_export_copies_mp4_files(self, tmp_path: Path) -> None:
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001"])
        videos_dir = store / "videos"
        videos_dir.mkdir()
        (videos_dir / "ep_0001_cam0.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")

        output = tmp_path / "export"
        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 0
        assert (output / "videos" / "ep_0001_cam0.mp4").exists()

    def test_export_no_error_when_no_videos_dir(self, tmp_path: Path) -> None:
        """If the store has no videos/ dir, export must still succeed silently."""
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001"])
        # Deliberately don't create store/videos/

        output = tmp_path / "export"
        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 0

    def test_export_fallback_to_source_ids_when_sampled_empty(self, tmp_path: Path) -> None:
        """When sampled_episode_ids is empty, fallback to source_episode_ids."""
        store, _ = _make_store(
            tmp_path,
            episode_ids=["ep_0001"],
            use_sampled_ids=False,  # sampled_episode_ids will be []
        )
        output = tmp_path / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 0
        assert (output / "episodes" / "ep_0001.parquet").exists()


# ---------------------------------------------------------------------------
# Missing Parquet files — HIGH fix: warn + exit 1
# ---------------------------------------------------------------------------


class TestExportMissingParquet:
    def test_missing_parquet_exits_one(self, tmp_path: Path) -> None:
        """If a recipe references an episode whose Parquet is gone, exit code must be 1."""
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        # Delete one of the Parquet files after the store is set up
        (store / "episodes" / "ep_0002.parquet").unlink()
        output = tmp_path / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert result.exit_code == 1

    def test_missing_parquet_warns_with_episode_id(self, tmp_path: Path) -> None:
        """The warning message must name the missing episode ID."""
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        (store / "episodes" / "ep_0002.parquet").unlink()
        output = tmp_path / "export"

        result = runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert "ep_0002" in result.output, f"Expected missing ID in output. Got: {result.output!r}"

    def test_present_parquets_still_copied_when_some_missing(self, tmp_path: Path) -> None:
        """Episodes that DO exist must still be exported even when others are missing."""
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        (store / "episodes" / "ep_0002.parquet").unlink()
        output = tmp_path / "export"

        runner.invoke(
            app, ["export", "pick_v1", "--store", str(store), "--output", str(output)]
        )

        assert (output / "episodes" / "ep_0001.parquet").exists()
        assert not (output / "episodes" / "ep_0002.parquet").exists()

    def test_json_output_includes_episodes_skipped(self, tmp_path: Path) -> None:
        """JSON output must include episodes_skipped key listing missing episode IDs."""
        store, _ = _make_store(tmp_path, episode_ids=["ep_0001", "ep_0002"])
        (store / "episodes" / "ep_0002.parquet").unlink()
        output = tmp_path / "export"

        result = runner.invoke(
            app,
            ["export", "pick_v1", "--store", str(store), "--output", str(output), "--json"],
        )

        assert result.exit_code == 1
        # CliRunner mixes stderr into output; find the JSON object in the output
        lines = result.output.strip().splitlines()
        json_line = next(l for l in lines if l.strip().startswith("{"))
        data = json.loads(json_line)
        assert "episodes_skipped" in data, f"Expected 'episodes_skipped' key. Got: {data}"
        assert data["episodes_skipped"] == ["ep_0002"]
