"""Unit tests for `tq list` and `tq info` CLI commands.

Covers:
    - tq list: empty store → exit 0, "No episodes found" (AC #6)
    - tq list: populated store → table with episode IDs (AC #1)
    - tq list --json: stdout is valid JSON array with required keys (AC #2)
    - tq info: happy path → key fields in output (AC #3)
    - tq info --json: stdout is valid JSON object (AC #5)
    - tq info: missing episode → exit 1, episode ID + "tq list" in output (AC #4)
    - tq info: headless → tq.config.quiet = True (AC via _is_headless)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from torq.cli.main import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Fixtures — helpers that build on-disk index + Parquet fixtures
# ---------------------------------------------------------------------------


def _make_index(store: Path, episodes: list[dict]) -> None:
    """Write minimal index shards for the given episodes.

    Each entry in ``episodes`` should have: id, task, embodiment, quality.
    """
    index_dir = store / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    by_task: dict[str, list[str]] = {}
    by_embodiment: dict[str, list[str]] = {}
    quality_list: list[list] = []

    for ep in episodes:
        ep_id = ep["id"]
        task = ep.get("task") or ""
        embodiment = ep.get("embodiment") or ""
        quality = ep.get("quality")

        if task:
            by_task.setdefault(task, []).append(ep_id)
        if embodiment:
            by_embodiment.setdefault(embodiment, []).append(ep_id)
        quality_list.append([quality, ep_id])

    (index_dir / "quality.json").write_text(json.dumps(quality_list))
    (index_dir / "by_task.json").write_text(json.dumps(by_task))
    (index_dir / "by_embodiment.json").write_text(json.dumps(by_embodiment))


def _make_parquet(store: Path, episode_id: str, duration_ns: int = 12_400_000_000) -> None:
    """Write a minimal Parquet file with just a timestamp_ns column."""
    episodes_dir = store / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    ts = [0, duration_ns]
    table = pa.table({"timestamp_ns": pa.array(ts, type=pa.int64())})
    pq.write_table(table, str(episodes_dir / f"{episode_id}.parquet"))


def _make_mock_episode(
    episode_id: str = "ep_0001",
    store: Path | None = None,
    with_quality: bool = True,
) -> MagicMock:
    """Build a MagicMock that mimics an Episode with all fields needed by tq info."""
    ep = MagicMock()
    ep.episode_id = episode_id
    ep.source_path = (
        (store / "episodes" / f"{episode_id}.parquet")
        if store
        else Path(f"/data/{episode_id}.parquet")
    )
    ep.timestamps = list(range(124))
    ep.duration_ns = 12_400_000_000
    ep.observation_keys = ["joint_pos", "wrist_cam"]
    ep.action_keys = ["actions"]
    ep.metadata = {"task": "pick_place", "embodiment": "aloha2"}
    ep.tags = ["success"]

    if with_quality:
        ep.quality = MagicMock(overall=0.83, smoothness=0.90, consistency=0.78, completeness=0.80)
    else:
        ep.quality = None

    return ep


# ---------------------------------------------------------------------------
# tq list — AC #6: empty / missing store
# ---------------------------------------------------------------------------


class TestListEmptyStore:
    def test_missing_store_exits_zero(self, tmp_path: Path) -> None:
        """Non-existent store returns exit 0 and a 'No episodes found' message."""
        nonexistent = tmp_path / "no_store_here"
        result = runner.invoke(app, ["list", "--store", str(nonexistent)])

        assert result.exit_code == 0, f"Output: {result.output!r}"
        assert "No episodes" in result.output or "no episodes" in result.output.lower(), (
            f"Expected 'No episodes' message. Got: {result.output!r}"
        )

    def test_empty_index_exits_zero(self, tmp_path: Path) -> None:
        """Store exists but has no episodes returns exit 0."""
        _make_index(tmp_path, [])  # empty quality.json = []
        result = runner.invoke(app, ["list", "--store", str(tmp_path)])

        assert result.exit_code == 0
        assert "No episodes" in result.output or "0" in result.output


# ---------------------------------------------------------------------------
# tq list — AC #1: human-readable table
# ---------------------------------------------------------------------------


class TestListTable:
    def test_episode_id_appears_in_output(self, tmp_path: Path) -> None:
        """Episode IDs from the index must appear in the table output."""
        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001")

        result = runner.invoke(app, ["list", "--store", str(tmp_path)])

        assert result.exit_code == 0, f"Output: {result.output!r}"
        assert "ep_0001" in result.output

    def test_task_and_quality_appear_in_output(self, tmp_path: Path) -> None:
        """Task name and quality score must appear in the table."""
        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001")

        result = runner.invoke(app, ["list", "--store", str(tmp_path)])

        assert "pick_place" in result.output
        assert "0.83" in result.output

    def test_multiple_episodes_all_shown(self, tmp_path: Path) -> None:
        """All episodes in the index should appear in table output."""
        eps = [
            {"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83},
            {"id": "ep_0002", "task": "push_cube", "embodiment": "franka", "quality": None},
        ]
        _make_index(tmp_path, eps)
        for ep in eps:
            _make_parquet(tmp_path, ep["id"])

        result = runner.invoke(app, ["list", "--store", str(tmp_path)])

        assert "ep_0001" in result.output
        assert "ep_0002" in result.output


# ---------------------------------------------------------------------------
# tq list --json — AC #2
# ---------------------------------------------------------------------------


class TestListJson:
    def test_json_output_is_valid_list(self, tmp_path: Path) -> None:
        """--json output must be a parseable JSON array."""
        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001")

        result = runner.invoke(app, ["list", "--store", str(tmp_path), "--json"])

        assert result.exit_code == 0, f"Output: {result.output!r}"
        data = json.loads(result.output.strip())
        assert isinstance(data, list)

    def test_json_has_required_keys(self, tmp_path: Path) -> None:
        """Each item in the JSON array must have the five required keys."""
        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001")

        result = runner.invoke(app, ["list", "--store", str(tmp_path), "--json"])

        data = json.loads(result.output.strip())
        assert len(data) == 1
        for key in ("id", "task", "embodiment", "duration_s", "quality"):
            assert key in data[0], f"Missing key '{key}' in: {data[0]}"

    def test_json_values_correct(self, tmp_path: Path) -> None:
        """JSON output values must match the index and Parquet data."""
        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001", duration_ns=12_400_000_000)

        result = runner.invoke(app, ["list", "--store", str(tmp_path), "--json"])

        data = json.loads(result.output.strip())
        ep = data[0]
        assert ep["id"] == "ep_0001"
        assert ep["quality"] == 0.83
        assert abs(ep["duration_s"] - 12.4) < 0.01  # within 10ms

    def test_json_empty_store_returns_empty_array(self, tmp_path: Path) -> None:
        """--json on empty store returns [] and exit 0."""
        _make_index(tmp_path, [])

        result = runner.invoke(app, ["list", "--store", str(tmp_path), "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert data == []


# ---------------------------------------------------------------------------
# tq info — AC #3: happy path
# ---------------------------------------------------------------------------


class TestInfoHappyPath:
    def test_key_fields_in_output(self, tmp_path: Path) -> None:
        """Episode ID, task, timesteps must appear in human-readable output."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path)])

        assert result.exit_code == 0, f"Output: {result.output!r}"
        assert "ep_0001" in result.output
        assert "pick_place" in result.output

    def test_quality_breakdown_in_output(self, tmp_path: Path) -> None:
        """Quality scores must appear in human-readable output."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path, with_quality=True)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path)])

        assert "0.83" in result.output  # overall quality

    def test_no_quality_shows_not_scored(self, tmp_path: Path) -> None:
        """Episode without quality report must show '(not scored)'."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path, with_quality=False)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path)])

        assert "not scored" in result.output.lower()


# ---------------------------------------------------------------------------
# tq info --json — AC #5
# ---------------------------------------------------------------------------


class TestInfoJson:
    def test_json_output_is_valid(self, tmp_path: Path) -> None:
        """--json output must be a parseable JSON object."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path), "--json"])

        assert result.exit_code == 0, f"Output: {result.output!r}"
        data = json.loads(result.output.strip())
        assert isinstance(data, dict)

    def test_json_has_required_keys(self, tmp_path: Path) -> None:
        """JSON object must contain all 10 required keys."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path), "--json"])

        data = json.loads(result.output.strip())
        for key in (
            "id",
            "source_path",
            "timesteps",
            "duration_s",
            "modalities",
            "action_keys",
            "task",
            "embodiment",
            "tags",
            "quality",
        ):
            assert key in data, f"Missing key '{key}' in: {data}"

    def test_json_quality_breakdown(self, tmp_path: Path) -> None:
        """JSON quality field must be a dict with overall and sub-scores."""
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        mock_ep = _make_mock_episode("ep_0001", store=tmp_path, with_quality=True)
        with patch("torq.storage.load", return_value=mock_ep):
            result = runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path), "--json"])

        data = json.loads(result.output.strip())
        assert data["quality"]["overall"] == 0.83
        assert data["quality"]["smoothness"] == 0.90


# ---------------------------------------------------------------------------
# tq info — AC #4: missing episode
# ---------------------------------------------------------------------------


class TestInfoMissingEpisode:
    def test_exit_one_when_missing(self, tmp_path: Path) -> None:
        """When episode Parquet does not exist, exit code must be 1."""
        result = runner.invoke(app, ["info", "ep_9999", "--store", str(tmp_path)])

        assert result.exit_code == 1

    def test_output_contains_episode_id(self, tmp_path: Path) -> None:
        """Error output must name the missing episode ID."""
        result = runner.invoke(app, ["info", "ep_9999", "--store", str(tmp_path)])

        assert "ep_9999" in result.output, f"Got: {result.output!r}"

    def test_output_suggests_tq_list(self, tmp_path: Path) -> None:
        """Error output must suggest running 'tq list'."""
        result = runner.invoke(app, ["info", "ep_9999", "--store", str(tmp_path)])

        assert "tq list" in result.output or "list" in result.output.lower(), (
            f"Expected 'tq list' suggestion. Got: {result.output!r}"
        )

    def test_json_missing_episode_returns_error_key(self, tmp_path: Path) -> None:
        """--json on a missing episode must return a JSON object with an 'error' key."""
        result = runner.invoke(
            app, ["info", "ep_9999", "--store", str(tmp_path), "--json"]
        )

        assert result.exit_code == 1
        data = json.loads(result.output.strip())
        assert isinstance(data, dict), f"Expected JSON dict. Got: {result.output!r}"
        assert "error" in data, f"Expected 'error' key in JSON. Got: {data}"
        assert "ep_9999" in data["error"]


# ---------------------------------------------------------------------------
# Headless detection — reused _is_headless() pattern from Story 6.1
# ---------------------------------------------------------------------------


class TestListHeadless:
    def test_headless_sets_config_quiet(self, tmp_path: Path) -> None:
        """When _is_headless() returns True during tq list, tq.config.quiet must be True."""
        import torq as tq

        quiet_during_call: list[bool] = []

        _make_index(
            tmp_path,
            [{"id": "ep_0001", "task": "pick_place", "embodiment": "aloha2", "quality": 0.83}],
        )
        _make_parquet(tmp_path, "ep_0001")

        original_read_duration = __import__(
            "torq.cli.main", fromlist=["_read_duration_s"]
        )._read_duration_s

        def capturing_read_duration(episodes_dir, episode_id):
            quiet_during_call.append(tq.config.quiet)
            return original_read_duration(episodes_dir, episode_id)

        with (
            patch("torq.cli.main._is_headless", return_value=True),
            patch("torq.cli.main._read_duration_s", side_effect=capturing_read_duration),
        ):
            tq.config.quiet = False
            runner.invoke(app, ["list", "--store", str(tmp_path)])

        assert quiet_during_call, "_read_duration_s was never called"
        assert quiet_during_call[0] is True


class TestInfoHeadless:
    def test_headless_sets_config_quiet(self, tmp_path: Path) -> None:
        """When _is_headless() returns True, tq.config.quiet must be True."""
        import torq as tq

        quiet_during_call: list[bool] = []

        def capturing_load(*args, **kwargs):
            quiet_during_call.append(tq.config.quiet)
            return _make_mock_episode("ep_0001", store=tmp_path)

        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        (episodes_dir / "ep_0001.parquet").touch()

        with (
            patch("torq.cli.main._is_headless", return_value=True),
            patch("torq.storage.load", side_effect=capturing_load),
        ):
            tq.config.quiet = False
            runner.invoke(app, ["info", "ep_0001", "--store", str(tmp_path)])

        assert quiet_during_call, "storage.load was never called"
        assert quiet_during_call[0] is True
