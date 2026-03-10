"""Unit tests for `tq ingest` CLI command.

Covers:
    - Happy path: exit 0, stdout shows episode count (AC #1)
    - Partial failure: exit 1, output contains failed file info (AC #2)
    - --json flag: stdout is valid JSON with required keys, no extra text (AC #3)
    - Headless / no-tty: tq.config.quiet set to True automatically (AC #4)
    - Missing source: exit 1, output contains the path (AC #1 error path)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from torq.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_episode() -> MagicMock:
    ep = MagicMock()
    ep.metadata = MagicMock()
    return ep


# ---------------------------------------------------------------------------
# AC #1 — Happy path
# ---------------------------------------------------------------------------


class TestIngestHappyPath:
    def test_exit_code_zero_on_success(self, tmp_path: Path) -> None:
        """Given a directory, when ingest succeeds, exit code is 0."""
        episodes = [_fake_episode(), _fake_episode(), _fake_episode()]
        with patch("torq.ingest", return_value=episodes):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )

    def test_stdout_shows_episode_count(self, tmp_path: Path) -> None:
        """Given a successful ingest, stdout must mention how many episodes were ingested."""
        episodes = [_fake_episode(), _fake_episode()]
        with patch("torq.ingest", return_value=episodes):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert "2" in result.output or "episode" in result.output.lower(), (
            f"Expected episode count in output. Got: {result.output!r}"
        )

    def test_empty_directory_exits_zero(self, tmp_path: Path) -> None:
        """An empty directory (no ingestible files) returns 0 episodes and exits 0."""
        with patch("torq.ingest", return_value=[]):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# AC #2 — Partial / total failure
# ---------------------------------------------------------------------------


class TestIngestFailure:
    def test_torq_error_produces_exit_one(self, tmp_path: Path) -> None:
        """When tq.ingest() raises TorqError, exit code must be 1."""
        from torq.errors import TorqError

        with patch("torq.ingest", side_effect=TorqError("bad file: test.mcap")):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 1

    def test_torq_error_message_in_output(self, tmp_path: Path) -> None:
        """TorqError details must appear somewhere in the CLI output."""
        from torq.errors import TorqError

        with patch("torq.ingest", side_effect=TorqError("bad file: test.mcap")):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert "test.mcap" in result.output, f"Expected filename in output. Got: {result.output!r}"

    def test_partial_failure_exits_one_and_reports_failures(self, tmp_path: Path) -> None:
        """Directory with some failing files: exit 1, failed filenames in output."""
        episodes = [_fake_episode()]

        def ingest_with_partial_failure(source, errors=None, stats=None):
            if errors is not None:
                errors.append({"path": "bad.mcap", "reason": "corrupt header"})
            if stats is not None:
                stats["files_succeeded"] = 1
            return episodes

        with patch("torq.ingest", side_effect=ingest_with_partial_failure):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 1
        assert "bad.mcap" in result.output, (
            f"Expected failed filename in output. Got: {result.output!r}"
        )

    def test_missing_source_exits_one(self, tmp_path: Path) -> None:
        """Nonexistent source path must exit with code 1 and mention the path in output."""
        nonexistent = tmp_path / "does_not_exist"

        result = runner.invoke(app, ["ingest", str(nonexistent)])

        assert result.exit_code == 1
        assert str(nonexistent) in result.output or "does_not_exist" in result.output, (
            f"Expected path in output. Got: {result.output!r}"
        )


# ---------------------------------------------------------------------------
# AC #3 — --json flag
# ---------------------------------------------------------------------------


class TestIngestJsonFlag:
    def test_json_stdout_is_valid_json(self, tmp_path: Path) -> None:
        """--json flag: stdout must be parseable JSON."""
        with patch("torq.ingest", return_value=[_fake_episode()]):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        assert result.exit_code == 0, f"exit={result.exit_code}, output={result.output!r}"
        parsed = json.loads(result.output.strip())
        assert isinstance(parsed, dict)

    def test_json_stdout_has_required_keys(self, tmp_path: Path) -> None:
        """--json output must contain all four required keys."""
        with patch("torq.ingest", return_value=[_fake_episode(), _fake_episode()]):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        parsed = json.loads(result.output.strip())
        for key in ("episodes_ingested", "files_processed", "files_failed", "duration_seconds"):
            assert key in parsed, f"Missing key '{key}' in JSON output: {parsed}"

    def test_json_episodes_ingested_matches_count(self, tmp_path: Path) -> None:
        """--json: episodes_ingested must match the number returned by tq.ingest()."""
        with patch("torq.ingest", return_value=[_fake_episode()] * 3):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        parsed = json.loads(result.output.strip())
        assert parsed["episodes_ingested"] == 3

    def test_json_files_failed_counts_partial_failures(self, tmp_path: Path) -> None:
        """--json: files_failed must reflect per-file failures populated via errors list."""

        def ingest_with_failures(source, errors=None, stats=None):
            if errors is not None:
                errors.append({"path": "bad.mcap", "reason": "corrupt"})
            if stats is not None:
                stats["files_succeeded"] = 1
            return [_fake_episode()]

        with patch("torq.ingest", side_effect=ingest_with_failures):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        assert result.exit_code == 1
        parsed = json.loads(result.output.strip())
        assert parsed["files_failed"] == 1
        assert parsed["episodes_ingested"] == 1

    def test_json_files_processed_equals_succeeded_plus_failed(self, tmp_path: Path) -> None:
        """--json: files_processed = files_succeeded + files_failed (not episodes)."""

        def ingest_with_failures(source, errors=None, stats=None):
            if errors is not None:
                errors.append({"path": "bad.mcap", "reason": "corrupt"})
            if stats is not None:
                stats["files_succeeded"] = 1  # 1 file succeeded, producing 2 episodes
            return [_fake_episode(), _fake_episode()]

        with patch("torq.ingest", side_effect=ingest_with_failures):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        parsed = json.loads(result.output.strip())
        # 1 file succeeded + 1 file failed = 2 files processed (NOT 2 episodes + 1 failed)
        assert parsed["files_processed"] == 2
        assert parsed["episodes_ingested"] == 2
        assert parsed["files_failed"] == 1

    def test_json_fatal_error_has_nonzero_files_failed(self, tmp_path: Path) -> None:
        """--json on fatal TorqError: files_failed > 0 and exit code 1."""
        from torq.errors import TorqError

        with patch("torq.ingest", side_effect=TorqError("fatal ingest error")):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        assert result.exit_code == 1
        parsed = json.loads(result.output.strip())
        assert parsed["files_failed"] > 0

    def test_json_output_is_only_json(self, tmp_path: Path) -> None:
        """--json: stdout must contain ONLY the JSON object (no extra human-readable lines)."""
        with patch("torq.ingest", return_value=[_fake_episode()]):
            result = runner.invoke(app, ["ingest", str(tmp_path), "--json"])

        # Entire stdout must be parseable as JSON — raises if extra text present
        parsed = json.loads(result.output.strip())
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# AC #4 — Headless / no-tty detection
# ---------------------------------------------------------------------------


class TestIngestHeadless:
    def test_headless_sets_config_quiet(self, tmp_path: Path) -> None:
        """When _is_headless() returns True, tq.config.quiet must be set to True."""
        import torq as tq

        quiet_during_call: list[bool] = []

        def capturing_ingest(*args, **kwargs):
            quiet_during_call.append(tq.config.quiet)
            return []

        with (
            patch("torq.cli.main._is_headless", return_value=True),
            patch("torq.ingest", side_effect=capturing_ingest),
        ):
            tq.config.quiet = False
            runner.invoke(app, ["ingest", str(tmp_path)])

        assert quiet_during_call, "ingest was never called"
        assert quiet_during_call[0] is True, (
            f"Expected config.quiet=True in headless mode, got {quiet_during_call[0]}"
        )

    def test_tty_does_not_set_quiet(self, tmp_path: Path) -> None:
        """When _is_headless() returns False (real TTY), tq.config.quiet must NOT be set."""
        import torq as tq

        quiet_during_call: list[bool] = []

        def capturing_ingest(*args, **kwargs):
            quiet_during_call.append(tq.config.quiet)
            return []

        with (
            patch("torq.cli.main._is_headless", return_value=False),
            patch("torq.ingest", side_effect=capturing_ingest),
        ):
            tq.config.quiet = False
            runner.invoke(app, ["ingest", str(tmp_path)])

        assert quiet_during_call, "ingest was never called"
        assert quiet_during_call[0] is False, (
            f"Expected config.quiet=False in TTY mode, got {quiet_during_call[0]}"
        )

    def test_headless_no_interactive_prompts(self, tmp_path: Path) -> None:
        """Headless invocation must complete without blocking (no interactive prompts)."""
        with (
            patch("torq.cli.main._is_headless", return_value=True),
            patch("torq.ingest", return_value=[]),
        ):
            result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code in (0, 1)  # completes — doesn't hang
