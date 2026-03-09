"""Unit tests for tq.ingest() — auto-detection and bulk import entry point.

Covers:
    - Auto-dispatches MCAP files (AC #1)
    - Auto-dispatches HDF5 files (AC #1)
    - Auto-dispatches LeRobot directories (AC #1)
    - Empty directory returns empty list (AC #3)
    - Unknown format raises TorqIngestError (AC #4)
    - Bulk mode skips corrupt files (AC #2)
    - Parquet files raise helpful error (Task 1 spec)
    - HDF5 magic byte verification (Task 1 spec)

Fixtures: sample_mcap, robomimic_hdf5, lerobot_dataset are provided
by tests/conftest.py (shared across test modules).
"""

import logging
from pathlib import Path

import pytest


class TestIngestAuto:
    def test_ingest_auto_dispatches_mcap(self, sample_mcap: Path) -> None:
        """tq.ingest(mcap_path) must return non-empty list of Episodes."""
        from torq.episode import Episode
        from torq.ingest import ingest

        episodes = ingest(sample_mcap)

        assert len(episodes) > 0, "Expected at least 1 episode from sample.mcap"
        for ep in episodes:
            assert isinstance(ep, Episode)

    def test_ingest_auto_dispatches_hdf5(self, robomimic_hdf5: Path) -> None:
        """tq.ingest(hdf5_path) must return non-empty list of Episodes."""
        from torq.episode import Episode
        from torq.ingest import ingest

        episodes = ingest(robomimic_hdf5)

        assert len(episodes) > 0, "Expected at least 1 episode from robomimic_simple.hdf5"
        for ep in episodes:
            assert isinstance(ep, Episode)

    def test_ingest_auto_dispatches_lerobot(self, lerobot_dataset: Path) -> None:
        """tq.ingest(lerobot_dir) must return non-empty list of Episodes."""
        from torq.episode import Episode
        from torq.ingest import ingest

        episodes = ingest(lerobot_dataset)

        assert len(episodes) > 0, "Expected at least 1 episode from lerobot/ fixture"
        for ep in episodes:
            assert isinstance(ep, Episode)

    def test_ingest_auto_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty directory must return [] with no exception raised."""
        from torq.ingest import ingest

        episodes = ingest(tmp_path)
        assert episodes == []

    def test_ingest_auto_unknown_format_raises_torq_ingest_error(self, tmp_path: Path) -> None:
        """Unrecognised file extension must raise TorqIngestError with 'unknown' label."""
        file_path = tmp_path / "mystery.xyz"
        file_path.write_text("not a real format")

        from torq.errors import TorqIngestError
        from torq.ingest import ingest

        with pytest.raises(TorqIngestError, match="'unknown'"):
            ingest(file_path)

    def test_ingest_auto_bulk_skips_corrupt_file(self, tmp_path: Path, sample_mcap: Path) -> None:
        """Directory with valid + corrupt files returns episodes from valid only."""
        import shutil

        from torq.ingest import ingest

        # Copy a valid MCAP into the temp directory
        shutil.copy2(sample_mcap, tmp_path / "good.mcap")
        # Create a corrupt MCAP file
        (tmp_path / "bad.mcap").write_bytes(b"not a real mcap file")

        episodes = ingest(tmp_path)

        # Should get episodes from the valid file only, no exception
        assert len(episodes) > 0, "Expected episodes from the valid MCAP file"


class TestIngestParquet:
    def test_parquet_file_raises_helpful_error(self, tmp_path: Path) -> None:
        """A .parquet file must raise TorqIngestError with LeRobot guidance."""
        parquet_path = tmp_path / "data.parquet"
        parquet_path.write_bytes(b"PAR1fake")

        from torq.errors import TorqIngestError
        from torq.ingest import ingest

        with pytest.raises(TorqIngestError, match=r"\.parquet files are not dataset roots"):
            ingest(parquet_path)


class TestDetectHdf5Magic:
    def test_hdf5_good_magic_detected(self, robomimic_hdf5: Path) -> None:
        """A valid HDF5 file must be detected as 'hdf5'."""
        from torq.ingest._detect import detect_format

        assert detect_format(robomimic_hdf5) == "hdf5"

    def test_hdf5_bad_magic_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """A .hdf5 file with wrong magic bytes still detects as 'hdf5' but warns."""
        bad_hdf5 = tmp_path / "fake.hdf5"
        bad_hdf5.write_bytes(b"NOT_HDF5_MAGIC_BYTES")

        from torq.ingest._detect import detect_format

        with caplog.at_level(logging.WARNING, logger="torq.ingest._detect"):
            result = detect_format(bad_hdf5)

        assert result == "hdf5"
        assert "magic bytes do not match" in caplog.text
