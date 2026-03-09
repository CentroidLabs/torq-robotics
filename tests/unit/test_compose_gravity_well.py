"""Unit tests for gravity well behavior in tq.compose() and tq.query() (Story 4.5).

Uses capsys to capture stdout. compose() tests mock query_index+load;
query() tests write minimal index files to tmp_path (since _query.py reads
JSON directly rather than using query_index from storage.index).

Covers:
    - compose() ≥5 episodes → GW-SDK-02 fires (episode count + datatorq.ai)
    - compose() 1–4 episodes → GW-SDK-05 fires (task/embodiment hint, NOT GW-SDK-02)
    - compose() 0 episodes → no gravity well
    - query() 1–4 results → GW-SDK-05 fires
    - query() ≥5 results → no gravity well
    - query() 0 results → no gravity well
    - config.quiet=True suppresses all compose gravity wells
    - config.quiet=True suppresses all query gravity wells
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import torq as tq
from torq._config import config
from torq.episode import Episode


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_episodes(n: int, task: str = "pick") -> list[MagicMock]:
    eps = []
    for i in range(n):
        ep = MagicMock(spec=Episode)
        ep.episode_id = f"ep_{i:03d}"
        ep.metadata = {"task": task}
        ep.quality = None
        eps.append(ep)
    return eps


def _ids(n: int) -> list[str]:
    return [f"ep_{i:03d}" for i in range(n)]


def _write_index(store_path: Path, n: int, task: str = "pick") -> list[MagicMock]:
    """Write a minimal quality.json (and by_task.json) index for n episodes."""
    index_root = store_path / "index"
    index_root.mkdir(parents=True, exist_ok=True)

    ids = _ids(n)
    quality_data = [[None, ep_id] for ep_id in ids]
    (index_root / "quality.json").write_text(json.dumps(quality_data), encoding="utf-8")

    by_task = {task: ids}
    (index_root / "by_task.json").write_text(json.dumps(by_task), encoding="utf-8")

    eps = _make_episodes(n, task=task)
    return eps


@pytest.fixture(autouse=True)
def reset_quiet(monkeypatch):
    monkeypatch.setattr(config, "quiet", False)


# ── tq.compose() gravity wells ────────────────────────────────────────────────

class TestComposeGravityWell:
    def test_ge5_episodes_fires_gw_sdk_02(self, tmp_path: Path, capsys) -> None:
        """≥5 episodes → GW-SDK-02: episode count + datatorq.ai URL."""
        eps = _make_episodes(6)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(6)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(store_path=tmp_path)

        out = capsys.readouterr().out
        assert "6" in out
        assert "datatorq.ai" in out

    def test_ge5_episodes_not_gw_sdk_05(self, tmp_path: Path, capsys) -> None:
        """≥5 episodes → GW-SDK-02, NOT the GW-SDK-05 community hint."""
        eps = _make_episodes(10)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(10)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(task="pick", store_path=tmp_path)

        out = capsys.readouterr().out
        assert "datatorq.ai" in out
        assert "community" not in out.lower()

    def test_1_to_4_episodes_fires_gw_sdk_05(self, tmp_path: Path, capsys) -> None:
        """1–4 episodes → GW-SDK-05: community dataset hint fires with task context."""
        eps = _make_episodes(3)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(3)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(task="pick", store_path=tmp_path)

        out = capsys.readouterr().out
        assert "datatorq.ai" in out
        # GW-SDK-05-specific content: the "Only N episode(s) matched" phrasing
        assert "Only" in out
        # Task name is included in the message
        assert "pick" in out

    def test_1_to_4_episodes_not_gw_sdk_02(self, tmp_path: Path, capsys) -> None:
        """1–4 episodes → GW-SDK-05 wins; GW-SDK-02 'Composed dataset' message must not appear."""
        eps = _make_episodes(2)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(2)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(task="pick", store_path=tmp_path)

        out = capsys.readouterr().out
        assert "Composed dataset" not in out
        assert "datatorq.ai" in out

    def test_0_episodes_no_gravity_well(self, tmp_path: Path, capsys) -> None:
        """0 episodes → no gravity well."""
        with patch("torq.compose._compose.query_index", return_value=[]):
            tq.compose(store_path=tmp_path)

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_quiet_suppresses_gw_sdk_02(self, tmp_path: Path, capsys, monkeypatch) -> None:
        """config.quiet=True suppresses GW-SDK-02 on ≥5 episodes."""
        monkeypatch.setattr(config, "quiet", True)
        eps = _make_episodes(8)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(8)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(store_path=tmp_path)

        assert capsys.readouterr().out == ""

    def test_quiet_suppresses_gw_sdk_05(self, tmp_path: Path, capsys, monkeypatch) -> None:
        """config.quiet=True suppresses GW-SDK-05 on 1–4 episodes."""
        monkeypatch.setattr(config, "quiet", True)
        eps = _make_episodes(2)
        ep_map = {e.episode_id: e for e in eps}
        with (
            patch("torq.compose._compose.query_index", return_value=_ids(2)),
            patch("torq.compose._compose.load", side_effect=lambda eid, sp: ep_map[eid]),
        ):
            tq.compose(task="pick", store_path=tmp_path)

        assert capsys.readouterr().out == ""


# ── tq.query() gravity wells ─────────────────────────────────────────────────

class TestQueryGravityWell:
    def test_1_to_4_results_fires_gw_sdk_05(self, tmp_path: Path, capsys) -> None:
        """1–4 query results → GW-SDK-05 fires with datatorq.ai URL."""
        eps = _write_index(tmp_path, n=3, task="pick")
        ep_map = {e.episode_id: e for e in eps}
        with patch("torq.compose._query.load", side_effect=lambda eid, sp: ep_map[eid]):
            list(tq.query(store_path=tmp_path))

        assert "datatorq.ai" in capsys.readouterr().out

    def test_ge5_results_no_gravity_well(self, tmp_path: Path, capsys) -> None:
        """≥5 query results → no gravity well (GW-SDK-02 is compose-only)."""
        eps = _write_index(tmp_path, n=7)
        ep_map = {e.episode_id: e for e in eps}
        with patch("torq.compose._query.load", side_effect=lambda eid, sp: ep_map[eid]):
            list(tq.query(store_path=tmp_path))

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_0_results_no_gravity_well(self, tmp_path: Path, capsys) -> None:
        """0 query results → no gravity well."""
        list(tq.query(store_path=tmp_path))  # no index files → 0 results

        assert "datatorq.ai" not in capsys.readouterr().out

    def test_quiet_suppresses_query_gw_sdk_05(self, tmp_path: Path, capsys, monkeypatch) -> None:
        """config.quiet=True suppresses GW-SDK-05 in tq.query()."""
        monkeypatch.setattr(config, "quiet", True)
        eps = _write_index(tmp_path, n=2)
        ep_map = {e.episode_id: e for e in eps}
        with patch("torq.compose._query.load", side_effect=lambda eid, sp: ep_map[eid]):
            list(tq.query(store_path=tmp_path))

        assert capsys.readouterr().out == ""

    def test_gw_fires_before_first_yield(self, tmp_path: Path, capsys) -> None:
        """Gravity well fires strictly before load() is called — verified by call order."""
        from torq._gravity_well import _gravity_well as real_gw

        eps = _write_index(tmp_path, n=3, task="pour")
        ep_map = {e.episode_id: e for e in eps}
        call_order: list[str] = []

        def tracking_gw(msg: str, feature: str) -> None:
            call_order.append("gw")
            real_gw(msg, feature)  # still fires so capsys captures it

        def tracking_load(eid: str, sp):
            call_order.append("load")
            return ep_map.get(eid, eps[0])

        with (
            patch("torq.compose._query._gravity_well", side_effect=tracking_gw),
            patch("torq.compose._query.load", side_effect=tracking_load),
        ):
            gen = tq.query(store_path=tmp_path)
            next(gen, None)  # consume only first item

        # Gravity well must appear before any load() call
        assert "gw" in call_order
        assert "load" in call_order
        assert call_order.index("gw") < call_order.index("load"), (
            f"Expected GW before load, got order: {call_order}"
        )
        assert "datatorq.ai" in capsys.readouterr().out
