"""Validation report generator for Torq R1 Alpha (SC-1.0).

Runs all validations and produces a summary report in human-readable and JSON format.

Usage:
    python tests/acceptance/run_validation_report.py
    python tests/acceptance/run_validation_report.py --output report.json
    TORQ_TEST_ALOHA2_PATH=/data/aloha2.mcap python tests/acceptance/run_validation_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Ensure src/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import torq as tq

FIXTURES_DATA = Path(__file__).parent.parent / "fixtures" / "data"


def _run_pipeline_on_file(mcap_path: Path, tmp_path: Path) -> dict:
    """Run the full pipeline on one MCAP file and return a result dict."""
    start = time.perf_counter()
    error = None
    episodes = []
    scores: list[float] = []

    try:
        tq.config.quiet = True
        episodes = tq.ingest(mcap_path)
        tq.quality.score(episodes)
        for ep in episodes:
            tq.save(ep, path=tmp_path, quiet=True)
        scores = [ep.quality.overall for ep in episodes if ep.quality is not None]
    except Exception as exc:
        error = f"{exc}\n{traceback.format_exc()}"
    finally:
        tq.config.quiet = False

    elapsed = time.perf_counter() - start
    return {
        "path": str(mcap_path),
        "episode_count": len(episodes),
        "quality_scores": [round(s, 4) for s in scores],
        "quality_min": round(min(scores), 4) if scores else None,
        "quality_max": round(max(scores), 4) if scores else None,
        "quality_varied": len(set(round(s, 6) for s in scores)) > 1 if len(scores) >= 2 else None,
        "elapsed_s": round(elapsed, 2),
        "status": "PASS" if error is None and len(episodes) > 0 else "FAIL",
        "error": error,
    }


def run_all_validations() -> dict:
    """Run all validations and return a consolidated report dict."""
    import tempfile

    report: dict = {
        "torq_version": tq.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [],
        "summary": {},
    }

    # ── Bundled fixtures ──────────────────────────────────────────────────────
    bundled = [
        ("sample_mcap", FIXTURES_DATA / "sample.mcap"),
        ("boundary_detection_mcap", FIXTURES_DATA / "boundary_detection.mcap"),
    ]
    for name, path in bundled:
        if not path.exists():
            report["results"].append({
                "name": name,
                "path": str(path),
                "status": "SKIP",
                "error": "fixture not found — run: python tests/fixtures/generate_fixtures.py",
            })
            continue
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_pipeline_on_file(path, Path(tmp))
            result["name"] = name
            report["results"].append(result)

    # ── Real-world datasets ───────────────────────────────────────────────────
    real_data = {
        "aloha2": os.environ.get("TORQ_TEST_ALOHA2_PATH"),
        "franka": os.environ.get("TORQ_TEST_FRANKA_PATH"),
    }
    for platform, raw_path in real_data.items():
        if raw_path is None:
            report["results"].append({
                "name": platform,
                "status": "SKIP",
                "error": f"TORQ_TEST_{platform.upper()}_PATH not set",
            })
            continue
        path = Path(raw_path)
        if not path.exists():
            report["results"].append({
                "name": platform,
                "path": str(path),
                "status": "SKIP",
                "error": f"file not found at {path}",
            })
            continue
        with tempfile.TemporaryDirectory() as tmp:
            result = _run_pipeline_on_file(path, Path(tmp))
            result["name"] = platform
            report["results"].append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    statuses = [r["status"] for r in report["results"]]
    report["summary"] = {
        "total": len(statuses),
        "passed": statuses.count("PASS"),
        "failed": statuses.count("FAIL"),
        "skipped": statuses.count("SKIP"),
        "overall": "PASS" if statuses.count("FAIL") == 0 and statuses.count("PASS") > 0 else "FAIL",
    }

    return report


def _print_report(report: dict) -> None:
    """Print a human-readable report to stdout."""
    print(f"\n{'═' * 60}")
    print("  Torq R1 Alpha — Validation Report")
    print(f"  Version: {report['torq_version']}  |  {report['timestamp']}")
    print(f"{'═' * 60}\n")

    for result in report["results"]:
        name = result.get("name", "?")
        status = result.get("status", "?")
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️ "}.get(status, "?")

        print(f"  {icon} {name:30s}  [{status}]")
        if status == "PASS":
            qmin = result.get("quality_min", "?")
            qmax = result.get("quality_max", "?")
            qrange = f"{qmin:.3f}–{qmax:.3f}" if isinstance(qmin, float) else "?"
            print(
                f"       episodes={result.get('episode_count', '?')}  "
                f"quality={qrange}  time={result.get('elapsed_s', '?')}s"
            )
        elif status == "FAIL":
            print(f"       ERROR: {result.get('error', 'unknown')}")
        elif status == "SKIP":
            print(f"       {result.get('error', '')}")

    s = report["summary"]
    print(f"\n{'─' * 60}")
    print(f"  Results: {s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped")
    print(f"  Overall: {'✅ PASS' if s['overall'] == 'PASS' else '❌ FAIL'}")
    print(f"{'═' * 60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Torq R1 validation report")
    parser.add_argument("--output", metavar="FILE", help="Write JSON report to file")
    args = parser.parse_args()

    report = run_all_validations()
    _print_report(report)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(f"JSON report written to: {args.output}")

    return 0 if report["summary"]["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
