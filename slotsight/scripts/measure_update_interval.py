"""Measure SlotSight's status update interval on this machine (R-04 evidence).

Runs a source through the real pipeline (frame sampling, vehicle detection, slot
classification, snapshot and history write) and reports:

- processing time per sampled frame
- interval between consecutive readings
- worst-case gap between changes on the dashboard: the reading interval plus
  the dashboard poll interval
- worst-case age of what the operator sees: processing time plus reading
  interval plus poll interval

Video files are played at recorded speed so they behave like a live camera.

Usage (from the repository root):
    python slotsight/scripts/measure_update_interval.py footage.mp4 --seconds 120
    python slotsight/scripts/measure_update_interval.py rtsp://camera/stream --seconds 300 --write
    python slotsight/scripts/measure_update_interval.py 0 --seconds 60          # webcam 0

History rows go to a temporary file unless --history is given, so a
measurement never mixes into occupancy_history.csv. The camera-shift reference
is temporary too, so a measurement never replaces the lot's reference frame.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as slotsight  # noqa: E402
from camera import CameraMonitor  # noqa: E402
from sources import parse_target, summarize  # noqa: E402

DASHBOARD_POLL_S = 2.0  # keep in sync with POLL_MS in static/app.js
TARGET_S = 5.0  # R-04


def device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
    except ImportError:
        pass
    return "CPU only"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="video file path, stream URL, or camera index")
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to measure (default 60)")
    parser.add_argument("--history", help="CSV to record readings into (default: a temporary file)")
    parser.add_argument("--note", default="", help="free-text note for the report, e.g. 'daytime lot footage'")
    parser.add_argument("--write", action="store_true", help="append the report to slotsight/docs/evaluation.md")
    args = parser.parse_args()

    scratch = Path(tempfile.mkdtemp(prefix="slotsight-measure-"))
    slotsight.HISTORY_PATH = Path(args.history) if args.history else scratch / "measure-history.csv"
    # The camera-shift reference belongs to the real lot layout: never let a measurement set it.
    slotsight.camera = CameraMonitor(scratch / "reference_frame.jpg")
    try:
        model = slotsight.require_model_blocking()  # downloads the default model on first run
    except slotsight.DetectionError as error:
        print(error, file=sys.stderr)
        return 1
    try:
        kind, target, name = parse_target(args.source)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    settings = slotsight.load_settings()
    print(f"Measuring {name} for up to {args.seconds:g} s (sampling every {settings['sample_interval_s']:g} s)...")
    worker = slotsight.start_source(kind, target, name, realtime=True)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and worker.running:
        time.sleep(0.25)
    worker.stop()
    status = worker.status()
    timings = worker.timings()
    if status["state"] == "error":
        print(f"Source error: {status['error']}", file=sys.stderr)
        return 1
    if len(timings["update_interval_s"]) < 2:
        print("Too few readings to measure; run for longer.", file=sys.stderr)
        return 1

    processing = summarize(timings["processing_s"])
    intervals = summarize(timings["update_interval_s"])
    worst_gap = intervals["max"] + DASHBOARD_POLL_S
    worst_age = processing["max"] + intervals["max"] + DASHBOARD_POLL_S
    frame = slotsight.latest_frame
    size = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "unknown"
    verdict = "PASS" if worst_gap <= TARGET_S else "FAIL"

    report = f"""
### R-04 timing run: {datetime.now().strftime('%Y-%m-%d %H:%M')}

- Source: {name} ({kind}{', played at recorded speed' if kind == 'file' else ''}){f' - {args.note}' if args.note else ''}
- Machine: {platform.processor() or platform.machine()} / {device_name()} / Python {platform.python_version()}
- Model: {model['name']} ({model['id']}); frame size {size}; {len(slotsight.load_slots())} slots
- Sample interval {settings['sample_interval_s']:g} s; dashboard poll {DASHBOARD_POLL_S:g} s; {status['frames_processed']} readings over {args.seconds:g} s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | {processing['mean']} | {processing['p95']} | {processing['max']} |
| Interval between readings (s) | {intervals['mean']} | {intervals['p95']} | {intervals['max']} |

- Worst-case gap between dashboard updates: {worst_gap:.2f} s (target <= {TARGET_S:g} s): **{verdict}**
- Worst-case age of the displayed reading: {worst_age:.2f} s
"""
    print(report)
    if args.write:
        evaluation = ROOT / "docs" / "evaluation.md"
        evaluation.parent.mkdir(exist_ok=True)
        if not evaluation.exists():
            evaluation.write_text("# SlotSight Evaluation\n\nMeasured evidence for the requirements in `requirements.md`.\n\n## R-04: Update interval\n", encoding="utf-8")
        with evaluation.open("a", encoding="utf-8") as file:
            file.write(report)
        print(f"Appended to {evaluation}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
