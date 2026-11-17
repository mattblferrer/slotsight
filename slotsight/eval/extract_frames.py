"""Pull frames out of lot recordings for labeling (see docs/recording-guide.md).

Each video becomes one session folder of JPEG frames:

    slotsight/data/eval_frames/<set>/<session>/0000030.0s.jpg ...

- <set> is "test" (default) or "tune". Tune sessions are kept apart for threshold
  tuning, so `evaluate.py --exclude "tune/*"` reports only on the others.
- <session> defaults to the video's file name. Keep one recording per session:
  frames seconds apart look almost identical, so a session must never be split
  between tune and test.
- A session.json beside the frames records the source video, the sampling and the
  condition you give, which the /label page uses as each frame's default condition.

The frames stay on this machine (data/ is gitignored); only labels are committed.

Usage (from the repository root):
    python slotsight/eval/extract_frames.py D:/lot/2026-11-20-morning.mp4 --condition daytime
    python slotsight/eval/extract_frames.py evening1.mp4 evening2.mp4 --condition evening --every 20
    python slotsight/eval/extract_frames.py rain.mp4 --condition rain --set tune
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import labeling  # noqa: E402

DEFAULT_FRAMES_DIR = ROOT / "data" / "eval_frames"
DEFAULT_EVERY_S = 30.0  # far enough apart that neighbouring frames aren't near-duplicates
SETS = ("test", "tune")


def session_name(video: Path) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", video.stem).strip("-.")
    return name or "session"


def extract(video: Path, out_dir: Path, every_s: float, start_s: float = 0.0, end_s: float | None = None, limit: int | None = None) -> list[Path]:
    """Save one frame every `every_s` seconds of video time. Returns the files written."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Can't open {video} as a video.")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 0 < fps < 1000 else 30.0
        step = max(1, round(fps * every_s))
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        index = 0
        while True:
            offset = index / fps
            if end_s is not None and offset > end_s:
                break
            if index % step == 0 and offset >= start_s:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                path = out_dir / f"{offset:09.1f}s.jpg"
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    encoded.tofile(str(path))
                    written.append(path)
                if limit and len(written) >= limit:
                    break
            elif not capture.grab():
                break
            index += 1
        return written
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path, help="lot recordings, one session each")
    parser.add_argument("--condition", choices=labeling.CONDITIONS, help="default condition for these frames (you can change it per frame when labeling)")
    parser.add_argument("--set", dest="split", choices=SETS, default="test", help="test (default) or tune (kept apart for threshold tuning)")
    parser.add_argument("--every", type=float, default=DEFAULT_EVERY_S, help=f"seconds of video between frames (default {DEFAULT_EVERY_S:g})")
    parser.add_argument("--start", type=float, default=0.0, help="skip this many seconds at the start (e.g. while mounting the camera)")
    parser.add_argument("--end", type=float, help="stop at this many seconds into the video")
    parser.add_argument("--limit", type=int, help="at most this many frames per video")
    parser.add_argument("--session", help="session folder name (only with a single video; default: the file name)")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR, help="where frames go (default slotsight/data/eval_frames)")
    parser.add_argument("--force", action="store_true", help="add to a session folder that already has frames")
    args = parser.parse_args()
    if args.every <= 0:
        parser.error("--every must be positive")
    if args.session and len(args.videos) > 1:
        parser.error("--session names one session; give one video at a time")

    failed = 0
    for video in args.videos:
        session = args.session or session_name(video)
        out_dir = args.frames_dir / args.split / session
        if out_dir.exists() and any(out_dir.glob("*.jpg")) and not args.force:
            print(f"{out_dir} already has frames; use --session for a new name or --force to add to it.", file=sys.stderr)
            failed += 1
            continue
        other = args.frames_dir / ("tune" if args.split == "test" else "test") / session
        if other.exists():
            print(f"Session {session} is already in {other.parent.name}/; a session must not be split between tune and test.", file=sys.stderr)
            failed += 1
            continue
        try:
            written = extract(video, out_dir, args.every, args.start, args.end, args.limit)
        except ValueError as error:
            print(error, file=sys.stderr)
            failed += 1
            continue
        info = {
            "source": video.name,
            "set": args.split,
            "condition": args.condition or "",
            "every_s": args.every,
            "start_s": args.start,
            "frames": len(written),
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (out_dir / labeling.SESSION_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"{video.name}: {len(written)} frames -> {out_dir}")
    if not failed:
        print("Next: start the dashboard and open http://127.0.0.1:5000/label")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
