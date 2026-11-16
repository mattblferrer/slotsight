"""Frame sources for SlotSight: recorded video files and live streams.

A SourceWorker reads frames on a background thread and hands one frame to a
callback every `interval_s` seconds: of video time for files, of wall-clock
time for live streams and cameras. The callback does detection and recording;
this module only deals with reading and pacing frames and with reporting
progress and timing.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import cv2

# (frame, offset into the video in seconds, or None for live sources)
FrameHandler = Callable[[Any, "float | None"], None]

FALLBACK_FPS = 30.0
TIMING_WINDOW = 500


class SourceError(RuntimeError):
    """The source could not be opened or stopped delivering frames."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def redact(target: str) -> str:
    """Drop credentials from a stream URL before it is shown or logged."""
    parts = urlsplit(target)
    if parts.password or parts.username:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return target


def parse_target(raw: str) -> tuple[str, str | int, str]:
    """Turn user input into (kind, target, display name).

    kind is "camera" (a device index such as 0), "stream" (a URL such as
    rtsp://...), or "file" (a path to a video on this machine).
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Enter a stream URL, a camera index such as 0, or the path to a video file.")
    if text.isdigit():
        return "camera", int(text), f"camera {text}"
    if "://" in text:
        return "stream", text, redact(text)
    path = Path(text.strip('"')).expanduser()
    if path.is_file():
        return "file", str(path), path.name
    raise ValueError(f"No video file at {path}. Enter a stream URL, a camera index such as 0, or an existing video path.")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": round(statistics.fmean(values), 3),
        "p95": round(percentile(values, 0.95), 3),
        "max": round(max(values), 3),
    }


class SourceWorker:
    """Reads one source on a background thread and samples frames from it."""

    def __init__(
        self,
        kind: str,
        target: str | int,
        name: str,
        interval_s: float,
        on_frame: FrameHandler,
        realtime: bool = False,
        cleanup_path: str | None = None,
        max_reconnects: int = 5,
        reconnect_delay_s: float = 2.0,
    ) -> None:
        self.kind = kind
        self.target = target
        self.name = name
        self.interval_s = interval_s
        self.on_frame = on_frame
        # Files normally run as fast as inference allows; realtime plays them at
        # recorded speed, which lets a recording stand in for a live camera.
        self.realtime = realtime and kind == "file"
        self.cleanup_path = cleanup_path
        self.max_reconnects = max_reconnects
        self.reconnect_delay_s = reconnect_delay_s
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._state = "starting"
        self._error: str | None = None
        self._started_at = now_iso()
        self._finished_at: str | None = None
        self._frames_processed = 0
        self._position_s: float | None = None
        self._duration_s: float | None = None
        self._last_frame_at: str | None = None
        self._last_update_monotonic: float | None = None
        self._processing_s: deque[float] = deque(maxlen=TIMING_WINDOW)
        self._update_intervals_s: deque[float] = deque(maxlen=TIMING_WINDOW)

    # Lifecycle -----------------------------------------------------------

    def start(self) -> "SourceWorker":
        self._thread = threading.Thread(target=self._run, name=f"slotsight-source-{self.kind}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict[str, Any]:
        with self._lock:
            progress = None
            if self._duration_s and self._position_s is not None:
                progress = round(min(1.0, self._position_s / self._duration_s), 3)
            if self._state == "finished" and self.kind == "file":
                progress = 1.0
            return {
                "kind": self.kind,
                "name": self.name,
                "state": self._state,
                "error": self._error,
                "realtime": self.realtime,
                "interval_s": self.interval_s,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "frames_processed": self._frames_processed,
                "position_s": None if self._position_s is None else round(self._position_s, 2),
                "duration_s": None if self._duration_s is None else round(self._duration_s, 2),
                "progress": progress,
                "last_frame_at": self._last_frame_at,
                "processing_s": summarize(list(self._processing_s)),
                "update_interval_s": summarize(list(self._update_intervals_s)),
            }

    def timings(self) -> dict[str, list[float]]:
        """Raw per-frame timings (most recent TIMING_WINDOW frames), for measurement scripts."""
        with self._lock:
            return {"processing_s": list(self._processing_s), "update_interval_s": list(self._update_intervals_s)}

    # Worker thread -------------------------------------------------------

    def _set(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, f"_{key}", value)

    def _open(self):
        capture = cv2.VideoCapture(self.target)
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"Could not open {self.name}.")
        if self.kind != "file":
            # Keep the driver's queue short so each sample is a recent frame.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        try:
            self._capture = self._open()
            self._set(state="running")
            if self.kind == "file":
                self._run_file()
            else:
                self._run_live()
            self._set(state="stopped" if self._stop.is_set() else "finished")
        except Exception as error:  # Reported through status(); the thread must not die silently.
            self._set(state="error", error=str(error) or error.__class__.__name__)
        finally:
            if self._capture is not None:
                self._capture.release()
            self._set(finished_at=now_iso())
            if self.cleanup_path:
                Path(self.cleanup_path).unlink(missing_ok=True)

    def _handle(self, frame, offset_s: float | None) -> None:
        started = time.monotonic()
        self.on_frame(frame, offset_s)
        finished = time.monotonic()
        with self._lock:
            self._processing_s.append(finished - started)
            if self._last_update_monotonic is not None:
                self._update_intervals_s.append(finished - self._last_update_monotonic)
            self._last_update_monotonic = finished
            self._frames_processed += 1
            self._last_frame_at = now_iso()
            if offset_s is not None:
                self._position_s = offset_s

    def _run_file(self) -> None:
        capture = self._capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 1000:
            fps = FALLBACK_FPS
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count and frame_count > 0:
            self._set(duration_s=frame_count / fps)
        step = max(1, round(fps * self.interval_s))
        playback_start = time.monotonic()
        index = 0
        while not self._stop.is_set():
            if index % step == 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                offset_s = index / fps
                if self.realtime:
                    delay = playback_start + offset_s - time.monotonic()
                    if delay > 0 and self._stop.wait(delay):
                        break
                self._handle(frame, offset_s)
            elif not capture.grab():
                break
            index += 1
        if self._frames_processed == 0 and not self._stop.is_set():
            raise SourceError(f"No readable frames in {self.name}.")

    def _run_live(self) -> None:
        next_due = 0.0
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok or frame is None:
                failures += 1
                if failures > self.max_reconnects:
                    raise SourceError(f"Lost {self.name} after {self.max_reconnects} reconnect attempts.")
                self._capture.release()
                if self._stop.wait(self.reconnect_delay_s):
                    break
                try:
                    self._capture = self._open()
                except SourceError:
                    continue
                continue
            failures = 0
            # Read every frame so the stream buffer never goes stale; process one per interval.
            now = time.monotonic()
            if now >= next_due:
                next_due = now + self.interval_s
                self._handle(frame, None)
