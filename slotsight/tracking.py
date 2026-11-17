"""Smoothing slot statuses over consecutive frames, and when each status began.

Safe-side hysteresis (decided 2026-11-17, see docs/PLAN.md): a wrong "free" reading
is the costly error (R-07), so the shown status moves towards caution quickly and
towards "vacant" slowly. Over a window of the last few single-frame (raw) statuses:

- the latest reading is "unknown"            -> unknown at once
- the last 2 readings are "occupied"         -> occupied
- the last 3 readings are "vacant"           -> vacant
- shown "vacant" but 2 of the last 3 aren't  -> unknown (the slot keeps flipping)
- otherwise                                  -> keep the shown status

So a car driving past a free slot (one occupied frame) doesn't flicker it, a car
parking shows within 2 frames (about 2-4 s at 2 s sampling), and a slot only
reads free after 3 agreeing frames (about 4-6 s).

Only sequential sources (videos, streams, cameras) are smoothed. A single image
has no neighbours, so its raw status is shown as-is.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

WINDOW = 3
CONFIRM_OCCUPIED = 2
CONFIRM_VACANT = 3


@dataclass
class SlotTrack:
    shown: str = "unknown"
    since: datetime | None = None  # when `shown` was first shown
    # The status may have begun before `since` (it was the first one after tracking
    # began), so `since` is only a lower bound: "occupied for at least 14 min".
    since_start: bool = True
    warming: bool = True  # still on the first status: nothing has changed yet
    recent: deque[str] = field(default_factory=lambda: deque(maxlen=WINDOW))
    # State before the latest reading, so that reading can be revised (re-classified).
    before: tuple | None = None


def next_status(shown: str, recent: list[str]) -> str:
    """The status to show after the readings in `recent` (oldest first)."""
    latest = recent[-1]
    if latest == "unknown":
        return "unknown"
    if len(recent) >= CONFIRM_OCCUPIED and all(status == "occupied" for status in recent[-CONFIRM_OCCUPIED:]):
        return "occupied"
    if len(recent) >= CONFIRM_VACANT and all(status == "vacant" for status in recent[-CONFIRM_VACANT:]):
        return "vacant"
    if shown == "vacant" and sum(status != "vacant" for status in recent[-WINDOW:]) >= 2:
        return "unknown"
    return shown


class SlotTracker:
    """Per-slot smoothed status and the time it began, for one sequential source."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: dict[str, SlotTrack] = {}

    def reset(self) -> None:
        with self._lock:
            self._tracks.clear()

    def update(self, results: list[dict[str, Any]], at: datetime) -> list[dict[str, Any]]:
        """Smooth one frame's raw results (statuses after the review threshold).

        Returns new result dicts with `status` (shown), `raw_status`, `confidence`
        (of the raw reading), `changed_at` and `changed_since_start`.
        """
        return self._apply(results, at, revise=False)

    def revise(self, results: list[dict[str, Any]], at: datetime) -> list[dict[str, Any]]:
        """Replace the latest reading with a re-classification of the same frame
        (after a threshold or layout change): smoothing state as if it had been read
        this way, without counting it as a second frame."""
        return self._apply(results, at, revise=True)

    def _apply(self, results: list[dict[str, Any]], at: datetime, revise: bool) -> list[dict[str, Any]]:
        smoothed = []
        with self._lock:
            for result in results:
                track = self._tracks.setdefault(result["id"], SlotTrack())
                if revise and track.before is not None and track.recent:
                    track.shown, track.since, track.since_start, track.warming = track.before
                    track.recent.pop()
                track.before = (track.shown, track.since, track.since_start, track.warming)
                track.recent.append(result["status"])
                status = next_status(track.shown, list(track.recent))
                if track.since is None:  # first frame for this slot
                    track.shown, track.since, track.since_start = status, at, True
                elif status != track.shown:
                    # The first change ends the warm-up; that status may predate tracking.
                    track.since_start = track.warming
                    track.shown, track.since, track.warming = status, at, False
                smoothed.append(
                    {
                        **result,
                        "status": track.shown,
                        "raw_status": result["status"],
                        "changed_at": track.since.isoformat(timespec="seconds") if track.since else None,
                        "changed_since_start": track.since_start,
                    }
                )
        return smoothed


def unsmoothed(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A single image: shown as read, with no history to say when the status began."""
    return [{**result, "raw_status": result["status"], "changed_at": None, "changed_since_start": False} for result in results]
