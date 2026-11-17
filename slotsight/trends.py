"""Occupancy trends from the history: free slots over time and busy hours of the day.

Works on the rows produced by app.history_rows() (header first), so it never
touches the history file itself. Each processed frame is one group of rows that
share timestamp, source and frame offset; its counts use the logged `status`,
which is what the dashboard showed (smoothed for videos and live sources).

Ranges end at the most recent reading, not at "now", so recorded footage from
another day still charts.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

RANGES = {
    "24h": (timedelta(hours=24), timedelta(minutes=5)),  # span, bucket size
    "7d": (timedelta(days=7), timedelta(hours=1)),
}
PROFILE_SPAN = timedelta(days=30)  # how much history the hour-of-day profile uses
PEAK_HOURS = 3


def _frames(rows: list[list[str]]) -> list[dict[str, Any]]:
    """One record per processed frame: time and its slot counts."""
    if not rows:
        return []
    header = rows[0]
    index = {name: header.index(name) for name in ("timestamp", "status", "source", "frame_time_s")}
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"vacant": 0, "occupied": 0, "unknown": 0})
    for row in rows[1:]:
        key = (row[index["timestamp"]], row[index["source"]], row[index["frame_time_s"]])
        status = row[index["status"]]
        groups[key][status if status in ("vacant", "occupied") else "unknown"] += 1
    frames = []
    for (timestamp, _, _), counts in groups.items():
        try:
            at = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        total = sum(counts.values())
        frames.append({"at": at, "free": counts["vacant"], "occupied": counts["occupied"], "unknown": counts["unknown"], "total": total})
    frames.sort(key=lambda frame: frame["at"])
    return frames


def summarize(rows: list[list[str]], range_key: str = "24h", local_tz=None) -> dict[str, Any]:
    """Bucketed series for the range and an hour-of-day profile (local time)."""
    span, bucket = RANGES[range_key]
    frames = _frames(rows)
    if not frames:
        return {"range": range_key, "start": None, "end": None, "bucket_s": int(bucket.total_seconds()), "series": [], "hours": [], "peak_hours": [], "frames": 0}
    end = frames[-1]["at"]
    start = end - span
    series_frames = [frame for frame in frames if frame["at"] > start]

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in series_frames:
        buckets[int((frame["at"] - start) / bucket)].append(frame)
    count = int(span / bucket)
    series = []
    for position in range(count):
        members = buckets.get(position, [])
        point: dict[str, Any] = {"t": (start + bucket * position).isoformat(timespec="seconds")}
        if members:
            point.update(
                free=round(sum(m["free"] for m in members) / len(members), 2),
                free_min=min(m["free"] for m in members),
                unknown=round(sum(m["unknown"] for m in members) / len(members), 2),
                total=max(m["total"] for m in members),
                frames=len(members),
            )
        else:
            point.update(free=None, free_min=None, unknown=None, total=None, frames=0)
        series.append(point)

    hourly: dict[int, list[float]] = defaultdict(list)
    for frame in frames:
        if frame["at"] > end - PROFILE_SPAN and frame["total"]:
            hourly[frame["at"].astimezone(local_tz).hour].append(frame["occupied"] / frame["total"])
    hours = [
        {"hour": hour, "occupancy": round(sum(values) / len(values), 3) if values else None, "frames": len(values)}
        for hour, values in ((hour, hourly.get(hour, [])) for hour in range(24))
    ]
    measured = [hour for hour in hours if hour["occupancy"] is not None]
    peaks = sorted(measured, key=lambda hour: hour["occupancy"], reverse=True)[:PEAK_HOURS]
    return {
        "range": range_key,
        "start": start.isoformat(timespec="seconds"),
        "end": end.isoformat(timespec="seconds"),
        "bucket_s": int(bucket.total_seconds()),
        "series": series,
        "hours": hours,
        "peak_hours": [hour["hour"] for hour in peaks],
        "frames": len(series_frames),
        "profile_frames": sum(hour["frames"] for hour in hours),
    }
