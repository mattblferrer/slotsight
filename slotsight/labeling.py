"""Reading and saving hand labels for the evaluation set (eval/labels.csv).

The /label page in the dashboard uses these functions. Labels are one row per
slot per frame: frame, slot_id, true_status, condition, occluded (the format the
evaluation harness reads, decided 2026-11-17). Frames live under
data/eval_frames/<set>/<session>/ (written by eval/extract_frames.py) and are
referred to by their path relative to that folder, with forward slashes.

Saving a frame replaces all of that frame's rows and leaves every other row as it
was. The file is rewritten atomically, so a crash can't leave half a labels file.
"""

from __future__ import annotations

import csv
import io
import json
import os
import threading
from pathlib import Path
from typing import Any

LABEL_COLUMNS = ["frame", "slot_id", "true_status", "condition", "occluded"]
TRUE_STATUSES = ("occupied", "vacant")
CONDITIONS = ("daytime", "evening", "rain")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SESSION_FILE = "session.json"

_lock = threading.Lock()  # one labels-file update at a time


class LabelingError(ValueError):
    """A request that can't be applied to the labels (bad path, status or slot)."""


def resolve_frame(frames_dir: Path, frame: str) -> Path:
    """The image for a frame path, refusing anything outside frames_dir."""
    if not frame or "\\" in frame or frame.startswith("/") or ":" in frame:
        raise LabelingError("Invalid frame path.")
    path = (frames_dir / frame).resolve()
    root = frames_dir.resolve()
    if root not in path.parents or path.suffix.lower() not in IMAGE_SUFFIXES:
        raise LabelingError("Invalid frame path.")
    if not path.is_file():
        raise LabelingError(f"Frame {frame} not found.")
    return path


def read_rows(labels_path: Path) -> list[dict[str, str]]:
    if not labels_path.exists():
        return []
    with labels_path.open(newline="", encoding="utf-8-sig") as file:
        return [dict(row) for row in csv.DictReader(file)]


def session_info(frames_dir: Path, frame: str) -> dict[str, Any]:
    """The session.json written by extract_frames.py next to a frame, if any."""
    path = (frames_dir / frame).parent / SESSION_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def list_frames(frames_dir: Path, labels_path: Path, slot_ids: list[str]) -> list[dict[str, Any]]:
    """Every frame image with how many of the current slots it has labels for."""
    labeled: dict[str, set[str]] = {}
    conditions: dict[str, str] = {}
    for row in read_rows(labels_path):
        labeled.setdefault(row.get("frame", ""), set()).add(row.get("slot_id", ""))
        conditions[row.get("frame", "")] = row.get("condition", "")
    frames = []
    if frames_dir.exists():
        for path in sorted(frames_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            name = path.relative_to(frames_dir).as_posix()
            done = len(labeled.get(name, set()) & set(slot_ids))
            frames.append(
                {
                    "frame": name,
                    "labeled": done,
                    "complete": bool(slot_ids) and done == len(slot_ids),
                    "condition": conditions.get(name) or session_info(frames_dir, name).get("condition") or "",
                }
            )
    return frames


def labels_for(labels_path: Path, frame: str) -> list[dict[str, str]]:
    return [row for row in read_rows(labels_path) if row.get("frame") == frame]


def save_frame(labels_path: Path, frame: str, condition: str, slots: list[dict[str, Any]], slot_ids: list[str]) -> int:
    """Replace the labels of one frame. `slots` holds {slot_id, true_status, occluded};
    slots left unset are simply not in the list. Returns the number of rows saved."""
    if condition not in CONDITIONS:
        raise LabelingError(f"Condition must be one of {', '.join(CONDITIONS)}.")
    known = set(slot_ids)
    new_rows, seen = [], set()
    for slot in slots:
        slot_id = str(slot.get("slot_id", ""))
        status = slot.get("true_status")
        if slot_id not in known:
            raise LabelingError(f"Slot {slot_id!r} is not in the current layout.")
        if slot_id in seen:
            raise LabelingError(f"Slot {slot_id} is labeled twice.")
        if status not in TRUE_STATUSES:
            raise LabelingError(f"Status must be occupied or vacant, not {status!r}.")
        seen.add(slot_id)
        new_rows.append({"frame": frame, "slot_id": slot_id, "true_status": status, "condition": condition, "occluded": "1" if slot.get("occluded") else "0"})
    order = {slot_id: index for index, slot_id in enumerate(slot_ids)}
    new_rows.sort(key=lambda row: order[row["slot_id"]])
    with _lock:
        rows = [row for row in read_rows(labels_path) if row.get("frame") != frame]
        rows.extend(new_rows)
        rows.sort(key=lambda row: (row.get("frame", ""), order.get(row.get("slot_id", ""), len(order))))
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=LABEL_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = labels_path.with_name(f".{labels_path.name}.{os.getpid()}.tmp")
        temporary.write_text(output.getvalue(), encoding="utf-8")
        os.replace(temporary, labels_path)
    return len(new_rows)
