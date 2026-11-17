"""Safeguard against cars the detector misses: does a "vacant" slot still look empty?

The detector-based classifier can only see cars it detects; a missed car leaves its
slot reading confidently free, the costly error behind R-07. This check learns what
each slot looks like when empty and turns a "vacant" reading into "unknown" when the
slot looks too different from that (decided 2026-11-17, docs/PLAN.md). It never
turns anything into "occupied".

How it works (the training-free empty-reference method of classical parking
systems, with the reference updated only while the slot reads vacant):

- Each frame is converted to CLAHE-equalised greyscale. A slot's bounding-box crop
  is resized to 48x48 and normalised to zero mean and unit spread, which softens
  brightness and contrast changes.
- The slot's reference is a running average of those crops, taken from the
  detector's vacant readings whose edge density is low for that slot: at most
  1.5x the slot's own baseline (the 25th percentile of its edge density over its
  last 30 vacant readings) + 0.01. Lighting changes add few edges and are learned;
  cars usually add many and are not. (An earlier rule that only absorbed readings
  already close to the reference went stale when frames were far apart in time:
  on PKLot it turned 56 % of truly empty slots unknown.)
- difference = mean absolute difference between the crop and the reference. Above
  the threshold (setting `empty_check_threshold`, default 0.8, 0 = off), a vacant
  reading becomes unknown, with the reason "doesn't look empty".

Measured on PKLot (public data, proxy only; 603 frames, frames hours apart): at
0.8 the check cut occupied slots shown as free from 9,167 to 3,604 (-61 %) while
4.0 % of truly empty slots became unknown (628 of 15,646). At 0.6: -82 % and 13.1 %.

During warm-up (fewer than MIN_HISTORY vacant readings, so no edge baseline yet), a
reading the check itself flags is not learned, so a car that appears in the first
few frames is not absorbed.

Limits: the first vacant reading of a slot seeds its reference, so a car the
detector misses from the very start is learned as "empty" until it leaves. A
missed car with fewer edges than the empty slot (a plain roof over textured
pavement such as cobbles) can also be learned as "empty"; check this on the lot's
own labeled sessions (`evaluate.py` reports what the check caught and cost). A moved
camera invalidates every reference (the camera-shift check warns about that), and
references are tied to each slot's polygon, so editing the layout starts afresh.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

import cv2
import numpy as np

PATCH = (48, 48)
DEFAULT_THRESHOLD = 0.8
REASON = "doesn't look empty"  # the `reason` this check gives (other steps give their own)
UPDATE_RATE = 0.2
HISTORY = 30  # edge densities kept per slot for its baseline
BASELINE_PERCENTILE = 25
MIN_HISTORY = 5
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def prepare(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CLAHE-equalised greyscale of a BGR frame and its Canny edges (shared by all slots)."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    grey = _CLAHE.apply(grey)
    return grey, cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 60, 150)


def slot_polygon(grey: np.ndarray, points: list[list[float]]) -> np.ndarray:
    height, width = grey.shape[:2]
    return (np.array(points, dtype=np.float32) * [width / 100, height / 100]).astype(np.int32)


def edge_density(edges: np.ndarray, polygon: np.ndarray) -> float:
    """Share of edge pixels inside the slot."""
    mask = np.zeros(edges.shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    inside = mask > 0
    return float(edges[inside].mean() / 255) if inside.any() else 0.0


def slot_patch(grey: np.ndarray, points: list[list[float]]) -> np.ndarray | None:
    """The slot's normalised 48x48 appearance, or None if it falls outside the frame."""
    polygon = slot_polygon(grey, points)
    x, y, w, h = cv2.boundingRect(polygon)
    x, y = max(0, x), max(0, y)
    crop = grey[y : y + h, x : x + w]
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    patch = cv2.resize(crop.astype(np.float32), PATCH)
    return (patch - patch.mean()) / (patch.std() + 1e-6)


def difference(patch: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(patch - reference).mean())


class EmptyReference:
    """Per-slot learned empty appearance for one camera view. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._references: dict[tuple, np.ndarray] = {}
        self._densities: dict[tuple, deque[float]] = {}

    def reset(self) -> None:
        with self._lock:
            self._references.clear()
            self._densities.clear()

    @staticmethod
    def _key(slot: dict[str, Any]) -> tuple:
        # Tied to the polygon: editing a slot's corners starts a fresh reference.
        return (str(slot["id"]), tuple(tuple(round(float(v), 3) for v in point) for point in slot["points"]))

    def check(self, frame: np.ndarray, slots: list[dict[str, Any]], results: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
        """Return results where vacant slots that don't look empty are unknown.

        `results` are single-frame statuses after the review threshold, in slot order.
        Adds `empty_diff` (the difference, or None) to every result and `reason` to
        the ones changed. Learns from the detector's vacant readings with low edge density.
        """
        if threshold <= 0:
            return [{**result, "empty_diff": None} for result in results]
        grey, edges = prepare(frame)
        checked = []
        with self._lock:
            for slot, result in zip(slots, results):
                result = {**result, "empty_diff": None}
                if result["status"] != "vacant":
                    checked.append(result)
                    continue
                patch = slot_patch(grey, slot["points"])
                if patch is None:
                    checked.append(result)
                    continue
                key = self._key(slot)
                reference = self._references.get(key)
                if reference is not None:
                    diff = difference(patch, reference)
                    result["empty_diff"] = round(diff, 3)
                    if diff > threshold:
                        result.update(status="unknown", reason=REASON)
                # Learn from the detector's vacant reading if its edges look like an empty slot.
                density = edge_density(edges, slot_polygon(grey, slot["points"]))
                history = self._densities.setdefault(key, deque(maxlen=HISTORY))
                baseline = float(np.percentile(history, BASELINE_PERCENTILE)) if len(history) >= MIN_HISTORY else None
                history.append(density)
                if reference is None:
                    self._references[key] = patch  # the first vacant reading seeds the reference
                elif baseline is None:
                    # Warm-up (no edge baseline yet): learn only readings that look empty.
                    if "reason" not in result:
                        self._references[key] = (1 - UPDATE_RATE) * reference + UPDATE_RATE * patch
                elif density <= baseline * 1.5 + 0.01:
                    self._references[key] = (1 - UPDATE_RATE) * reference + UPDATE_RATE * patch
                checked.append(result)
        return checked
