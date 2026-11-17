"""Warn when the camera seems to have moved, so slot polygons no longer line up.

Decided 2026-11-17 (docs/PLAN.md): warn only, never move the polygons.

A reference frame is kept beside the slot config (config/reference_frame.jpg). It is
saved when the layout is saved, or taken from the first processed frame if there is
none. Every CHECK_EVERY-th processed frame is matched against it: ORB features,
then a RANSAC homography (the approach of OpenCV's feature-matching and planar
tracking tutorials). The slot corners are mapped through that homography, and their
median displacement is the estimated shift, in percent of the frame width.

- shift > SHIFT_WARN_PCT for WARN_AFTER checks in a row  -> "moved" (warning)
- shift > SHIFT_WARN_PCT, fewer checks in a row          -> "suspect" (no warning yet)
- too few matches or inliers (night, fog, a covered lens) -> "unverified", not an alarm
- otherwise                                              -> "ok"

Lighting changes are softened by matching on CLAHE-equalised greyscale. Vehicles that
come and go are outliers to RANSAC as long as the fixed scene dominates the view.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("slotsight.camera")

CHECK_EVERY = 10  # processed frames between checks
WORK_WIDTH = 960  # frames are downscaled to this width before matching
SHIFT_WARN_PCT = 2.0
WARN_AFTER = 3  # consecutive checks above the threshold before warning
MIN_GOOD_MATCHES = 30
MIN_INLIERS = 20
RATIO_TEST = 0.75


def _prepare(frame: np.ndarray) -> tuple[np.ndarray, float]:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = WORK_WIDTH / grey.shape[1]
    if scale < 1:
        grey = cv2.resize(grey, (WORK_WIDTH, int(round(grey.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey), scale


def estimate_shift(reference: np.ndarray, frame: np.ndarray, slots: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare `frame` with `reference`: {"state": "ok"|"shifted"|"unverified", "shift_pct", "inliers"}."""
    ref_grey, _ = _prepare(reference)
    cur_grey, _ = _prepare(frame)
    if ref_grey.shape != cur_grey.shape:
        # Another camera or resolution, not a moved camera: can't compare.
        return {"state": "unverified", "shift_pct": None, "inliers": 0, "reason": "frame size differs from the reference"}
    orb = cv2.ORB_create(nfeatures=2000)
    ref_keys, ref_desc = orb.detectAndCompute(ref_grey, None)
    cur_keys, cur_desc = orb.detectAndCompute(cur_grey, None)
    if ref_desc is None or cur_desc is None or len(ref_keys) < MIN_GOOD_MATCHES or len(cur_keys) < MIN_GOOD_MATCHES:
        return {"state": "unverified", "shift_pct": None, "inliers": 0, "reason": "too few features to compare"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ref_desc, cur_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < RATIO_TEST * pair[1].distance]
    if len(good) < MIN_GOOD_MATCHES:
        return {"state": "unverified", "shift_pct": None, "inliers": len(good), "reason": "too few matching features"}
    source = np.float32([ref_keys[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target = np.float32([cur_keys[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inliers = int(mask.sum()) if mask is not None else 0
    if homography is None or inliers < MIN_INLIERS:
        return {"state": "unverified", "shift_pct": None, "inliers": inliers, "reason": "no consistent match"}
    height, width = ref_grey.shape
    corners = np.float32([[x * width / 100, y * height / 100] for slot in slots for x, y in slot["points"]])
    if not len(corners):
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    shift_pct = float(np.median(np.linalg.norm(moved - corners, axis=1)) / width * 100)
    state = "shifted" if shift_pct > SHIFT_WARN_PCT else "ok"
    return {"state": state, "shift_pct": round(shift_pct, 2), "inliers": inliers, "reason": None}


class CameraMonitor:
    """Holds the reference frame and the latest check result. Safe to call from any thread."""

    def __init__(self, reference_path: Path) -> None:
        self.reference_path = reference_path
        self._lock = threading.Lock()  # guards the fields below; checks run outside it
        self._reference: np.ndarray | None = None
        self._reference_loaded = False
        self._frames = 0
        self._streak = 0
        self._status: dict[str, Any] = {"state": "no_reference"}
        self._origin: str | None = None  # "layout" (saved with the layout) or "auto" (first frame)

    @property
    def _origin_path(self) -> Path:
        return self.reference_path.with_suffix(".json")

    def _load_reference(self) -> np.ndarray | None:
        if not self._reference_loaded:
            self._reference_loaded = True
            if self.reference_path.exists():
                data = np.fromfile(str(self.reference_path), dtype=np.uint8)
                self._reference = cv2.imdecode(data, cv2.IMREAD_COLOR)
                try:
                    self._origin = json.loads(self._origin_path.read_text(encoding="utf-8")).get("origin")
                except (OSError, ValueError):
                    self._origin = "layout"  # older references were all saved with the layout
        return self._reference

    def forget_automatic_reference(self) -> None:
        """A new source starts: drop a reference that was only taken from another
        source's first frame (one saved with the layout is kept)."""
        with self._lock:
            self._load_reference()
            if self._origin != "auto":
                return
            self._reference, self._origin = None, None
            self._frames, self._streak = 0, 0
            self._status = {"state": "no_reference"}
        self.reference_path.unlink(missing_ok=True)
        self._origin_path.unlink(missing_ok=True)

    def set_reference(self, frame: np.ndarray, origin: str = "layout") -> None:
        """Save `frame` as the reference and clear any warning. `origin` is "layout"
        (the frame the layout was saved on) or "auto" (a source's first frame)."""
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return
        temporary = self.reference_path.with_name(f".{self.reference_path.name}.tmp")
        encoded.tofile(str(temporary))
        temporary.replace(self.reference_path)
        self._origin_path.write_text(json.dumps({"origin": origin, "set_at": _now()}), encoding="utf-8")
        with self._lock:
            self._reference, self._reference_loaded, self._origin = frame.copy(), True, origin
            self._frames, self._streak = 0, 0
            self._status = {"state": "ok", "shift_pct": 0.0, "reference_at": _now(), "checked_at": _now()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def observe(self, frame: np.ndarray, slots: list[dict[str, Any]]) -> None:
        """Called for every processed frame; checks every CHECK_EVERY-th one.

        Runs the comparison outside the lock (it takes tens of milliseconds) and never
        raises: a failed check is reported as "unverified".
        """
        with self._lock:
            reference = self._load_reference()
            if reference is None:
                pass  # no reference yet: this frame becomes it (below, outside the lock)
            else:
                self._frames += 1
                if (self._frames - 1) % CHECK_EVERY:  # check frames 1, 1 + N, 1 + 2N, ...
                    return
        if reference is None:
            self.set_reference(frame, origin="auto")
            return
        started = time.perf_counter()
        try:
            result = estimate_shift(reference, frame, slots)
        except cv2.error as error:
            result = {"state": "unverified", "shift_pct": None, "inliers": 0, "reason": str(error)}
        result["check_s"] = round(time.perf_counter() - started, 3)
        with self._lock:
            self._streak = self._streak + 1 if result["state"] == "shifted" else 0
            # "suspect": shifted in fewer than WARN_AFTER checks in a row (no warning yet).
            state = "moved" if self._streak >= WARN_AFTER else ("suspect" if result["state"] == "shifted" else result["state"])
            reference_at = self._status.get("reference_at")
            self._status = {**result, "state": state, "streak": self._streak, "checked_at": _now(), "reference_at": reference_at}
        if state == "moved" and self._streak == WARN_AFTER:
            log.warning("Camera may have moved: slots appear about %.1f%% of the frame width off.", result["shift_pct"] or 0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
