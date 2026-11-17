"""Per-slot occupancy classifier: a tiny CNN that looks at each slot's own image.

Decided 2026-11-17 (docs/PLAN.md): the classifier is the primary decision and the
vehicle detector a safety vote: a slot reads "vacant" only if both agree.

Why: on PKLot (public data, proxy evidence) a COCO vehicle detector plus polygon
overlap reached about 50 % per-slot accuracy, missing densely parked and distant
cars. This classifier, trained on two PKLot views and tested on an unseen third,
reached 97-99 %, at about 0.16 ms per slot on 4 CPU threads. This is the
patch-classification approach of the PKLot/CNRPark literature (mAlexNet,
MobileNetV3 variants).

- Each slot's 4-corner polygon is warped to an upright 48x48 colour patch (a slot
  with more corners uses its minimum-area rectangle).
- The network (about 61k parameters) outputs P(vacant), P(occupied).
- Weights: models/slot_classifier.pt (fine-tuned on this lot by
  eval/train_classifier.py) if present, otherwise the committed base weights
  models/slot_classifier_base.pt (trained on a PKLot subset; CC BY 4.0, see
  models/README.md). Neither present, or torch missing: the app falls back to
  detector-only classification.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("slotsight.classifier")

PATCH = 48
BASE_WEIGHTS = "slot_classifier_base.pt"
LOCAL_WEIGHTS = "slot_classifier.pt"


def canonical_quad(quad: np.ndarray) -> np.ndarray:
    """Four corners in a fixed order whatever order they were entered in: clockwise
    on screen, starting from the corner nearest the top-left. Anything that isn't a
    convex quadrilateral uses its minimum-area rectangle."""
    hull = cv2.convexHull(quad.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        hull = cv2.boxPoints(cv2.minAreaRect(quad.astype(np.float32))).astype(np.float32)
    centre = hull.mean(axis=0)
    ordered = hull[np.argsort(np.arctan2(hull[:, 1] - centre[1], hull[:, 0] - centre[0]))]  # clockwise (y points down)
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def warp_slot(frame: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """The slot's image, straightened to an upright PATCH x PATCH colour patch."""
    height, width = frame.shape[:2]
    quad = canonical_quad(np.array(points, dtype=np.float32) * np.float32([width / 100, height / 100]))
    target = np.float32([[0, 0], [PATCH - 1, 0], [PATCH - 1, PATCH - 1], [0, PATCH - 1]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(quad, target), (PATCH, PATCH))


def build_network():
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2),
    )


def patches_tensor(patches: np.ndarray):
    """uint8 BGR patches (n, PATCH, PATCH, 3) -> float tensor (n, 3, PATCH, PATCH) in 0-1."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()


def weights_info(path: Path) -> dict[str, Any]:
    """Metadata saved beside the weights (what it was trained on), if any."""
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


class SlotClassifier:
    """Loads the weights once and classifies every slot of a frame. Thread-safe."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = threading.Lock()
        self._net = None
        self._loaded: tuple[Path, int] | None = None  # (path, mtime_ns) of the loaded weights
        self._error: str | None = None
        self._failed: dict[Path, tuple[int, str]] = {}  # weights that failed to load: (mtime_ns, error)

    def _candidates(self) -> list[Path]:
        return [self.models_dir / name for name in (LOCAL_WEIGHTS, BASE_WEIGHTS) if (self.models_dir / name).is_file()]

    def weights_path(self) -> Path | None:
        """The weights in use: fine-tuned if they load, otherwise the base weights."""
        for path in self._candidates():
            failed = self._failed.get(path)
            if failed is None or failed[0] != path.stat().st_mtime_ns:
                return path
        return None

    def status(self) -> dict[str, Any]:
        path = self.weights_path()
        problems = "; ".join(error for _, error in self._failed.values())
        if path is None:
            if problems:
                return {"state": "error", "weights": None, "message": problems}
            return {"state": "missing", "weights": None, "message": "No slot classifier weights; using the detector alone."}
        info = weights_info(path)
        kind = "fine-tuned on this lot" if path.name == LOCAL_WEIGHTS else "base (PKLot)"
        state = "error" if self._error else "ready"
        message = self._error or f"Slot classifier: {kind}" + (f" ({problems})" if problems else "")
        return {"state": state, "weights": path.name, "kind": kind, "trained": info, "message": message}

    def _network(self):
        # Fine-tuned weights that fail to load fall back to the base weights.
        while (path := self.weights_path()) is not None:
            key = (path, path.stat().st_mtime_ns)
            if self._loaded == key and self._net is not None:
                return self._net
            try:
                import torch

                net = build_network()
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                self._net, self._loaded, self._error = net.eval(), key, None
                self._failed.pop(path, None)
                log.info("Loaded slot classifier %s", path.name)
                return self._net
            except Exception as error:  # reported through status(); try the next weights
                self._failed[path] = (key[1], f"could not load {path.name}: {error}")
                self._net, self._loaded = None, None
                log.error("Could not load %s: %s", path.name, error)
        return None

    def predict(self, frame: np.ndarray, slots: list[dict[str, Any]]) -> list[tuple[float, float]] | None:
        """(P(vacant), P(occupied)) per slot, or None when no classifier can be used."""
        if not slots:
            return []
        with self._lock:
            net = self._network()
            if net is None:
                return None
            import torch

            try:
                patches = np.stack([warp_slot(frame, slot["points"]) for slot in slots])
                with torch.inference_mode():
                    probabilities = torch.softmax(net(patches_tensor(patches)), dim=1).numpy()
            except Exception as error:  # never crash a frame: fall back to the detector and say so
                self._error = f"Slot classifier failed: {error}"
                log.error(self._error)
                return None
            self._error = None
        return [(float(p_vacant), float(p_occupied)) for p_vacant, p_occupied in probabilities]


def combine(detector_results: list[dict[str, Any]], probabilities: list[tuple[float, float]] | None, review_threshold: float) -> list[dict[str, Any]]:
    """The classifier decides; the detector is a safety vote for "vacant".

    detector_results: classify_frame output (raw statuses and confidences).
    Returns results with status after the review threshold:
    - classifier unsure (top probability below review_threshold) -> unknown
    - classifier says occupied                                   -> occupied
    - classifier says vacant, detector confidently vacant too    -> vacant
    - classifier says vacant, detector disagrees or is unsure    -> unknown
    Without probabilities (no classifier), the detector's own results are used.
    """
    if probabilities is None:
        return [{**result, "status": result["status"] if result["confidence"] >= review_threshold else "unknown", "method": "detector"} for result in detector_results]
    combined = []
    for result, (p_vacant, p_occupied) in zip(detector_results, probabilities):
        # How sure the detector is that the slot is empty.
        detector_vacant_confidence = result["confidence"] if result["status"] == "vacant" else 1.0 - result["confidence"]
        classifier_confidence = max(p_vacant, p_occupied)
        # `confidence` always describes the status shown: "vacant" needs both models,
        # so it is the weaker of the two, and any "unknown" stays below the threshold.
        if classifier_confidence < review_threshold:
            status, reason, confidence = "unknown", "classifier unsure", classifier_confidence
        elif p_occupied >= p_vacant:
            status, reason, confidence = "occupied", None, p_occupied
        elif detector_vacant_confidence >= review_threshold:
            status, reason, confidence = "vacant", None, min(p_vacant, detector_vacant_confidence)
        else:
            status, reason, confidence = "unknown", "detector sees a vehicle", min(p_vacant, detector_vacant_confidence)
        entry = {
            "id": result["id"],
            "status": status,
            "confidence": round(confidence, 2),
            "method": "classifier",
            "p_occupied": round(p_occupied, 3),
            "detector_status": result["status"],
        }
        if reason:
            entry["reason"] = reason
        combined.append(entry)
    return combined
