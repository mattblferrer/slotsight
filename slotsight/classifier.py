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
INFERENCE_THREADS = 4


def warp_slot(frame: np.ndarray, points: list[list[float]]) -> np.ndarray:
    """The slot's image, straightened to an upright PATCH x PATCH colour patch."""
    height, width = frame.shape[:2]
    quad = np.array(points, dtype=np.float32) * np.float32([width / 100, height / 100])
    if len(quad) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(quad)).astype(np.float32)
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

    def weights_path(self) -> Path | None:
        for name in (LOCAL_WEIGHTS, BASE_WEIGHTS):
            path = self.models_dir / name
            if path.is_file():
                return path
        return None

    def status(self) -> dict[str, Any]:
        path = self.weights_path()
        if path is None:
            return {"state": "missing", "weights": None, "message": "No slot classifier weights; using the detector alone."}
        info = weights_info(path)
        kind = "fine-tuned on this lot" if path.name == LOCAL_WEIGHTS else "base (PKLot)"
        state = "error" if self._error else "ready"
        return {"state": state, "weights": path.name, "kind": kind, "trained": info, "message": self._error or f"Slot classifier: {kind}"}

    def _network(self):
        path = self.weights_path()
        if path is None:
            return None
        key = (path, path.stat().st_mtime_ns)
        if self._loaded != key:
            try:
                import torch

                net = build_network()
                net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                self._net, self._loaded, self._error = net.eval(), key, None
                log.info("Loaded slot classifier %s", path.name)
            except Exception as error:  # reported through status(); the app falls back to the detector
                self._net, self._loaded, self._error = None, key, f"Could not load {path.name}: {error}"
                log.error(self._error)
        return self._net

    def predict(self, frame: np.ndarray, slots: list[dict[str, Any]]) -> list[tuple[float, float]] | None:
        """(P(vacant), P(occupied)) per slot, or None when no classifier can be used."""
        if not slots:
            return []
        with self._lock:
            net = self._network()
            if net is None:
                return None
            import torch

            patches = np.stack([warp_slot(frame, slot["points"]) for slot in slots])
            previous = torch.get_num_threads()
            torch.set_num_threads(INFERENCE_THREADS)
            try:
                with torch.inference_mode():
                    probabilities = torch.softmax(net(patches_tensor(patches)), dim=1).numpy()
            finally:
                torch.set_num_threads(previous)
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
        detector_vacant = result["status"] == "vacant" and result["confidence"] >= review_threshold
        classifier_status = "occupied" if p_occupied >= p_vacant else "vacant"
        confidence = max(p_vacant, p_occupied)
        if confidence < review_threshold:
            status, reason = "unknown", "classifier unsure"
        elif classifier_status == "occupied":
            status, reason = "occupied", None
        elif detector_vacant:
            status, reason = "vacant", None
        else:
            status, reason = "unknown", "detector sees a vehicle"
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
