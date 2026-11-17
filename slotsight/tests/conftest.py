"""Shared fixtures. Every test runs against temporary config, data and models folders,
so the user's slots.json, settings.txt, history and downloaded models are never touched.

Run from the repository root:  python -m pytest slotsight/tests
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

SLOTSIGHT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SLOTSIGHT_DIR))
sys.path.insert(0, str(SLOTSIGHT_DIR / "eval"))

import app as slotsight_app  # noqa: E402
import detectors  # noqa: E402

# Two side-by-side slots covering the whole frame: easy to reason about in pixels.
TWO_SLOTS = [
    {"id": "L", "points": [[0, 0], [50, 0], [50, 100], [0, 100]]},
    {"id": "R", "points": [[50, 0], [100, 0], [100, 100], [50, 100]]},
]
FRAME_SHAPE = (480, 640, 3)  # height, width: slot L is x 0-320, slot R is x 320-640


class FakeDetector:
    """Returns preset boxes, or raises, and counts calls."""

    def __init__(self) -> None:
        self.boxes: list[tuple[float, float, float, float, float]] = []
        self.error: Exception | None = None
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.boxes)


@pytest.fixture
def app(tmp_path, monkeypatch):
    """The app module with every path pointed at tmp_path and no model state."""
    monkeypatch.delenv("SLOTSIGHT_MODEL", raising=False)
    monkeypatch.setattr(slotsight_app, "CONFIG_PATH", tmp_path / "config" / "slots.json")
    monkeypatch.setattr(slotsight_app, "SETTINGS_PATH", tmp_path / "config" / "settings.txt")
    monkeypatch.setattr(slotsight_app, "HISTORY_PATH", tmp_path / "data" / "occupancy_history.csv")
    monkeypatch.setattr(slotsight_app, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(slotsight_app, "models", detectors.ModelManager(tmp_path / "models"))
    for name in (
        "latest_frame_snapshot",
        "latest_frame",
        "latest_frame_detections",
        "latest_frame_time_s",
        "latest_frame_captured_at",
        "latest_frame_processed_at",
        "latest_frame_model",
        "active_source",
    ):
        monkeypatch.setattr(slotsight_app, name, None)
    yield slotsight_app
    worker = slotsight_app.active_source
    if worker is not None:
        worker.stop()


@pytest.fixture
def fake_detector(app, monkeypatch):
    """Make the configured model 'ready' and answer with a FakeDetector."""
    detector = FakeDetector()
    ready = {"id": "test/fake", "name": "Fake detector", "state": "ready", "message": "", "progress": None}
    monkeypatch.setattr(app.models, "status", lambda model_id: {**ready, "id": model_id})
    monkeypatch.setattr(app.models, "require", lambda model_id: detector)
    return detector


@pytest.fixture
def two_slots(app):
    app.save_slots(TWO_SLOTS)
    return TWO_SLOTS


@pytest.fixture
def client(app):
    return app.app.test_client()


def blank_frame(value: int = 128) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def png_upload(frame: np.ndarray, name: str = "frame.png") -> dict:
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return {"source": (io.BytesIO(buffer.tobytes()), name)}


def history_rows(app) -> list[dict[str, str]]:
    import csv

    if not app.HISTORY_PATH.exists():
        return []
    with app.HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def wait_until(predicate, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False
