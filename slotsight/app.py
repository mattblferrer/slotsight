from __future__ import annotations

import base64
import csv
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import detectors
from detectors import DETECTION_CONFIDENCE_FLOOR, DetectorError, ModelManager  # noqa: F401 (re-exported for scripts)
from sources import SourceWorker, parse_target

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"  # files you edit: slot polygons and settings (gitignored, created on first run)
DATA_DIR = BASE_DIR / "data"  # generated output: occupancy history
MODELS_DIR = BASE_DIR / "models"  # downloaded detection models (gitignored)
CONFIG_PATH = CONFIG_DIR / "slots.json"
SETTINGS_PATH = CONFIG_DIR / "settings.txt"
HISTORY_PATH = DATA_DIR / "occupancy_history.csv"
DEFAULT_LOT_NAME = "Lot 01"
DEFAULT_REVIEW_THRESHOLD = 0.7
DEFAULT_OVERLAP_THRESHOLD = 0.4
DEFAULT_MODEL = detectors.DEFAULT_MODEL_ID  # downloaded automatically on first run
# Seconds between sampled frames (video time for files, wall-clock time for live sources).
# With the dashboard polling every 2 s this keeps updates within R-04's 5 s.
DEFAULT_SAMPLE_INTERVAL_S = 2.0
SAMPLE_INTERVAL_RANGE_S = (0.2, 60.0)
MAX_UPLOAD_MB = 100  # images and videos uploaded from the dashboard
# Smallest slot accepted, in normalised units (percent of width x percent of height).
# A slot with no area can never overlap a vehicle and would read as confidently free (R-07).
MIN_SLOT_AREA = 1.0
MAX_SLOT_ID_LENGTH = 40
HISTORY_COLUMNS = ["timestamp", "slot_id", "status", "confidence", "source", "frame_time_s"]
# Model states in which frames can be (or are about to be) processed; anything else
# means no usable model, and only then may the dashboard show demo data.
USABLE_MODEL_STATES = ("ready", "loading")
PREPARING_MODEL_STATES = ("ready", "loading", "downloading")
# Only when no model is installed may the dashboard show demo data. A model that
# is installed but broken is shown as an error, never as demo data.
NO_MODEL_STATES = ("missing",)


class DetectionError(RuntimeError):
    """The detector could not produce a result for a frame.

    Callers must surface this as an error and never substitute demo data for it.
    """

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = status_code


app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
# Flask rejects larger request bodies with 413 before they are read (multipart
# overhead allowed on top of the file itself).
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024 + 1024 * 1024
state_lock = threading.Lock()  # guards the latest_frame_* globals and history writes
inference_lock = threading.Lock()  # one inference at a time; the model is shared across threads
source_lock = threading.Lock()  # guards active_source
active_source: SourceWorker | None = None
# Installs and loads detection models. Its status calls never block; loading happens
# on a background thread or in detect_vehicles, never while state_lock is held.
models = ModelManager(MODELS_DIR)
latest_frame_snapshot: dict[str, Any] | None = None
latest_frame = None
latest_frame_detections = None
latest_frame_source = "latest frame"
latest_frame_time_s: float | None = None
latest_frame_captured_at: datetime | None = None
latest_frame_processed_at: datetime | None = None
latest_frame_model: str | None = None

DEFAULT_SLOTS = [
    {"id": "A01", "points": [[12, 18], [27, 17], [29, 39], [13, 40]]},
    {"id": "A02", "points": [[31, 17], [46, 16], [48, 38], [32, 39]]},
    {"id": "A03", "points": [[50, 16], [65, 15], [67, 37], [51, 38]]},
    {"id": "A04", "points": [[69, 15], [84, 14], [87, 36], [70, 37]]},
    {"id": "B01", "points": [[12, 51], [27, 50], [29, 73], [13, 74]]},
    {"id": "B02", "points": [[31, 50], [46, 49], [48, 72], [32, 73]]},
    {"id": "B03", "points": [[50, 49], [65, 48], [67, 71], [51, 72]]},
    {"id": "B04", "points": [[69, 48], [84, 47], [87, 70], [70, 71]]},
    {"id": "C01", "points": [[12, 85], [27, 84], [29, 98], [13, 98]]},
    {"id": "C02", "points": [[31, 84], [46, 83], [48, 98], [32, 98]]},
    {"id": "C03", "points": [[50, 83], [65, 82], [67, 98], [51, 98]]},
    {"id": "C04", "points": [[69, 82], [84, 81], [87, 98], [70, 98]]},
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_recorded_at(value: Any) -> datetime | None:
    """Parse an optional ISO 8601 "recorded at" time into UTC; naive times are taken as local time."""
    if value is None or str(value).strip() == "":
        return None
    return datetime.fromisoformat(str(value).strip()).astimezone(timezone.utc)


def capture_time(recorded_at: datetime | None, frame_time_s: float | None) -> datetime | None:
    """Real-world capture time of a video frame: recording start plus its offset."""
    if recorded_at is None:
        return None
    return recorded_at + timedelta(seconds=frame_time_s or 0.0)


class ConfigReadError(OSError):
    """A config file exists but could not be read or parsed."""


# The last content read successfully from each config file, used when a read fails
# (a OneDrive lock, or a hand edit that left invalid JSON). Keyed by path.
_last_good_config: dict[Path, Any] = {}
_config_warnings: set[Path] = set()


def write_atomic(path: Path, text: str) -> None:
    """Replace `path` in one step, so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:  # Windows: another process (e.g. OneDrive) has the file open
            if attempt == 4:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def read_config(path: Path, expected: type) -> Any:
    """Parsed JSON from a config file, or None if it does not exist.

    Retries briefly, then raises ConfigReadError. Callers must never write over a
    file they could not read.
    """
    for attempt in range(3):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.02 * (attempt + 1))
            continue
        if isinstance(data, expected):
            _config_warnings.discard(path)
            return data
        break
    if path not in _config_warnings:  # log once per bad spell, not on every poll
        _config_warnings.add(path)
        app.logger.warning("Could not read %s; using the last good copy and leaving the file untouched.", path)
    raise ConfigReadError(path)


def load_slots() -> list[dict[str, Any]]:
    """Slot polygons from config/slots.json. On first run the file is created from
    DEFAULT_SLOTS. An unreadable file is left alone, and the last good copy (or the
    defaults) is used."""
    try:
        data = read_config(CONFIG_PATH, list)
    except ConfigReadError:
        cached = _last_good_config.get(CONFIG_PATH)
        return [dict(slot) for slot in (cached if cached is not None else DEFAULT_SLOTS)]
    if data is None:
        save_slots(DEFAULT_SLOTS)
        return [dict(slot) for slot in DEFAULT_SLOTS]
    _last_good_config[CONFIG_PATH] = data
    return data


def save_slots(slots: list[dict[str, Any]]) -> None:
    write_atomic(CONFIG_PATH, json.dumps(slots, indent=2))
    _last_good_config[CONFIG_PATH] = slots


def unit_interval(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def sample_interval(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return DEFAULT_SAMPLE_INTERVAL_S
    low, high = SAMPLE_INTERVAL_RANGE_S
    return max(low, min(high, number))


def normalize_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "lot_name": str(settings.get("lot_name") or DEFAULT_LOT_NAME).strip()[:80],
        "review_threshold": unit_interval(settings.get("review_threshold"), DEFAULT_REVIEW_THRESHOLD),
        "overlap_threshold": unit_interval(settings.get("overlap_threshold"), DEFAULT_OVERLAP_THRESHOLD),
        "sample_interval_s": sample_interval(settings.get("sample_interval_s")),
        # A model id such as "ultralytics/yolo11s". Older files had `model_path`
        # (e.g. "yolo11s.pt"), which is converted here and dropped.
        "model": detectors.normalize_model_id(str(settings.get("model") or settings.get("model_path") or DEFAULT_MODEL)),
    }


def load_settings() -> dict[str, Any]:
    """Settings from config/settings.txt, created from the defaults on first run.

    A file that can't be read right now (mid-save, locked, invalid JSON) is never
    overwritten: the last good settings are used instead.
    """
    try:
        raw = read_config(SETTINGS_PATH, dict)
    except ConfigReadError:
        cached = _last_good_config.get(SETTINGS_PATH)
        return dict(cached) if cached is not None else normalize_settings({})
    normalized = normalize_settings(raw or {})
    if raw != normalized:
        # First run, a missing key, or an older format: write the canonical file.
        write_atomic(SETTINGS_PATH, json.dumps(normalized, indent=2))
    _last_good_config[SETTINGS_PATH] = normalized
    return dict(normalized)


def save_settings(**updates: Any) -> dict[str, Any]:
    settings = load_settings()
    settings.update(updates)
    write_atomic(SETTINGS_PATH, json.dumps(settings, indent=2))
    _last_good_config[SETTINGS_PATH] = dict(settings)
    return settings


def ensure_history() -> None:
    """Create the history file, or upgrade an older one to the current columns."""
    if not HISTORY_PATH.exists():
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(HISTORY_COLUMNS)
        return
    with HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == HISTORY_COLUMNS:
            return
        rows = list(reader)
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_history(
    results: list[dict[str, Any]],
    source: str,
    frame_time_s: float | None = None,
    captured_at: datetime | None = None,
) -> None:
    """Append one row per slot.

    timestamp is when the frame was captured if known (recording start + offset for
    footage with a "recorded at" time), otherwise when it was processed.
    frame_time_s is the frame's position in a video file, if any.
    """
    ensure_history()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = captured_at.isoformat(timespec="seconds") if captured_at else now_iso()
        offset = "" if frame_time_s is None else f"{frame_time_s:.2f}"
        for result in results:
            writer.writerow([timestamp, result["id"], result["status"], result["confidence"], source, offset])


def model_overridden() -> bool:
    return bool(os.environ.get("SLOTSIGHT_MODEL", "").strip())


def configured_model_id() -> str:
    """The model to use: SLOTSIGHT_MODEL (an id or a legacy .pt path) overrides settings.txt."""
    if model_overridden():
        return detectors.normalize_model_id(os.environ["SLOTSIGHT_MODEL"])
    return load_settings()["model"]


def model_status() -> dict[str, Any]:
    """Non-blocking state of the configured model (safe to call under state_lock)."""
    return models.status(configured_model_id())


def model_problem(status: dict[str, Any]) -> str:
    """What stops frames from being processed, as a message for the user."""
    hints = {
        "downloading": "It is still downloading; try again in a moment.",
        "missing": "Install it under Configure slots > Detection model, or run: python slotsight/scripts/models.py install " + status["id"],
        "needs_package": "Its Python package is missing: " + status["message"],
    }
    return f"Detection model {status['name']} is not ready ({status['state']}). {hints.get(status['state'], status['message'])}"


def detect_vehicles(frame) -> list[tuple[float, float, float, float, float]]:
    """Return vehicle boxes as (x1, y1, x2, y2, confidence) in pixels.

    Loads the configured model if needed (blocking, so never call this while holding
    state_lock). Raises DetectionError when no model can be used or inference fails,
    so a failure can never be mistaken for an empty lot.
    """
    status = model_status()
    if status["state"] not in USABLE_MODEL_STATES:
        raise DetectionError(model_problem(status), status_code=503)
    try:
        detector = models.require(status["id"])
    except DetectorError as error:
        raise DetectionError(f"Could not load the detection model: {error}", status_code=503) from error
    try:
        with inference_lock:
            return detector.detect(frame)
    except Exception as error:
        raise DetectionError(f"Vehicle detection failed: {error}") from error


def prepare_model(download_default: bool = True) -> threading.Thread | None:
    """Get the configured model ready in the background after startup.

    The default model is downloaded on first run (R-03 needs a detector out of the
    box); other models are only downloaded when the user installs them. Errors show
    up in the dashboard's model status, never as demo data mixed into real readings.
    """
    status = model_status()
    try:
        if status["state"] == "missing" and download_default and status["id"] == DEFAULT_MODEL:
            app.logger.warning("Downloading the default detection model %s on first run...", status["name"])
            models.start(status["id"], download=True, load=True)
        elif status["state"] == "loading":
            models.start(status["id"], download=False, load=True)
        elif status["state"] not in ("ready", "downloading"):
            app.logger.warning(model_problem(status))
    except DetectorError as error:
        app.logger.warning("Model preparation skipped: %s", error)
    return None


def require_model_blocking(download_default: bool = True) -> dict[str, Any]:
    """For scripts: install the default model if missing, then load the configured model.

    Returns the final model status; raises DetectionError when no model can be used.
    """
    model_id = configured_model_id()
    status = models.status(model_id)
    if status["state"] == "missing" and download_default and model_id == DEFAULT_MODEL:
        print(f"Downloading {status['name']} (first run)...", flush=True)
        try:
            models.install(model_id)
        except DetectorError as error:
            raise DetectionError(str(error), status_code=503) from error
        status = models.status(model_id)
    if status["state"] not in USABLE_MODEL_STATES:
        raise DetectionError(model_problem(status), status_code=503)
    try:
        models.require(model_id)
    except DetectorError as error:
        raise DetectionError(str(error), status_code=503) from error
    return models.status(model_id)


def slot_polygon_pixels(slot: dict[str, Any], width: int, height: int) -> np.ndarray:
    return np.array(
        [[float(x) * width / 100, float(y) * height / 100] for x, y in slot["points"]],
        dtype=np.float32,
    )


def overlap_ratio(box: tuple[float, float, float, float], slot_hull: np.ndarray) -> float:
    """How strongly a vehicle box overlaps a slot, from 0 to 1.

    The larger of: the share of the slot covered by the box (large vehicles, small
    distant slots) and the share of the box inside the slot (small vehicles such as
    motorcycles in a full-size slot).
    """
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    slot_area = cv2.contourArea(slot_hull)
    if box_area <= 0 or slot_area <= 0:
        return 0.0
    box_polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    intersection, _ = cv2.intersectConvexConvex(box_polygon, slot_hull)
    return min(1.0, max(intersection / slot_area, intersection / box_area))


def classify_frame(frame, slots: list[dict[str, Any]], detections, overlap_threshold: float) -> list[dict[str, Any]]:
    """Classify every slot from the vehicle detections on a real frame.

    Occupied: a detection overlaps the slot by at least `overlap_threshold`; the
    confidence is the strongest such detection's score.
    Vacant: no detection reaches the threshold. Confidence starts at 1 and drops as
    weaker or partial overlaps appear, so a half-seen vehicle lowers it instead of
    reading as confidently free (R-07).
    """
    height, width = frame.shape[:2]
    results = []
    for slot in slots:
        hull = cv2.convexHull(slot_polygon_pixels(slot, width, height)).reshape(-1, 2)
        if cv2.contourArea(hull) < 1.0:
            # No area: no vehicle can ever overlap it, so it must not read as free (R-07).
            results.append({"id": slot["id"], "status": "unknown", "confidence": 0.0})
            continue
        occupied_confidence = 0.0
        vacancy_doubt = 0.0
        for *box, detection_confidence in detections:
            ratio = overlap_ratio(box, hull)
            if ratio <= 0:
                continue
            if ratio >= overlap_threshold:
                occupied_confidence = max(occupied_confidence, detection_confidence)
            else:
                vacancy_doubt = max(vacancy_doubt, detection_confidence * (ratio / overlap_threshold) ** 2)
        if occupied_confidence > 0:
            status, confidence = "occupied", occupied_confidence
        else:
            status, confidence = "vacant", 1.0 - vacancy_doubt
        results.append({"id": slot["id"], "status": status, "confidence": round(confidence, 2)})
    return results


def apply_review_threshold(results: list[dict[str, Any]], review_threshold: float) -> list[dict[str, Any]]:
    """Show any slot below `review_threshold` confidence as "unknown" rather than a guess (R-07)."""
    for result in results:
        if result["confidence"] < review_threshold:
            result["status"] = "unknown"
    return results


def demo_results(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable sample state so the dashboard is demonstrable without a model. Never used for real frames."""
    results = []
    for index, slot in enumerate(slots):
        occupied = (index * 7 + int(time.time() // 8)) % 5 in {0, 1}
        results.append({"id": slot["id"], "status": "occupied" if occupied else "vacant", "confidence": 0.86 if occupied else 0.92})
    return results


def waiting_results(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A model is loaded but no frame has been processed yet: nothing is known."""
    return [{"id": slot["id"], "status": "unknown", "confidence": 0.0} for slot in slots]


def annotate_frame(frame, slots: list[dict[str, Any]], results: list[dict[str, Any]], detections) -> str:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    colors = {"unknown": (121, 216, 246), "occupied": (125, 142, 240), "vacant": (99, 187, 121)}
    polygons = [slot_polygon_pixels(slot, width, height).astype(np.int32) for slot in slots]
    for points, result in zip(polygons, results):
        cv2.fillPoly(overlay, [points], colors[result["status"]])
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
    for points, result in zip(polygons, results):
        cv2.polylines(canvas, [points], True, colors[result["status"]], max(2, width // 500))

    encoded, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise ValueError("Could not encode annotated frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def snapshot(
    frame=None,
    source: str | None = None,
    detections=None,
    record: bool = False,
    frame_time_s: float | None = None,
    captured_at: datetime | None = None,
    processed_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the dashboard payload.

    With a frame: classify it from `detections` (detected here if not given) and,
    if `record` is set, append the reading to the history. `updated_at` is when
    the frame was processed (`processed_at`, default now), so re-classifying an
    old frame after a settings change does not make it look newer (R-06).
    Without a frame: demo data only when no model is installed; while a model
    downloads or loads, and when it is broken, every slot is "unknown" (the error is
    in `model_status`). These readings are never recorded.
    """
    slots = load_slots()
    settings = load_settings()
    current_model = model_status()
    has_model = current_model["state"] in PREPARING_MODEL_STATES
    no_model = current_model["state"] in NO_MODEL_STATES
    updated_at: str | None = None
    if frame is not None:
        if detections is None:
            detections = detect_vehicles(frame)
        results = classify_frame(frame, slots, detections, settings["overlap_threshold"])
        source = source or "latest frame"
        updated_at = (processed_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    elif has_model:
        results, source, record = waiting_results(slots), "none", False
    elif no_model:
        results, source, record = demo_results(slots), "demo", False
        updated_at = now_iso()
    else:
        results, source, record = waiting_results(slots), "model error", False

    review_threshold = settings["review_threshold"]
    apply_review_threshold(results, review_threshold)
    if record:
        append_history(results, source, frame_time_s, captured_at)
    occupied = sum(result["status"] == "occupied" for result in results)
    unknown = sum(result["status"] == "unknown" for result in results)
    for result, slot in zip(results, slots):
        result["points"] = slot["points"]
        # Every slot is re-read on every frame, so each slot's last update is the frame's.
        result["updated_at"] = updated_at
    response = {
        "slots": results,
        "total": len(results),
        "free": len(results) - occupied - unknown,
        "unknown": unknown,
        "updated_at": updated_at,
        "source": source,
        "frame_time_s": None if frame_time_s is None else round(frame_time_s, 2),
        "captured_at": captured_at.isoformat(timespec="seconds") if captured_at else None,
        "model": (latest_frame_model or current_model["name"]) if frame is not None else ("demo classifier" if no_model else current_model["name"]),
        "model_status": current_model,
        "lot_name": settings["lot_name"],
        "review_threshold": review_threshold,
        "overlap_threshold": settings["overlap_threshold"],
    }
    if frame is not None:
        response["preview_image"] = annotate_frame(frame, slots, results, detections)
    return response


def reclassify_latest_frame() -> None:
    """Re-run classification on the last uploaded frame after a settings or slot change.

    Reuses the cached detections, so no new inference runs and nothing is recorded.
    """
    global latest_frame_snapshot
    if latest_frame is not None:
        latest_frame_snapshot = snapshot(
            latest_frame,
            latest_frame_source,
            latest_frame_detections,
            frame_time_s=latest_frame_time_s,
            captured_at=latest_frame_captured_at,
            processed_at=latest_frame_processed_at,
        )


def store_reading(
    frame,
    source: str,
    detections,
    frame_time_s: float | None = None,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    """Make `frame` the latest reading and append it to the history. Caller holds state_lock."""
    global latest_frame_snapshot, latest_frame, latest_frame_detections, latest_frame_source
    global latest_frame_time_s, latest_frame_captured_at, latest_frame_processed_at, latest_frame_model
    latest_frame = frame
    latest_frame_model = model_status()["name"]
    latest_frame_detections = detections
    latest_frame_source = source
    latest_frame_time_s = frame_time_s
    latest_frame_captured_at = captured_at
    latest_frame_processed_at = datetime.now(timezone.utc)
    latest_frame_snapshot = snapshot(
        frame,
        source,
        detections,
        record=True,
        frame_time_s=frame_time_s,
        captured_at=captured_at,
        processed_at=latest_frame_processed_at,
    )
    return latest_frame_snapshot


def frame_handler(source_name: str, recorded_at: datetime | None = None):
    """Callback for SourceWorker: detect outside state_lock so status polls never wait on inference.

    `on_frame.owner` holds the worker it belongs to. A frame that finishes after that
    worker was stopped or replaced is dropped, so it can't log rows under the old
    source or replace the new source's reading.
    """
    owner: list[SourceWorker] = []

    def on_frame(frame, frame_time_s: float | None) -> None:
        detections = detect_vehicles(frame)
        with state_lock:
            worker = owner[0] if owner else None
            if worker is None or worker.stopping or active_source is not worker:
                return
            store_reading(frame, source_name, detections, frame_time_s, capture_time(recorded_at, frame_time_s))

    on_frame.owner = owner  # type: ignore[attr-defined]
    return on_frame


def start_source(
    kind: str,
    target: str | int,
    name: str,
    realtime: bool = False,
    cleanup_path: str | None = None,
    recorded_at: datetime | None = None,
) -> SourceWorker:
    """Stop any running source and start reading a new one (one camera, one lot)."""
    global active_source
    with source_lock:
        if active_source is not None:
            active_source.stop()
        handler = frame_handler(name, recorded_at)
        worker = SourceWorker(
            kind,
            target,
            name,
            load_settings()["sample_interval_s"],
            handler,
            realtime=realtime,
            cleanup_path=cleanup_path,
        )
        handler.owner.append(worker)
        active_source = worker  # before start(), so its first frame is accepted
        return worker.start()


def source_status() -> dict[str, Any] | None:
    worker = active_source
    return worker.status() if worker is not None else None


def model_unavailable():
    """503 response when frames can't be processed yet, or None when the model is usable."""
    status = model_status()
    if status["state"] in USABLE_MODEL_STATES:
        return None
    return jsonify({"error": model_problem(status), "model_status": status}), 503


@app.get("/")
def index():
    return render_template("index.html")


def current_payload() -> dict[str, Any]:
    """The latest reading (or the no-reading view) with the model's live status.

    The cached reading keeps the model that produced it in `model`, but
    `model_status` is always fresh, so model switches and errors reach the dashboard.
    """
    with state_lock:
        payload = latest_frame_snapshot if latest_frame_snapshot is not None else snapshot()
    return {**payload, "model_status": model_status(), "job": source_status()}


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": f"Files must be smaller than {MAX_UPLOAD_MB} MB. For longer recordings, enter the file's path under Live source."}), 413


@app.get("/api/status")
def status():
    return jsonify(current_payload())


def slot_config_problem(slots: list[Any]) -> str | None:
    """Why a slot list can't be saved, or None if it is valid."""
    seen: set[str] = set()
    for index, slot in enumerate(slots, start=1):
        if not isinstance(slot, dict):
            return f"Slot {index} must be an object with an id and points."
        slot_id = slot.get("id")
        if not isinstance(slot_id, str) or not slot_id.strip() or len(slot_id) > MAX_SLOT_ID_LENGTH:
            return f"Slot {index} needs a text id of 1 to {MAX_SLOT_ID_LENGTH} characters."
        if slot_id in seen:
            return f"Slot id {slot_id} is used twice; each slot needs its own id."
        seen.add(slot_id)
        points = slot.get("points")
        try:
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError
            values = np.array([[float(x), float(y)] for x, y in points], dtype=np.float32)
        except (TypeError, ValueError):
            return f"Slot {slot_id} needs at least three [x, y] points."
        if not np.all((values >= 0) & (values <= 100)):
            return f"Slot {slot_id} has points outside 0-100."
        if cv2.contourArea(cv2.convexHull(values)) < MIN_SLOT_AREA:
            return f"Slot {slot_id} has (almost) no area; its corners can't all lie on one line."
    return None


@app.post("/api/config")
def config():
    payload = request.get_json(silent=True) or {}
    slots = payload.get("slots")
    settings = load_settings()
    lot_name = payload.get("lot_name", settings["lot_name"])
    review_threshold = payload.get("review_threshold", settings["review_threshold"])
    if not isinstance(slots, list):
        return jsonify({"error": "Provide a slots array."}), 400
    if not isinstance(lot_name, str) or not lot_name.strip() or len(lot_name.strip()) > 80:
        return jsonify({"error": "Lot name must be between 1 and 80 characters."}), 400
    try:
        review_threshold = float(review_threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "Review confidence must be a number from 0 to 1."}), 400
    if not 0 <= review_threshold <= 1:
        return jsonify({"error": "Review confidence must be between 0% and 100%."}), 400
    problem = slot_config_problem(slots)
    if problem:
        return jsonify({"error": problem}), 400
    with state_lock:
        save_slots(slots)
        save_settings(lot_name=lot_name.strip(), review_threshold=review_threshold)
        reclassify_latest_frame()
        return jsonify({"slots": slots, "lot_name": lot_name.strip(), "review_threshold": review_threshold, "message": "Lot configuration saved."})


@app.post("/api/review-threshold")
def review_threshold():
    payload = request.get_json(silent=True) or {}
    raw_threshold: Any = payload.get("review_threshold")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "Review confidence must be a number from 0 to 1."}), 400
    if not 0 <= threshold <= 1:
        return jsonify({"error": "Review confidence must be between 0% and 100%."}), 400
    with state_lock:
        save_settings(review_threshold=threshold)
        reclassify_latest_frame()
        return jsonify(latest_frame_snapshot if latest_frame_snapshot is not None else snapshot())


def is_readable_video(path: str) -> bool:
    video = cv2.VideoCapture(path)
    try:
        ok, frame = video.read()
        return bool(ok) and frame is not None
    finally:
        video.release()


@app.post("/api/process")
def process():
    """Process an uploaded image now, or start a background job that samples an uploaded video."""
    uploaded = request.files.get("source")  # larger than MAX_CONTENT_LENGTH: 413 before this
    if uploaded is None or not uploaded.filename:
        return jsonify(current_payload())
    try:
        recorded_at = parse_recorded_at(request.form.get("recorded_at"))
    except ValueError:
        return jsonify({"error": "Recorded-at must be a date and time, e.g. 2026-11-17T08:30."}), 400
    # Stream the upload to disk instead of holding it in memory.
    descriptor, path = tempfile.mkstemp(prefix="slotsight-", suffix=Path(uploaded.filename).suffix or ".bin")
    with os.fdopen(descriptor, "wb") as file:
        uploaded.save(file)
    frame = read_image(path)
    if frame is None:
        return process_video_upload(uploaded.filename, path, recorded_at)
    Path(path).unlink(missing_ok=True)
    try:
        detections = detect_vehicles(frame)
    except DetectionError as error:
        app.logger.exception("Could not process %s", uploaded.filename)
        return jsonify({"error": str(error)}), error.status_code
    with state_lock:
        payload = store_reading(frame, uploaded.filename, detections, captured_at=recorded_at)
    return jsonify({**payload, "job": source_status()})


def read_image(path: str):
    """Decode `path` as an image, or None if it isn't one (e.g. a video)."""
    try:
        if not cv2.haveImageReader(path):
            return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def process_video_upload(filename: str, path: str, recorded_at: datetime | None = None):
    """Sample an uploaded video, already saved at `path`, in the background (R-03)."""
    if not is_readable_video(path):
        Path(path).unlink(missing_ok=True)
        return jsonify({"error": "Upload a readable image or video file."}), 400
    unavailable = model_unavailable()
    if unavailable is not None:
        Path(path).unlink(missing_ok=True)
        return unavailable
    worker = start_source("file", path, filename, cleanup_path=path, recorded_at=recorded_at)
    interval = worker.interval_s
    return jsonify({"job": worker.status(), "message": f"Processing {filename}: one frame every {interval:g} s of video."}), 202


@app.get("/api/source")
def get_source():
    return jsonify({"job": source_status()})


@app.post("/api/source")
def set_source():
    """Start reading a live stream URL, a camera index, or a video file on this machine (R-03, R-04)."""
    payload = request.get_json(silent=True) or {}
    try:
        kind, target, name = parse_target(str(payload.get("target", "")))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    try:
        # Only recorded files have a start time; live frames are captured as they arrive.
        recorded_at = parse_recorded_at(payload.get("recorded_at")) if kind == "file" else None
    except ValueError:
        return jsonify({"error": "Recorded-at must be a date and time, e.g. 2026-11-17T08:30."}), 400
    unavailable = model_unavailable()
    if unavailable is not None:
        return unavailable
    worker = start_source(kind, target, name, realtime=bool(payload.get("realtime")), recorded_at=recorded_at)
    return jsonify({"job": worker.status(), "message": f"Reading {name}: one frame every {worker.interval_s:g} s."}), 202


@app.post("/api/source/stop")
def stop_source():
    with source_lock:
        worker = active_source
        if worker is not None:
            worker.stop()
    return jsonify({"job": source_status()})


@app.get("/api/models")
def list_models():
    """Every model SlotSight can use, whether it is installed, and the running install/load job."""
    active = configured_model_id()
    return jsonify({"models": models.catalogue(active), "active": active, "overridden": model_overridden(), "job": models.job()})


@app.post("/api/models/install")
def install_model():
    """Download a model's weights in the background (progress arrives through GET /api/models)."""
    model_id = str((request.get_json(silent=True) or {}).get("id", ""))
    if model_id not in detectors.MODELS:
        return jsonify({"error": f"Unknown model {model_id!r}."}), 404
    try:
        job = models.start(model_id, download=True, load=False)
    except DetectorError as error:
        return jsonify({"error": str(error)}), 409
    return jsonify({"job": job, "message": f"Installing {detectors.MODELS[model_id].name}."}), 202


@app.post("/api/models/select")
def select_model():
    """Make an installed model the active one and load it in the background."""
    model_id = str((request.get_json(silent=True) or {}).get("id", ""))
    if model_overridden():
        return jsonify({"error": "SLOTSIGHT_MODEL is set, so it decides the model. Unset it to choose here."}), 409
    try:
        spec = detectors.spec_for(model_id, MODELS_DIR)
    except DetectorError as error:
        return jsonify({"error": str(error)}), 404
    status = models.status(model_id)
    # "error" here means installed but failed to load: selecting it retries the load.
    # A failed download stays "missing" and is refused.
    if status["state"] not in ("ready", "loading", "error") or not detectors.is_installed(spec, MODELS_DIR):
        return jsonify({"error": model_problem(status)}), 409
    try:
        job = None if status["state"] == "ready" else models.start(model_id, download=False, load=True)
    except DetectorError as error:
        return jsonify({"error": str(error)}), 409
    with state_lock:
        save_settings(model=model_id)
    return jsonify({"job": job, "active": model_id, "message": f"Using {spec.name}. New frames use it; the last reading is unchanged."})


def history_rows() -> list[list[str]]:
    """The history as rows in the current column order, header first. Read-only.

    The file is copied under state_lock (so no append is half-written) and parsed
    outside it. An older file with fewer columns gets blanks for the new ones.
    """
    with state_lock:
        text = HISTORY_PATH.read_text(encoding="utf-8") if HISTORY_PATH.exists() else ""
    rows = [HISTORY_COLUMNS]
    for record in csv.DictReader(io.StringIO(text)):
        rows.append([record.get(column) or "" for column in HISTORY_COLUMNS])
    return rows


@app.get("/api/history.csv")
def export_history():
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(history_rows())
    data = io.BytesIO(output.getvalue().encode("utf-8"))
    return send_file(data, as_attachment=True, download_name="slotsight-occupancy-history.csv", mimetype="text/csv")


@app.get("/api/history.xlsx")
def export_excel():
    try:
        from openpyxl import Workbook
    except ImportError:
        return jsonify({"error": "Excel export requires the optional openpyxl package. Use CSV export or install requirements.txt."}), 501
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Occupancy history"
    for row in history_rows():
        sheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="slotsight-occupancy-history.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    ensure_history()
    load_slots()  # creates config/slots.json and settings.txt on first run
    # SLOTSIGHT_DEBUG=1 turns on Flask's debugger and auto-reloader (off by default:
    # the debugger runs code from the browser, and the reloader restarts on every edit).
    debug = os.environ.get("SLOTSIGHT_DEBUG", "").strip().lower() in ("1", "true", "yes")
    # With the debug reloader, this block runs in a watcher process and in the serving
    # child; only the child (WERKZEUG_RUN_MAIN) should download or load the model.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        prepare_model()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=debug)
