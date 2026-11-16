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

from sources import SourceWorker, parse_target

try:
    from ultralytics import YOLO
except ImportError:  # Optional for lightweight demo mode.
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"  # files you edit: slot polygons and settings
DATA_DIR = BASE_DIR / "data"  # generated output: occupancy history
MODELS_DIR = BASE_DIR / "models"  # detection model weights (.pt)
CONFIG_PATH = CONFIG_DIR / "slots.json"
SETTINGS_PATH = CONFIG_DIR / "settings.txt"
HISTORY_PATH = DATA_DIR / "occupancy_history.csv"
DEFAULT_LOT_NAME = "Lot 01"
DEFAULT_REVIEW_THRESHOLD = 0.7
DEFAULT_OVERLAP_THRESHOLD = 0.4
DEFAULT_MODEL_PATH = "yolo11s.pt"
# Seconds between sampled frames (video time for files, wall-clock time for live sources).
# With the dashboard polling every 2 s this keeps updates within R-04's 5 s.
DEFAULT_SAMPLE_INTERVAL_S = 2.0
SAMPLE_INTERVAL_RANGE_S = (0.2, 60.0)
HISTORY_COLUMNS = ["timestamp", "slot_id", "status", "confidence", "source", "frame_time_s"]
VEHICLE_CLASS_IDS = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck
# Keep weak detections: they should push a slot to "unknown" rather than read as free (R-07).
DETECTION_CONFIDENCE_FLOOR = 0.1


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
state_lock = threading.Lock()  # guards the latest_frame_* globals and history writes
model_lock = threading.Lock()  # guards loading the model
inference_lock = threading.Lock()  # one inference at a time; the model is shared across threads
source_lock = threading.Lock()  # guards active_source
active_source: SourceWorker | None = None
model = None
model_loaded_path: Path | None = None
model_failed_path: Path | None = None
latest_frame_snapshot: dict[str, Any] | None = None
latest_frame = None
latest_frame_detections = None
latest_frame_source = "latest frame"
latest_frame_time_s: float | None = None
latest_frame_captured_at: datetime | None = None

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


def load_slots() -> list[dict[str, Any]]:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return DEFAULT_SLOTS.copy()


def save_slots(slots: list[dict[str, Any]]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(slots, indent=2), encoding="utf-8")


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


def load_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update(data)
        except (OSError, json.JSONDecodeError):
            pass

    normalized = {
        "lot_name": str(settings.get("lot_name") or DEFAULT_LOT_NAME).strip()[:80],
        "review_threshold": unit_interval(settings.get("review_threshold"), DEFAULT_REVIEW_THRESHOLD),
        "overlap_threshold": unit_interval(settings.get("overlap_threshold"), DEFAULT_OVERLAP_THRESHOLD),
        "sample_interval_s": sample_interval(settings.get("sample_interval_s")),
        "model_path": str(settings.get("model_path") or DEFAULT_MODEL_PATH),
    }
    if not SETTINGS_PATH.exists() or set(normalized) - set(settings):
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def save_settings(**updates: Any) -> dict[str, Any]:
    settings = load_settings()
    settings.update(updates)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
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


def resolve_model_path() -> Path:
    """SLOTSIGHT_MODEL overrides settings.txt. Relative paths resolve against models/,
    falling back to the app folder so a value like "models/yolo11s.pt" also works."""
    path = Path(os.environ.get("SLOTSIGHT_MODEL") or load_settings()["model_path"])
    if path.is_absolute():
        return path
    in_models = MODELS_DIR / path
    if in_models.exists() or not (BASE_DIR / path).exists():
        return in_models
    return BASE_DIR / path


def get_model():
    """Return the loaded detector, reloading it when the configured path changes."""
    global model, model_loaded_path, model_failed_path
    path = resolve_model_path()
    current = model
    if current is not None and model_loaded_path == path:
        return current
    with model_lock:
        if model is not None and model_loaded_path == path:
            return model
        model, model_loaded_path = None, None
        if YOLO is None or not path.exists() or model_failed_path == path:
            return None
        try:
            loaded = YOLO(str(path))
            # The first inference is several times slower (lazy setup); pay it at load
            # time rather than on the first real frame, so it never delays an update (R-04).
            loaded(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            model, model_loaded_path = loaded, path
        except Exception:
            app.logger.exception("Could not load detection model from %s", path)
            model_failed_path = path
        return model


def detect_vehicles(frame) -> list[tuple[float, float, float, float, float]]:
    """Return vehicle boxes as (x1, y1, x2, y2, confidence) in pixels.

    Raises DetectionError when no model is loaded or inference fails, so a failure
    can never be mistaken for an empty lot.
    """
    detector = get_model()
    if detector is None:
        raise DetectionError(
            "No detection model is loaded. Install ultralytics, put the model in models/ and set model_path in config/settings.txt.",
            status_code=503,
        )
    try:
        with inference_lock:
            prediction = detector(frame, verbose=False, conf=DETECTION_CONFIDENCE_FLOOR, classes=list(VEHICLE_CLASS_IDS))[0]
        boxes = zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist(), prediction.boxes.cls.tolist())
        return [(*box, float(confidence)) for box, confidence, class_id in boxes if int(class_id) in VEHICLE_CLASS_IDS]
    except Exception as error:
        raise DetectionError(f"Vehicle detection failed: {error}") from error


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
) -> dict[str, Any]:
    """Build the dashboard payload.

    With a frame: classify it from `detections` (detected here if not given) and,
    if `record` is set, append the reading to the history.
    Without a frame: demo data when no model is loaded, otherwise every slot is
    "unknown" until a frame is processed. These readings are never recorded.
    """
    slots = load_slots()
    settings = load_settings()
    has_model = get_model() is not None
    if frame is not None:
        if detections is None:
            detections = detect_vehicles(frame)
        results = classify_frame(frame, slots, detections, settings["overlap_threshold"])
        source = source or "latest frame"
    elif has_model:
        results, source, record = waiting_results(slots), "none", False
    else:
        results, source, record = demo_results(slots), "demo", False

    review_threshold = settings["review_threshold"]
    for result in results:
        if result["confidence"] < review_threshold:
            result["status"] = "unknown"
    if record:
        append_history(results, source, frame_time_s, captured_at)
    occupied = sum(result["status"] == "occupied" for result in results)
    unknown = sum(result["status"] == "unknown" for result in results)
    for result, slot in zip(results, slots):
        result["points"] = slot["points"]
    response = {
        "slots": results,
        "total": len(results),
        "free": len(results) - occupied - unknown,
        "unknown": unknown,
        "updated_at": now_iso(),
        "source": source,
        "frame_time_s": None if frame_time_s is None else round(frame_time_s, 2),
        "captured_at": captured_at.isoformat(timespec="seconds") if captured_at else None,
        "model": "YOLO" if has_model else "demo classifier",
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
    global latest_frame_time_s, latest_frame_captured_at
    latest_frame = frame
    latest_frame_detections = detections
    latest_frame_source = source
    latest_frame_time_s = frame_time_s
    latest_frame_captured_at = captured_at
    latest_frame_snapshot = snapshot(
        frame, source, detections, record=True, frame_time_s=frame_time_s, captured_at=captured_at
    )
    return latest_frame_snapshot


def frame_handler(source_name: str, recorded_at: datetime | None = None):
    """Callback for SourceWorker: detect outside state_lock so status polls never wait on inference."""

    def on_frame(frame, frame_time_s: float | None) -> None:
        detections = detect_vehicles(frame)
        with state_lock:
            store_reading(frame, source_name, detections, frame_time_s, capture_time(recorded_at, frame_time_s))

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
        active_source = SourceWorker(
            kind,
            target,
            name,
            load_settings()["sample_interval_s"],
            frame_handler(name, recorded_at),
            realtime=realtime,
            cleanup_path=cleanup_path,
        ).start()
        return active_source


def source_status() -> dict[str, Any] | None:
    worker = active_source
    return worker.status() if worker is not None else None


def model_unavailable():
    return jsonify({"error": "No detection model is loaded. Install ultralytics, put the model in models/ and set model_path in config/settings.txt."}), 503


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with state_lock:
        payload = latest_frame_snapshot if latest_frame_snapshot is not None else snapshot()
    return jsonify({**payload, "job": source_status()})


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
    try:
        for slot in slots:
            if not slot.get("id") or len(slot.get("points", [])) < 3:
                raise ValueError
            for point in slot["points"]:
                if len(point) != 2 or not all(0 <= float(value) <= 100 for value in point):
                    raise ValueError
    except (AttributeError, TypeError, ValueError):
        return jsonify({"error": "Each slot needs an id and at least three points between 0 and 100."}), 400
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
    uploaded = request.files.get("source")
    if uploaded is None or not uploaded.filename:
        with state_lock:
            payload = latest_frame_snapshot if latest_frame_snapshot is not None else snapshot()
        return jsonify({**payload, "job": source_status()})
    if uploaded.content_length and uploaded.content_length > 100 * 1024 * 1024:
        return jsonify({"error": "Files must be smaller than 100 MB."}), 413
    try:
        recorded_at = parse_recorded_at(request.form.get("recorded_at"))
    except ValueError:
        return jsonify({"error": "Recorded-at must be a date and time, e.g. 2026-11-17T08:30."}), 400
    raw = uploaded.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return process_video_upload(uploaded.filename, raw, recorded_at)
    try:
        detections = detect_vehicles(frame)
    except DetectionError as error:
        app.logger.exception("Could not process %s", uploaded.filename)
        return jsonify({"error": str(error)}), error.status_code
    with state_lock:
        payload = store_reading(frame, uploaded.filename, detections, captured_at=recorded_at)
    return jsonify({**payload, "job": source_status()})


def process_video_upload(filename: str, raw: bytes, recorded_at: datetime | None = None):
    """Save the upload to a temporary file and sample it in the background (R-03)."""
    descriptor, path = tempfile.mkstemp(prefix="slotsight-", suffix=Path(filename).suffix or ".mp4")
    with os.fdopen(descriptor, "wb") as file:
        file.write(raw)
    if not is_readable_video(path):
        Path(path).unlink(missing_ok=True)
        return jsonify({"error": "Upload a readable image or video file."}), 400
    if get_model() is None:
        Path(path).unlink(missing_ok=True)
        return model_unavailable()
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
    if get_model() is None:
        return model_unavailable()
    worker = start_source(kind, target, name, realtime=bool(payload.get("realtime")), recorded_at=recorded_at)
    return jsonify({"job": worker.status(), "message": f"Reading {name}: one frame every {worker.interval_s:g} s."}), 202


@app.post("/api/source/stop")
def stop_source():
    with source_lock:
        worker = active_source
        if worker is not None:
            worker.stop()
    return jsonify({"job": source_status()})


@app.get("/api/history.csv")
def export_history():
    ensure_history()
    return send_file(HISTORY_PATH, as_attachment=True, download_name="slotsight-occupancy-history.csv", mimetype="text/csv")


@app.get("/api/history.xlsx")
def export_excel():
    ensure_history()
    try:
        from openpyxl import Workbook
    except ImportError:
        return jsonify({"error": "Excel export requires the optional openpyxl package. Use CSV export or install requirements.txt."}), 501
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Occupancy history"
    with HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            sheet.append(row)
    output = io.BytesIO()
    workbook.save(output)
    output.seek(0)
    return send_file(output, as_attachment=True, download_name="slotsight-occupancy-history.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    ensure_history()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=True)
