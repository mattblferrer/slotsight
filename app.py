from __future__ import annotations

import base64
import csv
import io
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

try:
    from ultralytics import YOLO
except ImportError:  # Optional for lightweight demo mode.
    YOLO = None

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "slots.json"
SETTINGS_PATH = BASE_DIR / "settings.txt"
HISTORY_PATH = BASE_DIR / "occupancy_history.csv"
DEFAULT_LOT_NAME = "Lot 01"
DEFAULT_REVIEW_THRESHOLD = 0.7
DEFAULT_OVERLAP_THRESHOLD = 0.4
DEFAULT_MODEL_PATH = "yolo11s.pt"
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
state_lock = threading.Lock()
model = None
model_loaded_path: Path | None = None
model_failed_path: Path | None = None
latest_frame_snapshot: dict[str, Any] | None = None
latest_frame = None
latest_frame_detections = None
latest_frame_source = "latest frame"

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
    CONFIG_PATH.write_text(json.dumps(slots, indent=2), encoding="utf-8")


def unit_interval(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


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
        "model_path": str(settings.get("model_path") or DEFAULT_MODEL_PATH),
    }
    if not SETTINGS_PATH.exists() or set(normalized) - set(settings):
        SETTINGS_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def save_settings(**updates: Any) -> dict[str, Any]:
    settings = load_settings()
    settings.update(updates)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def ensure_history() -> None:
    if not HISTORY_PATH.exists():
        with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(["timestamp", "slot_id", "status", "confidence", "source"])


def append_history(results: list[dict[str, Any]], source: str) -> None:
    ensure_history()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = now_iso()
        for result in results:
            writer.writerow([timestamp, result["id"], result["status"], result["confidence"], source])


def resolve_model_path() -> Path:
    """SLOTSIGHT_MODEL overrides settings.txt; relative paths resolve against the project folder."""
    path = Path(os.environ.get("SLOTSIGHT_MODEL") or load_settings()["model_path"])
    return path if path.is_absolute() else BASE_DIR / path


def get_model():
    """Return the loaded detector, reloading it when the configured path changes."""
    global model, model_loaded_path, model_failed_path
    path = resolve_model_path()
    if model is not None and model_loaded_path == path:
        return model
    model, model_loaded_path = None, None
    if YOLO is None or not path.exists() or model_failed_path == path:
        return None
    try:
        model = YOLO(str(path))
        model_loaded_path = path
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
            "No detection model is loaded. Install ultralytics and set model_path in settings.txt.",
            status_code=503,
        )
    try:
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


def snapshot(frame=None, source: str | None = None, detections=None, record: bool = False) -> dict[str, Any]:
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
        append_history(results, source)
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
        latest_frame_snapshot = snapshot(latest_frame, latest_frame_source, latest_frame_detections)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    with state_lock:
        if latest_frame_snapshot is not None:
            return jsonify(latest_frame_snapshot)
        return jsonify(snapshot())


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


@app.post("/api/process")
def process():
    global latest_frame_snapshot, latest_frame, latest_frame_detections, latest_frame_source
    uploaded = request.files.get("source")
    if uploaded is None or not uploaded.filename:
        with state_lock:
            return jsonify(latest_frame_snapshot if latest_frame_snapshot is not None else snapshot())
    if uploaded.content_length and uploaded.content_length > 100 * 1024 * 1024:
        return jsonify({"error": "Files must be smaller than 100 MB."}), 413
    raw = uploaded.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        suffix = Path(uploaded.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temporary_file:
            temporary_file.write(raw)
            temporary_file.flush()
            video = cv2.VideoCapture(temporary_file.name)
            success, frame = video.read()
            video.release()
    if frame is None:
        return jsonify({"error": "Upload a readable image or video file."}), 400
    with state_lock:
        try:
            detections = detect_vehicles(frame)
        except DetectionError as error:
            app.logger.exception("Could not process %s", uploaded.filename)
            return jsonify({"error": str(error)}), error.status_code
        latest_frame = frame
        latest_frame_detections = detections
        latest_frame_source = uploaded.filename
        latest_frame_snapshot = snapshot(frame, uploaded.filename, detections, record=True)
        return jsonify(latest_frame_snapshot)


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
