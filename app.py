from __future__ import annotations

import csv
import base64
import io
import json
import math
import os
import random
import threading
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
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
DEFAULT_REVIEW_THRESHOLD = 0.45
DEFAULT_MODEL_PATH = "yolo11n.pt"

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)
state_lock = threading.Lock()
model = None
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


def load_settings() -> dict[str, Any]:
    settings: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update(data)
        except (OSError, json.JSONDecodeError):
            pass

    try:
        review_threshold = float(settings.get("review_threshold", DEFAULT_REVIEW_THRESHOLD))
    except (TypeError, ValueError):
        review_threshold = DEFAULT_REVIEW_THRESHOLD
    normalized = {
        "lot_name": str(settings.get("lot_name") or DEFAULT_LOT_NAME).strip()[:80],
        "review_threshold": max(0.0, min(1.0, review_threshold)),
        "model_path": str(settings.get("model_path") or os.environ.get("SLOTSIGHT_MODEL", DEFAULT_MODEL_PATH)),
    }
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def save_settings(lot_name: str, review_threshold: float, model_path: str | None = None) -> None:
    current = load_settings()
    settings = {
        "lot_name": lot_name,
        "review_threshold": review_threshold,
        "model_path": model_path or current["model_path"],
    }
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


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


def polygon_area(points: list[list[float]]) -> float:
    return abs(sum(points[index][0] * points[(index + 1) % len(points)][1] - points[(index + 1) % len(points)][0] * points[index][1] for index in range(len(points))) / 2)


def slot_points_pixels(slot: dict[str, Any], frame) -> Any:
    height, width = frame.shape[:2]
    return __import__("numpy").array(
        [[float(x) * width / 100, float(y) * height / 100] for x, y in slot["points"]],
        dtype="float32",
    )


def vehicle_overlaps_slot(box: tuple[float, float, float, float], slot: dict[str, Any], frame) -> bool:
    x1, y1, x2, y2 = box
    polygon = slot_points_pixels(slot, frame)
    candidates = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), ((x1 + x2) / 2, (y1 + y2) / 2)]
    if any(cv2.pointPolygonTest(polygon, point, False) >= 0 for point in candidates):
        return True
    center = tuple(__import__("numpy").mean(polygon, axis=0))
    return x1 <= center[0] <= x2 and y1 <= center[1] <= y2


def get_model():
    global model
    model_path = load_settings()["model_path"]
    if model is None and YOLO is not None and Path(model_path).exists():
        model = YOLO(model_path)
    return model


def detect_vehicles(frame) -> list[tuple[float, float, float, float, float]]:
    detector = get_model()
    detections: list[tuple[float, float, float, float, float]] = []
    if detector is not None and frame is not None:
        try:
            prediction = detector(frame, verbose=False)[0]
            for box, confidence, class_id in zip(prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist(), prediction.boxes.cls.tolist()):
                if int(class_id) in {2, 3, 5, 7} and confidence >= 0.25:
                    detections.append((*box, float(confidence)))
        except Exception:
            detections = []
    return detections


def classify_frame(frame, slots: list[dict[str, Any]], detections=None) -> list[dict[str, Any]]:
    detections = detect_vehicles(frame) if detections is None else detections

    results = []
    for index, slot in enumerate(slots):
        occupied = False
        confidence = 0.0
        if detections:
            for x1, y1, x2, y2, detection_confidence in detections:
                if vehicle_overlaps_slot((x1, y1, x2, y2), slot, frame):
                    occupied = True
                    confidence = max(confidence, detection_confidence)
            status = "occupied" if occupied else "vacant"
            if not occupied:
                confidence = 0.92
        else:
            # Keep the product demonstrable without downloading a model: stable sample state.
            occupied = (index * 7 + int(time.time() // 8)) % 5 in {0, 1}
            confidence = 0.86 if occupied else 0.92
            status = "occupied" if occupied else "vacant"
        results.append({"id": slot["id"], "status": status, "confidence": round(confidence, 2)})
    return results


def annotate_frame(frame, slots: list[dict[str, Any]], results: list[dict[str, Any]], detections) -> str:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    colors = {"unknown": (121, 216, 246), "occupied": (125, 142, 240), "vacant": (99, 187, 121)}
    for slot, result in zip(slots, results):
        points = __import__("numpy").array(
            [[int(x * width / 100), int(y * height / 100)] for x, y in slot["points"]],
            dtype="int32",
        )
        cv2.fillPoly(overlay, [points], colors[result["status"]])
    canvas = cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0)
    for slot, result in zip(slots, results):
        points = __import__("numpy").array(
            [[int(x * width / 100), int(y * height / 100)] for x, y in slot["points"]],
            dtype="int32",
        )
        cv2.polylines(canvas, [points], True, colors[result["status"]], max(2, width // 500))

    encoded, buffer = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise ValueError("Could not encode annotated frame")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def demo_results(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return classify_frame(None, slots)


def snapshot(source: str = "demo", frame=None, detections=None, record: bool = True) -> dict[str, Any]:
    slots = load_slots()
    detections = detect_vehicles(frame) if frame is not None and detections is None else (detections or [])
    results = classify_frame(frame, slots, detections) if frame is not None else demo_results(slots)
    settings = load_settings()
    unknown_threshold = settings["review_threshold"]
    for result in results:
        if result["confidence"] < unknown_threshold:
            result["status"] = "unknown"
    if record:
        append_history(results, source)
    occupied = sum(result["status"] == "occupied" for result in results)
    unknown = sum(result["status"] == "unknown" for result in results)
    for result, slot in zip(results, slots):
        result["points"] = slot["points"]
    response = {"slots": results, "total": len(results), "free": len(results) - occupied - unknown, "unknown": unknown, "updated_at": now_iso(), "source": source, "model": "YOLO" if get_model() else "demo classifier", "lot_name": settings["lot_name"], "review_threshold": unknown_threshold}
    if frame is not None:
        response["preview_image"] = annotate_frame(frame, slots, results, detections)
    return response


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
    global latest_frame_snapshot
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
        save_settings(lot_name.strip(), review_threshold)
        latest_frame_snapshot = None
        return jsonify({"slots": slots, "lot_name": lot_name.strip(), "review_threshold": review_threshold, "message": "Lot configuration saved."})


@app.post("/api/review-threshold")
def review_threshold():
    global latest_frame_snapshot
    payload = request.get_json(silent=True) or {}
    raw_threshold: Any = payload.get("review_threshold")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError):
        return jsonify({"error": "Review confidence must be a number from 0 to 1."}), 400
    if not 0 <= threshold <= 1:
        return jsonify({"error": "Review confidence must be between 0% and 100%."}), 400
    with state_lock:
        settings = load_settings()
        save_settings(settings["lot_name"], threshold)
        if latest_frame is not None:
            latest_frame_snapshot = snapshot(latest_frame_source, latest_frame, latest_frame_detections, record=False)
        elif latest_frame_snapshot is not None:
            latest_frame_snapshot["review_threshold"] = threshold
        else:
            latest_frame_snapshot = snapshot()
        return jsonify(latest_frame_snapshot)


@app.post("/api/process")
def process():
    global latest_frame_snapshot, latest_frame, latest_frame_detections, latest_frame_source
    uploaded = request.files.get("source")
    if uploaded is None or not uploaded.filename:
        with state_lock:
            if latest_frame_snapshot is not None:
                return jsonify(latest_frame_snapshot), 200
            return jsonify(snapshot()), 200
    if uploaded.content_length and uploaded.content_length > 100 * 1024 * 1024:
        return jsonify({"error": "Files must be smaller than 100 MB."}), 413
    raw = uploaded.read()
    array = __import__("numpy").frombuffer(raw, dtype=__import__("numpy").uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
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
        latest_frame = frame.copy()
        latest_frame_detections = detect_vehicles(latest_frame)
        latest_frame_source = uploaded.filename
        latest_frame_snapshot = snapshot(uploaded.filename, latest_frame, latest_frame_detections)
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
