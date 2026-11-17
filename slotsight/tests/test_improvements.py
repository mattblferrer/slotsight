"""Milestone 2 improvements: detection boxes, smoothing and last-changed time, camera-shift
check, and occupancy trends."""

import csv
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pytest
from conftest import TWO_SLOTS, blank_frame, history_rows, png_upload, wait_until

import camera as camera_module
import trends
import tracking

T0 = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)


# --- Detection boxes on the preview -------------------------------------------------------


def test_detection_counts_only_when_it_overlaps_a_slot_enough(app):
    import app as slotsight

    hull = cv2.convexHull(slotsight.slot_polygon_pixels(TWO_SLOTS[0], 640, 480)).reshape(-1, 2)
    assert slotsight.detection_counts((20, 20, 300, 460), [hull], 0.4)
    assert not slotsight.detection_counts((300, 0, 640, 480), [hull], 0.4)


def test_preview_draws_boxes_in_their_own_colour(app):
    frame = np.zeros((480, 640, 3), np.uint8)
    results = [{"id": "L", "status": "occupied", "confidence": 0.9}, {"id": "R", "status": "vacant", "confidence": 1.0}]
    without = app.annotate_frame(frame, TWO_SLOTS, results, [])
    with_boxes = app.annotate_frame(frame, TWO_SLOTS, results, [(40, 40, 280, 440, 0.9), (330, 100, 400, 200, 0.2)])
    assert without != with_boxes

    canvas = frame.copy()
    hulls = [cv2.convexHull(app.slot_polygon_pixels(slot, 640, 480)).reshape(-1, 2) for slot in TWO_SLOTS]
    app.draw_detections(canvas, [(40, 40, 280, 440, 0.9)], hulls, 0.4, 0.7)
    assert tuple(canvas[300, 40]) == app.BOX_COLOR  # solid left edge of a strong box
    dashed = frame.copy()
    app.draw_detections(dashed, [(330, 100, 400, 400, 0.2)], hulls, 0.4, 0.7)
    left_edge = dashed[150:400, 330]
    assert 0 < np.count_nonzero(left_edge.any(axis=1)) < len(left_edge)  # dashed: gaps along the edge


# --- Smoothing and last-changed time ------------------------------------------------------------


def statuses(sequence, shown="unknown"):
    shown_now, recent, out = shown, [], []
    for raw in sequence:
        recent = (recent + [raw])[-tracking.WINDOW :]
        shown_now = tracking.next_status(shown_now, recent)
        out.append(shown_now)
    return out


def test_vacant_needs_three_agreeing_frames():
    assert statuses(["vacant"] * 3) == ["unknown", "unknown", "vacant"]


def test_occupied_needs_two_and_unknown_is_immediate():
    assert statuses(["occupied", "occupied"]) == ["unknown", "occupied"]
    assert statuses(["unknown"], shown="vacant") == ["unknown"]


def test_a_passing_car_does_not_flicker_a_free_slot():
    assert statuses(["occupied", "vacant", "vacant"], shown="vacant") == ["vacant", "vacant", "vacant"]


def test_a_parking_car_shows_within_two_frames():
    assert statuses(["occupied", "occupied"], shown="vacant") == ["vacant", "occupied"]


def test_a_flipping_slot_becomes_unknown_never_stays_free():
    assert statuses(["occupied", "vacant", "occupied"], shown="vacant")[-1] == "unknown"


def test_leaving_needs_three_vacant_frames():
    assert statuses(["vacant", "vacant", "vacant"], shown="occupied") == ["occupied", "occupied", "vacant"]


def raw(status_by_slot):
    return [{"id": slot, "status": status, "confidence": 0.9} for slot, status in status_by_slot.items()]


def test_tracker_reports_when_a_status_began():
    tracker = tracking.SlotTracker()
    out = []
    for i, status in enumerate(["occupied", "occupied", "occupied", "vacant", "vacant", "vacant"]):
        out.append(tracker.update(raw({"A": status}), T0 + timedelta(seconds=2 * i))[0])
    # First confirmed status after warm-up: may predate tracking, so a lower bound.
    assert out[1]["status"] == "occupied" and out[1]["changed_since_start"] is True
    assert out[1]["changed_at"] == (T0 + timedelta(seconds=2)).isoformat(timespec="seconds")
    assert out[3]["status"] == "occupied" and out[3]["raw_status"] == "vacant"  # still confirming
    assert out[5]["status"] == "vacant" and out[5]["changed_since_start"] is False
    assert out[5]["changed_at"] == (T0 + timedelta(seconds=10)).isoformat(timespec="seconds")


def test_single_images_are_not_smoothed(client, fake_detector, two_slots, app):
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    assert {slot["status"] for slot in data["slots"]} == {"vacant"}
    assert all(slot["changed_at"] is None for slot in data["slots"])


def test_video_is_smoothed_and_history_keeps_both_statuses(client, fake_detector, two_slots, app, tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for _ in range(80):  # 8 s at 10 fps, sampled every 2 s: 4 frames
        writer.write(blank_frame())
    writer.release()
    with video.open("rb") as handle:
        assert client.post("/api/process", data={"source": (handle, "clip.mp4")}, content_type="multipart/form-data").status_code == 202
    assert wait_until(lambda: client.get("/api/source").get_json()["job"]["state"] == "finished")
    rows = [row for row in history_rows(app) if row["slot_id"] == "L"]
    assert [row["raw_status"] for row in rows] == ["vacant"] * 4
    assert [row["status"] for row in rows] == ["unknown", "unknown", "vacant", "vacant"]


def test_older_history_gains_the_raw_status_column(app):
    app.HISTORY_PATH.parent.mkdir(parents=True)
    app.HISTORY_PATH.write_text("timestamp,slot_id,status,confidence,source,frame_time_s\n2026-01-01T00:00:00+00:00,A01,vacant,0.9,x.png,\n", encoding="utf-8")
    app.ensure_history()
    with app.HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == app.HISTORY_COLUMNS and rows[0]["raw_status"] == ""


# --- Camera-shift check -------------------------------------------------------------------------------


def textured_scene(seed=1):
    rng = np.random.default_rng(seed)
    scene = np.full((720, 1280, 3), 120, np.uint8)
    for _ in range(250):
        x, y = rng.integers(0, 1240), rng.integers(0, 680)
        w, h = rng.integers(10, 60), rng.integers(10, 60)
        cv2.rectangle(scene, (int(x), int(y)), (int(x + w), int(y + h)), tuple(int(c) for c in rng.integers(0, 255, 3)), -1)
    return scene


def shifted(image, dx_pct):
    dx = image.shape[1] * dx_pct / 100
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, 0]]), (image.shape[1], image.shape[0]), borderValue=(120, 120, 120))


def test_estimate_shift_ok_for_the_same_view_and_flags_a_move():
    scene = textured_scene()
    same = camera_module.estimate_shift(scene, scene, TWO_SLOTS)
    assert same["state"] == "ok" and same["shift_pct"] < 0.5
    moved = camera_module.estimate_shift(scene, shifted(scene, 5), TWO_SLOTS)
    assert moved["state"] == "shifted" and 4 < moved["shift_pct"] < 6


def test_featureless_frames_are_unverified_not_an_alarm():
    flat = np.full((720, 1280, 3), 30, np.uint8)
    assert camera_module.estimate_shift(textured_scene(), flat, TWO_SLOTS)["state"] == "unverified"


def test_monitor_warns_only_after_repeated_shifts(tmp_path, monkeypatch):
    monkeypatch.setattr(camera_module, "CHECK_EVERY", 1)
    monitor = camera_module.CameraMonitor(tmp_path / "reference.jpg")
    scene = textured_scene()
    monitor.observe(scene, TWO_SLOTS)  # first frame becomes the reference
    assert (tmp_path / "reference.jpg").exists()
    moved = shifted(scene, 5)
    states = []
    for _ in range(3):
        monitor.observe(moved, TWO_SLOTS)
        states.append(monitor.status()["state"])
    assert states == ["suspect", "suspect", "moved"]
    monitor.set_reference(moved)  # the user re-saves the layout
    assert monitor.status()["state"] == "ok"


def test_saving_a_changed_layout_sets_the_reference(client, fake_detector, two_slots, app):
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    client.post("/api/config", json={"slots": two_slots, "lot_name": "Renamed", "review_threshold": 0.7})
    assert not app.REFERENCE_PATH.exists()  # same slots: a rename must not reset the reference
    changed = [{"id": "L", "points": [[0, 0], [40, 0], [40, 100], [0, 100]]}, two_slots[1]]
    response = client.post("/api/config", json={"slots": changed, "lot_name": "Renamed", "review_threshold": 0.7})
    assert "camera reference" in response.get_json()["message"]
    assert app.REFERENCE_PATH.exists()
    assert client.get("/api/status").get_json()["camera"]["state"] == "ok"


# --- Trends -----------------------------------------------------------------------------------------------------


def history(frames):
    """Rows like app.history_rows(): (time, {slot: status}) per frame."""
    rows = [["timestamp", "slot_id", "status", "confidence", "source", "frame_time_s", "raw_status"]]
    for at, statuses_by_slot in frames:
        for slot, status in statuses_by_slot.items():
            rows.append([at.isoformat(timespec="seconds"), slot, status, "0.9", "cam", "", status])
    return rows


def test_trends_are_empty_without_history():
    data = trends.summarize([], "24h")
    assert data["series"] == [] and data["frames"] == 0


def test_trends_bucket_free_slots_and_find_peak_hours():
    frames = []
    for hour in range(24):
        busy = hour in (8, 9, 17)
        for minute in (0, 30):
            at = T0.replace(hour=0) + timedelta(hours=hour, minutes=minute)
            frames.append((at, {"A": "occupied" if busy else "vacant", "B": "occupied" if busy else "vacant", "C": "unknown"}))
    data = trends.summarize(history(frames), "24h", local_tz=timezone.utc)
    assert data["end"] == frames[-1][0].isoformat(timespec="seconds")
    filled = [point for point in data["series"] if point["free"] is not None]
    assert filled and all(point["total"] == 3 for point in filled)
    assert {point["free"] for point in filled} <= {0.0, 2.0}
    assert sorted(data["peak_hours"]) == [8, 9, 17]
    assert data["hours"][8]["occupancy"] == pytest.approx(2 / 3, abs=0.001)


def test_trends_range_ends_at_the_latest_reading_not_now():
    old = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    data = trends.summarize(history([(old, {"A": "vacant"})]), "7d")
    assert data["end"].startswith("2025-01-01") and data["frames"] == 1


def test_trends_endpoint_is_read_only_and_validates_range(client, fake_detector, two_slots, app):
    assert client.get("/api/trends?range=1y").status_code == 400
    data = client.get("/api/trends").get_json()
    assert data["frames"] == 0 and not app.HISTORY_PATH.exists()
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    data = client.get("/api/trends?range=7d").get_json()
    assert data["frames"] == 1 and data["range"] == "7d"
