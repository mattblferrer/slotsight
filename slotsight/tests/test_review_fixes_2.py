"""Regression tests for the 2026-11-18 code review (docs/PLAN.md review log, 686fd1b..4c5615a)."""

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest
from conftest import TWO_SLOTS, blank_frame, png_upload

import camera as camera_module
import classifier
import tracking
import trends

T0 = datetime(2026, 11, 18, 8, tzinfo=timezone.utc)


def scene(car=False):
    image = np.full((480, 640, 3), 90, np.uint8)
    if car:
        cv2.rectangle(image, (60, 80), (260, 400), (220, 220, 230), -1)
        cv2.rectangle(image, (90, 120), (230, 180), (30, 30, 30), -1)
        for x, y in ((60, 130), (260, 130), (60, 350), (260, 350)):
            cv2.circle(image, (x, y), 18, (20, 20, 20), -1)
    return image


# --- 1. Re-classifying a live reading keeps the safeguards --------------------------------------


def test_review_slider_keeps_a_missed_car_unknown_during_a_live_run(client, fake_detector, two_slots, app):
    with app.state_lock:
        for _ in range(3):
            app.store_reading(scene(), "cam", [], smooth=True)
        before = app.store_reading(scene(car=True), "cam", [], smooth=True)["slots"][0]
    assert before["status"] == "unknown" and before["reason"] == "doesn't look empty"
    after = client.post("/api/review-threshold", json={"review_threshold": 0.6}).get_json()["slots"][0]
    assert after["status"] == "unknown" and after["reason"] == "doesn't look empty"
    assert after["changed_at"] is not None  # still smoothed, with its history


def test_revising_a_reading_does_not_count_as_another_frame():
    tracker = tracking.SlotTracker()
    reading = [{"id": "A", "status": "vacant", "confidence": 1.0}]
    tracker.update(reading, T0)
    tracker.update(reading, T0)
    assert tracker.revise(reading, T0)[0]["status"] == "unknown"  # still 2 frames: vacant needs 3
    assert tracker.update(reading, T0)[0]["status"] == "vacant"


def test_revise_replaces_the_last_reading():
    tracker = tracking.SlotTracker()
    occupied = [{"id": "A", "status": "occupied", "confidence": 0.9}]
    tracker.update(occupied, T0)
    assert tracker.update(occupied, T0)[0]["status"] == "occupied"
    # The same frame re-read as unknown (e.g. a higher threshold): unknown at once.
    assert tracker.revise([{"id": "A", "status": "unknown", "confidence": 0.6}], T0)[0]["status"] == "unknown"


# --- 2. Never save over an unreadable settings file ------------------------------------------------


def test_saving_over_an_unreadable_settings_file_is_refused(client, app):
    app.SETTINGS_PATH.parent.mkdir(parents=True)
    app.SETTINGS_PATH.write_text('{"lot_name": "My lot", oops', encoding="utf-8")
    response = client.post("/api/review-threshold", json={"review_threshold": 0.6})
    assert response.status_code == 409 and "Nothing was saved" in response.get_json()["error"]
    assert app.SETTINGS_PATH.read_text(encoding="utf-8") == '{"lot_name": "My lot", oops'
    config = client.post("/api/config", json={"slots": TWO_SLOTS, "lot_name": "x", "review_threshold": 0.7})
    assert config.status_code == 409
    assert app.load_slots() != TWO_SLOTS  # settings are checked before the new layout is written


# --- 3. The classifier never runs under state_lock on threshold changes ----------------------------------


def test_threshold_changes_reuse_the_classifier_output(client, fake_detector, two_slots, app, monkeypatch):
    calls = []

    def predict(frame, slots):
        calls.append(app.state_lock.locked())
        return [(0.1, 0.9), (0.95, 0.05)]

    monkeypatch.setattr(app.slot_classifier, "predict", predict)
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    client.post("/api/review-threshold", json={"review_threshold": 0.6})
    assert calls == [False]  # once, for the new frame, outside state_lock; reused afterwards
    moved = [{"id": "L", "points": [[0, 0], [45, 0], [45, 100], [0, 100]]}, TWO_SLOTS[1]]
    client.post("/api/config", json={"slots": moved, "lot_name": "x", "review_threshold": 0.7})
    assert calls == [False, False]  # a new layout is classified before the lock is taken


# --- 4 and 5. Trends -------------------------------------------------------------------------------------


def test_a_single_reading_shows_in_the_trend_series():
    rows = [["timestamp", "slot_id", "status", "confidence", "source", "frame_time_s", "raw_status"], [T0.isoformat(), "A", "vacant", "1", "x", "", "vacant"]]
    for range_key in trends.RANGES:
        series = trends.summarize(rows, range_key)["series"]
        assert series[-1]["free"] == 1.0


def test_trends_cache_serves_both_ranges(client, fake_detector, two_slots, app):
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    for range_key in ("24h", "7d", "24h"):
        assert client.get(f"/api/trends?range={range_key}").get_json()["range"] == range_key


# --- 6. Corner order doesn't change the classifier's view ------------------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        [[25, 25], [75, 25], [25, 75], [75, 75]],  # Z order
        [[75, 25], [75, 75], [25, 75], [25, 25]],  # starts at another corner
        [[25, 75], [75, 75], [75, 25], [25, 25]],  # anticlockwise
    ],
)
def test_corner_order_does_not_change_the_patch(points):
    frame = np.zeros((200, 200, 3), np.uint8)
    frame[50:150, 50:100] = 255
    reference = classifier.warp_slot(frame, [[25, 25], [75, 25], [75, 75], [25, 75]])
    assert np.array_equal(classifier.warp_slot(frame, points), reference)


# --- 7. Broken fine-tuned weights fall back to the base weights -------------------------------------------


def test_broken_fine_tuned_weights_fall_back_to_base(tmp_path):
    import torch

    torch.save(classifier.build_network().state_dict(), tmp_path / classifier.BASE_WEIGHTS)
    (tmp_path / classifier.LOCAL_WEIGHTS).write_bytes(b"truncated")
    slot_classifier = classifier.SlotClassifier(tmp_path)
    assert slot_classifier.predict(blank_frame(), TWO_SLOTS) is not None
    status = slot_classifier.status()
    assert status["weights"] == classifier.BASE_WEIGHTS and "could not load slot_classifier.pt" in status["message"]


# --- 8. Confidence describes the status shown ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("p_occupied", "detector", "status"),
    [(0.01, {"status": "occupied", "confidence": 0.9}, "unknown"), (0.01, {"status": "vacant", "confidence": 0.5}, "unknown"), (0.6, {"status": "vacant", "confidence": 1.0}, "unknown")],
)
def test_unknown_always_has_confidence_below_the_threshold(p_occupied, detector, status):
    result = classifier.combine([{"id": "A", **detector}], [(1 - p_occupied, p_occupied)], 0.7)[0]
    assert result["status"] == status and result["confidence"] < 0.7


def test_vacant_confidence_is_the_weaker_of_the_two():
    result = classifier.combine([{"id": "A", "status": "vacant", "confidence": 0.8}], [(0.99, 0.01)], 0.7)[0]
    assert result["status"] == "vacant" and result["confidence"] == 0.8


# --- 9. Camera reference follows the source ------------------------------------------------------------------


def noise(shape, seed):
    return np.random.default_rng(seed).integers(0, 255, shape, dtype=np.uint8)


def test_an_automatic_reference_is_replaced_by_the_next_source(tmp_path):
    monitor = camera_module.CameraMonitor(tmp_path / "ref.jpg")
    monitor.observe(noise((480, 640, 3), 0), TWO_SLOTS)
    assert (tmp_path / "ref.jpg").exists()
    monitor.forget_automatic_reference()
    assert not (tmp_path / "ref.jpg").exists() and monitor.status()["state"] == "no_reference"


def test_a_layout_reference_is_kept_across_sources(tmp_path):
    monitor = camera_module.CameraMonitor(tmp_path / "ref.jpg")
    monitor.set_reference(noise((480, 640, 3), 0))
    monitor.forget_automatic_reference()
    assert (tmp_path / "ref.jpg").exists()
    reloaded = camera_module.CameraMonitor(tmp_path / "ref.jpg")  # after a restart too
    reloaded.forget_automatic_reference()
    assert (tmp_path / "ref.jpg").exists()


def test_a_different_frame_size_is_unverified_not_moved(tmp_path, monkeypatch):
    monkeypatch.setattr(camera_module, "CHECK_EVERY", 1)
    monitor = camera_module.CameraMonitor(tmp_path / "ref.jpg")
    monitor.set_reference(noise((480, 640, 3), 0))
    for _ in range(4):
        monitor.observe(noise((720, 1280, 3), 1), TWO_SLOTS)
    assert monitor.status()["state"] == "unverified"


def test_frames_of_a_stopped_source_are_not_observed(fake_detector, two_slots, app, monkeypatch):
    seen = []
    monkeypatch.setattr(app.camera, "observe", lambda frame, slots: seen.append(1))

    class Worker:
        stopping = True

        def stop(self):
            pass

    worker = Worker()
    handler = app.frame_handler("cam")
    handler.owner.append(worker)
    monkeypatch.setattr(app, "active_source", worker)
    handler(blank_frame(), None)
    assert seen == []


# --- 10. History reads don't take state_lock, and skip a half-written row ------------------------------------


def test_history_rows_skip_a_half_written_last_row(app, monkeypatch):
    app.HISTORY_PATH.parent.mkdir(parents=True)
    header = ",".join(app.HISTORY_COLUMNS)
    app.HISTORY_PATH.write_text(f"{header}\n2026-11-18T08:00:00+00:00,A,vacant,1.0,x,,vacant\n2026-11-18T08:00:02+00:00,A,occ", encoding="utf-8")

    class NoLock:
        def __enter__(self):
            raise AssertionError("history_rows took state_lock")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(app, "state_lock", NoLock())
    rows = app.history_rows()
    assert len(rows) == 2 and rows[1][1] == "A"


def test_history_upgrade_leaves_no_temporary_file(app):
    app.HISTORY_PATH.parent.mkdir(parents=True)
    app.HISTORY_PATH.write_text("timestamp,slot_id,status,confidence,source\n2026-01-01T00:00:00+00:00,A,vacant,0.9,x\n", encoding="utf-8")
    app.ensure_history()
    assert [p.name for p in app.HISTORY_PATH.parent.iterdir()] == [app.HISTORY_PATH.name]
