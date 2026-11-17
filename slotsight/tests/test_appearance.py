"""The "doesn't look empty" safeguard (appearance.py) for cars the detector misses."""

import csv
import sys

import cv2
import numpy as np
from conftest import TWO_SLOTS, history_rows, png_upload, wait_until

import appearance
import evaluate

H, W = 480, 640


# A static scene, as from a fixed camera: the same pavement texture in every frame.
_PAVEMENT = cv2.GaussianBlur(np.random.default_rng(3).normal(0, 14, (H, W)).astype(np.float32), (0, 0), 3)


def lot(car_in_left=False, brightness=0, seed=3):
    """Asphalt with a fixed texture and a little sensor noise; optionally a car in slot L
    (body, windscreen, rear window and wheels) that the detector is assumed to miss."""
    noise = np.random.default_rng(seed).normal(0, 2, (H, W)).astype(np.float32)
    grey = np.clip(95 + brightness + _PAVEMENT + noise, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    cv2.line(image, (320, 0), (320, H), (230, 230, 230), 4)  # the line between the slots
    if car_in_left:
        cv2.rectangle(image, (60, 80), (260, 400), (210, 210, 220), -1)
        cv2.rectangle(image, (90, 120), (230, 180), (40, 40, 40), -1)
        cv2.rectangle(image, (90, 300), (230, 360), (40, 40, 40), -1)
        for x, y in ((60, 130), (260, 130), (60, 350), (260, 350)):
            cv2.circle(image, (x, y), 18, (20, 20, 20), -1)
    return image


def vacant_both():
    return [{"id": "L", "status": "vacant", "confidence": 1.0}, {"id": "R", "status": "vacant", "confidence": 1.0}]


def test_a_missed_car_turns_unknown_and_the_empty_slot_stays_vacant():
    reference = appearance.EmptyReference()
    for seed in range(3):  # learn the empty lot
        reference.check(lot(seed=seed), TWO_SLOTS, vacant_both(), 0.8)
    out = reference.check(lot(car_in_left=True, seed=9), TWO_SLOTS, vacant_both(), 0.8)  # detector missed the car
    assert out[0]["status"] == "unknown" and out[0]["reason"] == "doesn't look empty"
    assert out[0]["empty_diff"] > 0.8
    assert out[1]["status"] == "vacant" and "reason" not in out[1]


def test_slow_lighting_changes_are_learned_but_a_car_is_not():
    reference = appearance.EmptyReference()
    for step in range(12):  # dusk: the lot darkens a little every frame
        out = reference.check(lot(brightness=-3 * step, seed=step), TWO_SLOTS, vacant_both(), 0.8)
        assert [r["status"] for r in out] == ["vacant", "vacant"]
    for _ in range(5):  # a missed car stays flagged: it is never absorbed into "empty"
        out = reference.check(lot(car_in_left=True, brightness=-36), TWO_SLOTS, vacant_both(), 0.8)
        assert out[0]["status"] == "unknown"


def test_only_vacant_readings_are_checked_and_threshold_zero_is_off():
    reference = appearance.EmptyReference()
    reference.check(lot(), TWO_SLOTS, vacant_both(), 0.8)
    occupied = [{"id": "L", "status": "occupied", "confidence": 0.9}, {"id": "R", "status": "vacant", "confidence": 1.0}]
    assert reference.check(lot(car_in_left=True), TWO_SLOTS, occupied, 0.8)[0]["status"] == "occupied"
    assert reference.check(lot(car_in_left=True), TWO_SLOTS, vacant_both(), 0.0)[0]["status"] == "vacant"


def test_editing_a_slot_starts_a_fresh_reference():
    reference = appearance.EmptyReference()
    reference.check(lot(), TWO_SLOTS, vacant_both(), 0.8)
    moved = [{"id": "L", "points": [[0, 0], [45, 0], [45, 100], [0, 100]]}, TWO_SLOTS[1]]
    out = reference.check(lot(car_in_left=True), moved, vacant_both(), 0.8)
    assert out[0]["status"] == "vacant" and out[0]["empty_diff"] is None  # seeding, not comparing


def write_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 1, (W, H))
    for frame in frames:
        writer.write(frame)
    writer.release()


def test_video_frames_are_checked_and_logged(client, fake_detector, two_slots, app, tmp_path):
    app.save_settings(sample_interval_s=1)
    video = tmp_path / "clip.avi"
    write_video(video, [lot(seed=s) for s in range(4)] + [lot(car_in_left=True, seed=s) for s in range(4, 8)])
    with video.open("rb") as handle:
        assert client.post("/api/process", data={"source": (handle, "clip.avi")}, content_type="multipart/form-data").status_code == 202
    assert wait_until(lambda: client.get("/api/source").get_json()["job"]["state"] == "finished")
    left = [row for row in history_rows(app) if row["slot_id"] == "L"]
    assert [row["raw_status"] for row in left[:4]] == ["vacant"] * 4
    assert [row["raw_status"] for row in left[4:]] == ["unknown"] * 4  # the detector never saw the car


def test_single_images_are_not_checked(client, fake_detector, two_slots, app):
    client.post("/api/process", data=png_upload(lot()), content_type="multipart/form-data")
    data = client.post("/api/process", data=png_upload(lot(car_in_left=True)), content_type="multipart/form-data").get_json()
    assert {slot["status"] for slot in data["slots"]} == {"vacant"}


def test_evaluation_runs_the_check_per_session_and_reports_it(app, fake_detector, tmp_path, monkeypatch):
    app.save_slots(TWO_SLOTS)
    frames = tmp_path / "frames"
    rows = []
    for folder in ("test/s1", ""):  # a session folder, and frames with no folder at all
        (frames / folder).mkdir(parents=True, exist_ok=True)
        for index in range(6):
            name = f"{folder}/{index:04d}.png".lstrip("/")
            car = index >= 3
            cv2.imwrite(str(frames / name), lot(car_in_left=car, seed=index))
            rows += [[name, "L", "occupied" if car else "vacant", "daytime", "0"], [name, "R", "vacant", "daytime", "0"]]
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([evaluate.LABEL_COLUMNS] + rows)
    fake_detector.detect = lambda frame, **kwargs: []  # misses every car
    monkeypatch.setattr(evaluate, "DEFAULT_CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--labels", str(labels), "--frames-dir", str(frames), "--predictions", str(tmp_path / "p.csv")])
    outcomes = evaluate.run_pipeline(evaluate.read_labels(labels, ["L", "R"]), TWO_SLOTS, frames, 0.4, 0.7, None, 0.8)
    results = evaluate.score(outcomes, ["L", "R"])
    assert results["empty_check"]["caught"] == 6  # 3 missed cars in each session
    assert results["empty_check"]["cost"] == 0
    assert results["empty_check"]["false_free_without"] == 6 and results["overall"]["false_free"] == 0
