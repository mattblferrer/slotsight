"""Flask API: dashboard states, uploads, history writes, config validation, video jobs, models."""

import time

import cv2
import numpy as np
from conftest import blank_frame, history_rows, png_upload, wait_until

import detectors

# --- What the dashboard shows without a reading ---------------------------------


def test_no_model_shows_labelled_demo_data(client, app):
    status = client.get("/api/status").get_json()
    assert status["source"] == "demo"
    assert status["model"] == "demo classifier"
    assert status["model_status"]["state"] == "missing"


def test_model_ready_but_no_frame_shows_unknown(client, fake_detector):
    status = client.get("/api/status").get_json()
    assert status["source"] == "none"
    assert {slot["status"] for slot in status["slots"]} == {"unknown"}
    assert status["updated_at"] is None


def test_model_downloading_shows_unknown_not_demo(client, app, monkeypatch):
    monkeypatch.setattr(app.models, "status", lambda model_id: {"id": model_id, "name": "X", "state": "downloading", "message": "", "progress": 0.5})
    status = client.get("/api/status").get_json()
    assert status["source"] == "none"
    assert {slot["status"] for slot in status["slots"]} == {"unknown"}


def test_status_polls_never_write_history(client, fake_detector, two_slots, app):
    for _ in range(2):
        client.get("/api/status")
    assert history_rows(app) == []
    # And after a real reading exists, polling still only reads it.
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    rows = len(history_rows(app))
    for _ in range(3):
        client.get("/api/status")
        client.get("/api/models")
        client.get("/api/source")
    assert len(history_rows(app)) == rows


# --- Processing an image ----------------------------------------------------------


def test_blank_frame_gives_all_vacant_and_logs_one_row_per_slot(client, fake_detector, two_slots, app):
    response = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    data = response.get_json()
    assert response.status_code == 200
    assert {slot["status"] for slot in data["slots"]} == {"vacant"}
    assert data["free"] == 2
    rows = history_rows(app)
    assert [row["slot_id"] for row in rows] == ["L", "R"]
    assert all(row["status"] == "vacant" and row["source"] == "frame.png" for row in rows)


def test_detected_vehicle_marks_its_slot_occupied(client, fake_detector, two_slots):
    fake_detector.boxes = [(20, 20, 300, 460, 0.9)]
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    statuses = {slot["id"]: slot["status"] for slot in data["slots"]}
    assert statuses == {"L": "occupied", "R": "vacant"}
    assert data["free"] == 1


def test_each_slot_carries_the_reading_time(client, fake_detector, two_slots):
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    assert data["updated_at"]
    assert all(slot["updated_at"] == data["updated_at"] for slot in data["slots"])


def test_recorded_at_sets_the_history_timestamp(client, fake_detector, two_slots, app):
    upload = {**png_upload(blank_frame()), "recorded_at": "2026-11-17T08:30:00+00:00"}
    client.post("/api/process", data=upload, content_type="multipart/form-data")
    assert {row["timestamp"] for row in history_rows(app)} == {"2026-11-17T08:30:00+00:00"}


def test_detector_error_is_an_error_not_demo_data(client, fake_detector, two_slots, app):
    fake_detector.error = RuntimeError("boom")
    response = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    assert response.status_code == 500
    assert "boom" in response.get_json()["error"]
    assert history_rows(app) == []


def test_upload_without_a_usable_model_returns_503_and_logs_nothing(client, two_slots, app):
    response = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    assert response.status_code == 503
    assert "not ready" in response.get_json()["error"]
    assert history_rows(app) == []


def test_unreadable_upload_is_rejected(client, fake_detector, two_slots):
    response = client.post(
        "/api/process", data={"source": (__import__("io").BytesIO(b"not media"), "x.bin")}, content_type="multipart/form-data"
    )
    assert response.status_code == 400


def test_bad_recorded_at_is_rejected(client, fake_detector, two_slots):
    upload = {**png_upload(blank_frame()), "recorded_at": "yesterday"}
    assert client.post("/api/process", data=upload, content_type="multipart/form-data").status_code == 400


def test_post_without_a_file_logs_nothing(client, fake_detector, two_slots, app):
    assert client.post("/api/process").status_code == 200
    assert history_rows(app) == []


# --- Re-classification keeps the reading time and logs nothing ---------------------


def test_config_and_threshold_changes_keep_the_reading_time(client, fake_detector, two_slots, app):
    first = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    rows = len(history_rows(app))
    time.sleep(1.1)  # timestamps have one-second resolution
    client.post("/api/config", json={"slots": two_slots, "lot_name": "Test", "review_threshold": 0.7})
    client.post("/api/review-threshold", json={"review_threshold": 0.6})
    after = client.get("/api/status").get_json()
    assert after["updated_at"] == first["updated_at"]
    assert len(history_rows(app)) == rows
    assert fake_detector.calls == 1  # cached detections are reused


# --- Config validation --------------------------------------------------------------


def test_config_rejects_bad_payloads(client, app):
    good = [{"id": "A", "points": [[0, 0], [10, 0], [10, 10]]}]
    cases = [
        {"lot_name": "x"},  # no slots
        {"slots": [{"id": "A", "points": [[0, 0], [1, 1]]}]},  # two points
        {"slots": [{"id": "", "points": good[0]["points"]}]},  # empty id
        {"slots": [{"id": "A", "points": [[0, 0], [10, 0], [10, 101]]}]},  # out of range
        {"slots": good, "lot_name": ""},
        {"slots": good, "lot_name": "x" * 81},
        {"slots": good, "review_threshold": 1.5},
        {"slots": good, "review_threshold": "high"},
    ]
    for payload in cases:
        assert client.post("/api/config", json=payload).status_code == 400, payload


def test_config_saves_slots_and_settings(client, app):
    slots = [{"id": "A", "points": [[0, 0], [10, 0], [10, 10]]}]
    response = client.post("/api/config", json={"slots": slots, "lot_name": "  North  ", "review_threshold": 0.6})
    assert response.status_code == 200
    assert app.load_slots() == slots
    settings = app.load_settings()
    assert settings["lot_name"] == "North" and settings["review_threshold"] == 0.6


def test_review_threshold_validation(client, app):
    assert client.post("/api/review-threshold", json={"review_threshold": -0.1}).status_code == 400
    assert client.post("/api/review-threshold", json={"review_threshold": "x"}).status_code == 400
    assert client.post("/api/review-threshold", json={"review_threshold": 0.5}).status_code == 200


# --- Video ---------------------------------------------------------------------------


def write_video(path, seconds=6, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    assert writer.isOpened()
    for index in range(seconds * fps):
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.putText(frame, str(index), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_video_upload_logs_one_set_of_rows_per_sampled_frame(client, fake_detector, two_slots, app, tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video)  # 6 s at 10 fps, sampled every 2 s: frames at 0, 2 and 4 s
    with video.open("rb") as handle:
        response = client.post(
            "/api/process",
            data={"source": (handle, "clip.mp4"), "recorded_at": "2026-11-17T08:00:00+00:00"},
            content_type="multipart/form-data",
        )
    assert response.status_code == 202
    assert wait_until(lambda: client.get("/api/source").get_json()["job"]["state"] == "finished")
    rows = history_rows(app)
    assert len(rows) == 3 * len(two_slots)
    assert sorted({row["frame_time_s"] for row in rows}) == ["0.00", "2.00", "4.00"]
    assert sorted({row["timestamp"] for row in rows}) == [
        "2026-11-17T08:00:00+00:00",
        "2026-11-17T08:00:02+00:00",
        "2026-11-17T08:00:04+00:00",
    ]


def test_video_upload_without_a_model_is_refused(client, two_slots, app, tmp_path):
    video = tmp_path / "clip.mp4"
    write_video(video, seconds=1)
    with video.open("rb") as handle:
        response = client.post("/api/process", data={"source": (handle, "clip.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 503
    assert history_rows(app) == []


def test_live_source_rejects_an_empty_target(client, fake_detector):
    assert client.post("/api/source", json={"target": ""}).status_code == 400


# --- Model endpoints -----------------------------------------------------------------


def test_models_catalogue_lists_every_provider(client, app):
    data = client.get("/api/models").get_json()
    providers = {row["provider"] for row in data["models"]}
    assert providers == {provider.name for provider in detectors.PROVIDERS.values()}
    assert data["active"] == app.DEFAULT_MODEL
    assert [row["id"] for row in data["models"] if row["active"]] == [app.DEFAULT_MODEL]


def test_install_unknown_model_is_404(client):
    assert client.post("/api/models/install", json={"id": "nope/x"}).status_code == 404


def test_select_requires_an_installed_model(client, app):
    response = client.post("/api/models/select", json={"id": "ultralytics/yolo11m"})
    assert response.status_code == 409
    assert app.load_settings()["model"] == app.DEFAULT_MODEL


def test_select_is_refused_while_slotsight_model_is_set(client, monkeypatch):
    monkeypatch.setenv("SLOTSIGHT_MODEL", "ultralytics/yolo11n")
    assert client.post("/api/models/select", json={"id": "ultralytics/yolo11s"}).status_code == 409


def test_select_saves_an_installed_model(client, app, monkeypatch):
    spec = detectors.MODELS["opencv/yolox-s"]
    target = detectors.file_path(app.MODELS_DIR, spec.files[0])
    target.parent.mkdir(parents=True)
    target.write_bytes(b"placeholder")  # installed as far as the file check goes
    monkeypatch.setattr(app.models, "start", lambda model_id, download, load: {"id": model_id, "action": "load", "state": "running"})
    response = client.post("/api/models/select", json={"id": "opencv/yolox-s"})
    assert response.status_code == 200
    assert app.load_settings()["model"] == "opencv/yolox-s"
