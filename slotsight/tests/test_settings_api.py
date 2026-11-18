"""POST /api/settings: the Settings dialog saves each control as it changes, without
re-sending (or re-saving) the slot layout. Decided 2026-11-18 (instant apply)."""

import time

import pytest
from conftest import blank_frame, history_rows, png_upload
from test_classifier import save_random_weights


def test_settings_save_only_what_is_sent(client, app, two_slots):
    before = app.load_settings()
    slots_mtime = app.CONFIG_PATH.stat().st_mtime_ns
    response = client.post("/api/settings", json={"lot_name": "  North lot  "})
    assert response.status_code == 200
    assert response.get_json()["lot_name"] == "North lot"
    after = app.load_settings()
    assert after["lot_name"] == "North lot"
    assert {k: v for k, v in after.items() if k != "lot_name"} == {k: v for k, v in before.items() if k != "lot_name"}
    # The layout is not re-saved, so a settings change can never reset the camera reference.
    assert app.CONFIG_PATH.stat().st_mtime_ns == slots_mtime
    assert app.load_slots() == two_slots
    assert not app.REFERENCE_PATH.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"lot_name": ""},
        {"lot_name": "x" * 81},
        {"review_threshold": 1.5},
        {"review_threshold": "high"},
        {"inference_size": 1000},
        {"slot_classifier": "yes"},
        {"lot_name": "Fine", "inference_size": 7},  # one bad value: nothing is saved
    ],
)
def test_settings_reject_bad_values_and_save_nothing(client, app, payload):
    before = app.load_settings()
    assert client.post("/api/settings", json=payload).status_code == 400
    assert app.load_settings() == before


def test_settings_reclassify_the_latest_frame_without_logging(client, app, fake_detector, two_slots):
    first = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    rows = len(history_rows(app))
    time.sleep(1.1)  # timestamps have one-second resolution
    data = client.post("/api/settings", json={"review_threshold": 0.6, "inference_size": 960}).get_json()
    assert data["review_threshold"] == 0.6 and data["inference_size"] == 960
    assert data["updated_at"] == first["updated_at"]
    assert len(history_rows(app)) == rows
    assert fake_detector.calls == 1  # cached detections are reused


def test_switching_the_classifier_off_stops_it_deciding_the_latest_frame(client, app, fake_detector, two_slots, monkeypatch):
    app.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_random_weights(app.MODELS_DIR)
    monkeypatch.setattr(app.slot_classifier, "predict", lambda frame, slots: [(0.05, 0.95), (0.98, 0.02)])
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    assert all(s["method"] == "classifier" for s in data["slots"])

    off = client.post("/api/settings", json={"slot_classifier": False}).get_json()
    assert all(s["method"] == "detector" for s in off["slots"])
    assert not off["classifier"]["enabled"]

    on = client.post("/api/settings", json={"slot_classifier": True}).get_json()
    assert all(s["method"] == "classifier" for s in on["slots"])


def test_config_still_accepts_settings_with_the_layout(client, app, two_slots):
    payload = {"slots": two_slots, "lot_name": " Lot ", "inference_size": 1280, "slot_classifier": False}
    assert client.post("/api/config", json=payload).status_code == 200
    settings = app.load_settings()
    assert settings["lot_name"] == "Lot" and settings["inference_size"] == 1280 and settings["slot_classifier"] is False
    # Only the layout: every setting is kept.
    assert client.post("/api/config", json={"slots": two_slots}).status_code == 200
    assert app.load_settings() == settings
