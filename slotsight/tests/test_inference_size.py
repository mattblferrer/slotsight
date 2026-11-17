"""The detection resolution setting (inference_size)."""

import json

import pytest
from conftest import TWO_SLOTS, blank_frame, png_upload

import detectors
import evaluate


@pytest.mark.parametrize(("value", "expected"), [(None, 640), ("960", 960), (1280, 1280), (1000, 640), ("big", 640)])
def test_inference_size_is_one_of_the_allowed_sizes(app, value, expected):
    app.SETTINGS_PATH.parent.mkdir(parents=True)
    app.SETTINGS_PATH.write_text(json.dumps({"inference_size": value}), encoding="utf-8")
    assert app.load_settings()["inference_size"] == expected


def test_the_setting_reaches_the_detector(client, fake_detector, two_slots, app):
    app.save_settings(inference_size=1280)
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    assert fake_detector.last_inference_size == 1280
    assert client.get("/api/status").get_json()["inference_size"] == 1280


def test_detectors_without_the_option_are_called_without_it(client, fake_detector, two_slots, app):
    fake_detector.supports_inference_size = False
    fake_detector.detect = lambda frame: []  # would fail if called with inference_size
    assert client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").status_code == 200


def test_config_saves_and_validates_the_resolution(client, app):
    payload = {"slots": TWO_SLOTS, "lot_name": "Lot", "review_threshold": 0.7}
    assert client.post("/api/config", json={**payload, "inference_size": 1000}).status_code == 400
    assert client.post("/api/config", json={**payload, "inference_size": 960}).status_code == 200
    assert app.load_settings()["inference_size"] == 960
    assert client.post("/api/config", json=payload).status_code == 200  # omitted: keeps the current value
    assert app.load_settings()["inference_size"] == 960


def test_only_ultralytics_models_take_the_size():
    assert detectors.UltralyticsDetector.supports_inference_size
    assert not detectors.RFDETRDetector.supports_inference_size
    assert not detectors.TransformersDetector.supports_inference_size
    assert not detectors.YoloxDetector.supports_inference_size


def test_evaluation_cache_keys_include_the_size(app, tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(b"x")
    assert evaluate.cache_key("m", image, 640) != evaluate.cache_key("m", image, 1280)
