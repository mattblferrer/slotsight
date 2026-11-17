"""First-run config files, settings migration and model ids."""

import json

import pytest

import detectors


def test_first_run_creates_both_config_files_from_defaults(app):
    slots = app.load_slots()
    settings = app.load_settings()
    assert app.CONFIG_PATH.exists() and app.SETTINGS_PATH.exists()
    assert slots == app.DEFAULT_SLOTS
    assert settings == {
        "lot_name": app.DEFAULT_LOT_NAME,
        "review_threshold": app.DEFAULT_REVIEW_THRESHOLD,
        "overlap_threshold": app.DEFAULT_OVERLAP_THRESHOLD,
        "sample_interval_s": app.DEFAULT_SAMPLE_INTERVAL_S,
        "inference_size": 640,
        "empty_check_threshold": 0.8,
        "model": app.DEFAULT_MODEL,
    }


def test_default_slots_are_not_shared_mutable_state(app):
    app.load_slots()
    app.CONFIG_PATH.unlink()
    first = app.load_slots()
    first[0]["id"] = "changed"
    assert app.DEFAULT_SLOTS[0]["id"] == "A01"


def test_unreadable_slots_file_is_left_alone(app):
    app.CONFIG_PATH.parent.mkdir(parents=True)
    app.CONFIG_PATH.write_text("{not json", encoding="utf-8")
    assert app.load_slots() == app.DEFAULT_SLOTS
    assert app.CONFIG_PATH.read_text(encoding="utf-8") == "{not json"


def test_legacy_model_path_migrates_and_keeps_other_settings(app):
    app.SETTINGS_PATH.parent.mkdir(parents=True)
    legacy = {"lot_name": "My lot", "review_threshold": 0.55, "overlap_threshold": 0.3, "sample_interval_s": 3, "model_path": "yolo11s.pt"}
    app.SETTINGS_PATH.write_text(json.dumps(legacy), encoding="utf-8")
    settings = app.load_settings()
    written = json.loads(app.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert settings["model"] == "ultralytics/yolo11s"
    assert "model_path" not in written
    assert (written["lot_name"], written["review_threshold"], written["overlap_threshold"]) == ("My lot", 0.55, 0.3)


def test_settings_are_not_rewritten_when_unchanged(app):
    app.load_settings()
    before = app.SETTINGS_PATH.stat().st_mtime_ns
    app.load_settings()
    assert app.SETTINGS_PATH.stat().st_mtime_ns == before


def test_out_of_range_values_are_clamped(app):
    app.SETTINGS_PATH.parent.mkdir(parents=True)
    app.SETTINGS_PATH.write_text(json.dumps({"review_threshold": 5, "overlap_threshold": "x", "sample_interval_s": 0}), encoding="utf-8")
    settings = app.load_settings()
    assert settings["review_threshold"] == 1.0
    assert settings["overlap_threshold"] == app.DEFAULT_OVERLAP_THRESHOLD
    assert settings["sample_interval_s"] == app.SAMPLE_INTERVAL_RANGE_S[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", detectors.DEFAULT_MODEL_ID),
        ("yolo11s.pt", "ultralytics/yolo11s"),
        ("models/yolo11n.pt", "ultralytics/yolo11n"),
        ("my.pt", "ultralytics/file:my.pt"),
        ("models/my.pt", "ultralytics/file:my.pt"),
        ("huggingface/rtdetr-v2-r18", "huggingface/rtdetr-v2-r18"),
        ("ultralytics/file:x.pt", "ultralytics/file:x.pt"),
    ],
)
def test_normalize_model_id(value, expected):
    assert detectors.normalize_model_id(value) == expected


def test_slotsight_model_env_overrides_settings(app, monkeypatch):
    monkeypatch.setenv("SLOTSIGHT_MODEL", "yolo11n.pt")
    assert app.configured_model_id() == "ultralytics/yolo11n"


def test_every_registry_download_has_a_hash():
    for spec in detectors.MODELS.values():
        for file in spec.files:
            assert file.url.startswith("https://"), spec.id
            assert file.sha256 or file.md5, f"{spec.id} {file.path} has no hash"


def test_default_model_is_in_the_registry():
    assert detectors.DEFAULT_MODEL_ID in detectors.MODELS
