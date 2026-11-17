"""The per-slot classifier (classifier.py), its combination with the detector, and training."""

import csv
import json
import sys

import cv2
import numpy as np
import pytest
from conftest import TWO_SLOTS, blank_frame, png_upload

import classifier
import train_classifier

BASE = classifier.BASE_WEIGHTS


def detector_result(status, confidence=0.9, slot_id="L"):
    return {"id": slot_id, "status": status, "confidence": confidence}


@pytest.mark.parametrize(
    ("p_occupied", "detector", "expected", "reason"),
    [
        (0.95, ("vacant", 1.0), "occupied", None),  # the classifier sees a car the detector missed
        (0.05, ("vacant", 1.0), "vacant", None),  # both agree it's free
        (0.05, ("occupied", 0.9), "unknown", "detector sees a vehicle"),  # disagree: never free
        (0.05, ("vacant", 0.5), "unknown", "detector sees a vehicle"),  # detector unsure: never free
        (0.55, ("vacant", 1.0), "unknown", "classifier unsure"),  # below the review threshold
    ],
)
def test_classifier_decides_and_the_detector_votes_for_vacant(p_occupied, detector, expected, reason):
    result = classifier.combine([detector_result(*detector)], [(1 - p_occupied, p_occupied)], 0.7)[0]
    assert result["status"] == expected and result.get("reason") == reason
    assert result["method"] == "classifier"


def test_without_a_classifier_the_detector_decides():
    results = classifier.combine([detector_result("occupied", 0.9), detector_result("vacant", 0.4, "R")], None, 0.7)
    assert [r["status"] for r in results] == ["occupied", "unknown"]
    assert {r["method"] for r in results} == {"detector"}


def test_warp_straightens_a_slot_to_a_square_patch():
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :320] = 200  # slot L is bright, slot R dark
    left = classifier.warp_slot(frame, TWO_SLOTS[0]["points"])
    assert left.shape == (classifier.PATCH, classifier.PATCH, 3) and left.mean() > 150
    assert classifier.warp_slot(frame, TWO_SLOTS[1]["points"]).mean() < 50
    triangle = classifier.warp_slot(frame, [[10, 10], [40, 10], [25, 60]])
    assert triangle.shape == (classifier.PATCH, classifier.PATCH, 3)


def test_missing_weights_mean_no_classifier(tmp_path):
    slot_classifier = classifier.SlotClassifier(tmp_path)
    assert slot_classifier.predict(blank_frame(), TWO_SLOTS) is None
    assert slot_classifier.status()["state"] == "missing"


def test_the_committed_base_weights_load_and_predict():
    from pathlib import Path

    models = Path(classifier.__file__).resolve().parent / "models"
    assert (models / BASE).exists(), "the base weights are tracked in git"
    slot_classifier = classifier.SlotClassifier(models)
    probabilities = slot_classifier.predict(blank_frame(), TWO_SLOTS)
    assert len(probabilities) == 2 and all(abs(sum(p) - 1) < 1e-4 for p in probabilities)
    info = json.loads((models / "slot_classifier_base.json").read_text(encoding="utf-8"))
    import re

    assert "PKLot" in info["dataset"]
    # No local paths (drive letters, backslashes or user folders) in the committed metadata.
    assert not re.search(r"[A-Za-z]:[\/]|\|Users", json.dumps(info))


def test_a_corrupt_weights_file_falls_back_with_an_error(tmp_path):
    (tmp_path / BASE).write_bytes(b"not a model")
    slot_classifier = classifier.SlotClassifier(tmp_path)
    assert slot_classifier.predict(blank_frame(), TWO_SLOTS) is None
    assert slot_classifier.status()["state"] == "error"


def save_random_weights(folder):
    import torch

    torch.manual_seed(1)
    torch.save(classifier.build_network().state_dict(), folder / BASE)


def test_the_dashboard_uses_the_classifier_when_weights_exist(client, fake_detector, two_slots, app, monkeypatch):
    app.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_random_weights(app.MODELS_DIR)
    monkeypatch.setattr(app.slot_classifier, "predict", lambda frame, slots: [(0.05, 0.95), (0.98, 0.02)])
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    assert {s["id"]: s["status"] for s in data["slots"]} == {"L": "occupied", "R": "vacant"}
    assert data["classifier"]["enabled"] and data["classifier"]["state"] == "ready"
    assert all(s["method"] == "classifier" for s in data["slots"])


def test_the_classifier_can_be_switched_off(client, fake_detector, two_slots, app, monkeypatch):
    monkeypatch.setattr(app.slot_classifier, "predict", lambda frame, slots: pytest.fail("classifier used while switched off"))
    payload = {"slots": two_slots, "lot_name": "Lot", "review_threshold": 0.7, "slot_classifier": False}
    assert client.post("/api/config", json=payload).status_code == 200
    data = client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()
    assert all(s["method"] == "detector" for s in data["slots"])
    assert client.post("/api/config", json={**payload, "slot_classifier": "yes"}).status_code == 400


def make_dataset(folder, frames=8):
    """Slot L holds a bright 'car' in half the frames; R is always empty asphalt."""
    (folder / "frames").mkdir(parents=True)
    (folder / "slots.json").write_text(json.dumps(TWO_SLOTS), encoding="utf-8")
    rows = []
    for index in range(frames):
        image = np.full((480, 640, 3), 90, np.uint8)
        occupied = index % 2 == 0
        if occupied:
            cv2.rectangle(image, (60, 80), (260, 400), (220, 220, 230), -1)
        name = f"{index:03d}.png"
        cv2.imwrite(str(folder / "frames" / name), image)
        rows += [[name, "L", "occupied" if occupied else "vacant", "daytime", "0"], [name, "R", "vacant", "daytime", "0"]]
    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([["frame", "slot_id", "true_status", "condition", "occluded"]] + rows)


def test_training_from_a_dataset_writes_weights_and_metadata(tmp_path, monkeypatch):
    make_dataset(tmp_path / "lotA")
    out = tmp_path / "out" / "weights.pt"
    monkeypatch.setattr(sys, "argv", ["train_classifier.py", "--from-scratch", "--dataset", str(tmp_path / "lotA"), "--epochs", "2", "--out", str(out)])
    assert train_classifier.main() == 0
    info = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert info["sources"] == ["lotA"] and info["frames"] + info["validation_frames"] == 8
    assert classifier.SlotClassifier(out.parent).weights_path() is None  # custom name: not picked up by the app
    assert out.exists()


def test_fine_tuning_uses_only_tune_sessions(app, tmp_path, monkeypatch):
    app.save_slots(TWO_SLOTS)
    image = np.full((480, 640, 3), 90, np.uint8)
    for frame in ("tune/s1/000.png", "test/s2/000.png"):
        path = app.EVAL_FRAMES_DIR / frame
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    app.LABELS_PATH.parent.mkdir(parents=True)
    with app.LABELS_PATH.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([
            ["frame", "slot_id", "true_status", "condition", "occluded"],
            ["tune/s1/000.png", "L", "vacant", "daytime", "0"],
            ["test/s2/000.png", "L", "vacant", "daytime", "0"],
        ])
    frames = train_classifier.load_set(app.load_slots(), app.LABELS_PATH, app.EVAL_FRAMES_DIR, ["tune/*"], [])
    assert len(frames) == 1  # the test session is never trained on
