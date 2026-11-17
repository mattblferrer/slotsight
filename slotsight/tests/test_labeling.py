"""Frame extraction, hand labeling (/label) and the tune/test split in the evaluation."""

import csv
import json
import sys

import cv2
import numpy as np
import pytest
from conftest import TWO_SLOTS

import evaluate
import extract_frames
import labeling


def write_video(path, seconds, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for index in range(seconds * fps):
        frame = np.full((240, 320, 3), 100, np.uint8)
        cv2.putText(frame, str(index), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return path


def save_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))[1].tofile(str(path))
    return path


# --- Extracting frames -------------------------------------------------------------------


def test_extract_takes_one_frame_every_n_seconds(tmp_path):
    video = write_video(tmp_path / "clip.mp4", seconds=65)
    written = extract_frames.extract(video, tmp_path / "out", every_s=30)
    assert [path.name for path in written] == ["0000000.0s.jpg", "0000030.0s.jpg", "0000060.0s.jpg"]


def test_extract_cli_writes_a_session_and_refuses_to_split_it(tmp_path, monkeypatch, capsys):
    video = write_video(tmp_path / "2026-11-20 morning.mp4", seconds=25)
    frames = tmp_path / "frames"
    run = lambda *extra: (monkeypatch.setattr(sys, "argv", ["extract_frames.py", str(video), "--frames-dir", str(frames), "--every", "10", *extra]), extract_frames.main())[1]  # noqa: E731
    assert run("--condition", "daytime") == 0
    session = frames / "test" / "2026-11-20-morning"
    assert len(list(session.glob("*.jpg"))) == 3
    info = json.loads((session / labeling.SESSION_FILE).read_text(encoding="utf-8"))
    assert info["condition"] == "daytime" and info["set"] == "test" and info["frames"] == 3
    assert run() == 1  # already has frames
    assert run("--set", "tune") == 1  # the same session can't also go into tune
    assert "must not be split" in capsys.readouterr().err


# --- Labels file -----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../secret.jpg", "test/../../x.jpg", "C:/x.jpg", "/etc/x.jpg", "a\\b.jpg", "test/notes.txt", ""])
def test_frame_paths_cannot_leave_the_frames_folder(tmp_path, bad):
    (tmp_path / "frames").mkdir()
    save_image(tmp_path / "secret.jpg")
    with pytest.raises(labeling.LabelingError):
        labeling.resolve_frame(tmp_path / "frames", bad)


def test_saving_replaces_one_frame_and_keeps_the_others(tmp_path):
    labels = tmp_path / "labels.csv"
    ids = ["L", "R"]
    labeling.save_frame(labels, "test/s/b.jpg", "evening", [{"slot_id": "R", "true_status": "vacant"}], ids)
    labeling.save_frame(labels, "test/s/a.jpg", "daytime", [{"slot_id": "R", "true_status": "vacant"}, {"slot_id": "L", "true_status": "occupied", "occluded": True}], ids)
    labeling.save_frame(labels, "test/s/a.jpg", "daytime", [{"slot_id": "L", "true_status": "vacant"}], ids)
    with labels.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == labeling.LABEL_COLUMNS
    assert [(r["frame"], r["slot_id"], r["true_status"], r["occluded"]) for r in rows] == [
        ("test/s/a.jpg", "L", "vacant", "0"),
        ("test/s/b.jpg", "R", "vacant", "0"),
    ]
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("condition", "slots", "message"),
    [
        ("night", [], "Condition"),
        ("daytime", [{"slot_id": "X", "true_status": "vacant"}], "not in the current layout"),
        ("daytime", [{"slot_id": "L", "true_status": "busy"}], "occupied or vacant"),
        ("daytime", [{"slot_id": "L", "true_status": "vacant"}, {"slot_id": "L", "true_status": "vacant"}], "twice"),
    ],
)
def test_invalid_labels_are_refused(tmp_path, condition, slots, message):
    with pytest.raises(labeling.LabelingError, match=message):
        labeling.save_frame(tmp_path / "labels.csv", "test/s/a.jpg", condition, slots, ["L", "R"])
    assert not (tmp_path / "labels.csv").exists()


def test_frame_list_counts_labels_and_uses_the_session_condition(tmp_path):
    frames, labels = tmp_path / "frames", tmp_path / "labels.csv"
    save_image(frames / "test" / "s" / "a.jpg")
    save_image(frames / "test" / "s" / "b.jpg")
    (frames / "test" / "s" / labeling.SESSION_FILE).write_text(json.dumps({"condition": "rain"}), encoding="utf-8")
    labeling.save_frame(labels, "test/s/a.jpg", "daytime", [{"slot_id": "L", "true_status": "vacant"}, {"slot_id": "R", "true_status": "vacant"}], ["L", "R"])
    listing = {item["frame"]: item for item in labeling.list_frames(frames, labels, ["L", "R"])}
    assert listing["test/s/a.jpg"]["complete"] and listing["test/s/a.jpg"]["condition"] == "daytime"
    assert listing["test/s/b.jpg"]["labeled"] == 0 and listing["test/s/b.jpg"]["condition"] == "rain"


def test_evaluate_and_labeling_agree_on_the_format():
    assert evaluate.LABEL_COLUMNS == labeling.LABEL_COLUMNS


# --- /label API ------------------------------------------------------------------------------------


def test_label_api_lists_serves_and_saves_without_touching_real_files(client, app):
    app.save_slots(TWO_SLOTS)
    save_image(app.EVAL_FRAMES_DIR / "test" / "s" / "a.jpg")
    assert client.get("/label").status_code == 200
    listing = client.get("/api/label/frames").get_json()
    assert [frame["frame"] for frame in listing["frames"]] == ["test/s/a.jpg"]
    assert [slot["id"] for slot in listing["slots"]] == ["L", "R"]
    assert client.get("/api/label/image?frame=test/s/a.jpg").status_code == 200
    assert client.get("/api/label/image?frame=../../config/slots.json").status_code == 404
    assert not app.LABELS_PATH.exists()  # GETs never write

    response = client.post("/api/label/frame", json={"frame": "test/s/a.jpg", "condition": "evening", "slots": [{"slot_id": "L", "true_status": "occupied"}]})
    assert response.status_code == 200
    assert "1 slot(s) left unset" in response.get_json()["message"]
    assert client.get("/api/label/frame?frame=test/s/a.jpg").get_json()["labels"][0]["true_status"] == "occupied"
    assert str(app.LABELS_PATH).startswith(str(app.EVAL_FRAMES_DIR.parent.parent))  # the temporary folder
    assert client.post("/api/label/frame", json={"frame": "test/s/missing.jpg", "condition": "evening", "slots": []}).status_code == 400


# --- Tune/test split in the evaluation ----------------------------------------------------------------


def labels(*frames):
    return [evaluate.Label(frame, "L", "vacant", "daytime", False) for frame in frames]


def test_only_and_exclude_select_sessions():
    all_labels = labels("test/a/1.jpg", "test/b/1.jpg", "tune/c/1.jpg")
    assert [l.frame for l in evaluate.select_frames(all_labels, [], ["tune/*"])] == ["test/a/1.jpg", "test/b/1.jpg"]
    assert [l.frame for l in evaluate.select_frames(all_labels, ["tune/*"], [])] == ["tune/c/1.jpg"]
    assert [l.frame for l in evaluate.select_frames(all_labels, ["test/*"], ["test/b/*"])] == ["test/a/1.jpg"]
    with pytest.raises(evaluate.LabelError, match="No labeled frames"):
        evaluate.select_frames(all_labels, ["nothing/*"], [])
