"""The accuracy harness (eval/evaluate.py): label checks, scoring rules, reports, caching."""

import csv
import json
import sys

import cv2
import pytest
from conftest import TWO_SLOTS, blank_frame

import evaluate

L, O = evaluate.Label, evaluate.Outcome


def write_labels(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(evaluate.LABEL_COLUMNS)
        writer.writerows(rows)
    return path


# --- Label validation -----------------------------------------------------------


def test_bad_labels_are_reported_with_line_numbers(tmp_path):
    path = write_labels(
        tmp_path / "labels.csv",
        [
            ["a.png", "L", "busy", "daytime", "0"],
            ["a.png", "X", "vacant", "night", "2"],
            ["a.png", "L", "vacant", "evening", ""],
        ],
    )
    with pytest.raises(evaluate.LabelError) as error:
        evaluate.read_labels(path, ["L", "R"])
    message = str(error.value)
    for expected in ["line 2", "line 3", "line 4", "labeled twice", "both 'daytime' and 'evening'", "not in the slot config"]:
        assert expected in message


def test_empty_labels_file_is_reported(tmp_path):
    with pytest.raises(evaluate.LabelError, match="no labeled rows"):
        evaluate.read_labels(write_labels(tmp_path / "labels.csv", []), ["L"])


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("frame,slot_id\n", encoding="utf-8")
    with pytest.raises(evaluate.LabelError, match="missing columns"):
        evaluate.read_labels(path, ["L"])


def test_the_tracked_labels_template_has_the_right_header():
    with evaluate.DEFAULT_LABELS.open(newline="", encoding="utf-8") as file:
        assert next(csv.reader(file)) == evaluate.LABEL_COLUMNS


# --- Scoring rules (decided 2026-11-17) ---------------------------------------------

OUTCOMES = [
    O(L("f1", "L", "occupied", "daytime", False), "occupied", "occupied", 0.9),
    O(L("f1", "R", "vacant", "daytime", False), "vacant", "vacant", 0.95),
    O(L("f2", "L", "occupied", "daytime", True), "occupied", "occupied", 0.8),
    O(L("f2", "R", "vacant", "daytime", False), "unknown", "vacant", 0.5),
    O(L("f3", "L", "occupied", "evening", True), "vacant", "vacant", 0.9),
    O(L("f3", "R", "vacant", "evening", False), "vacant", "vacant", 0.99),
    O(L("f4", "L", "occupied", "rain", True), "unknown", "vacant", 0.4),
]


def test_unknown_counts_as_wrong_and_coverage_is_reported():
    overall = evaluate.score(OUTCOMES, ["L", "R"])["overall"]
    assert overall["accuracy"] == pytest.approx(4 / 7)
    assert overall["coverage"] == pytest.approx(5 / 7)
    assert overall["committed_accuracy"] == pytest.approx(4 / 5)
    assert overall["false_free"] == 1


def test_free_count_uses_only_fully_labeled_frames():
    free = evaluate.score(OUTCOMES, ["L", "R"])["free_count"]
    assert (free["frames"], free["skipped_frames"], free["matches"], free["over_reported"]) == (3, 1, 1, 1)


def test_condition_ratios_and_r05_verdicts():
    results = evaluate.score(OUTCOMES, ["L", "R"])
    assert results["condition_ratios"]["evening"]["ratio"] == pytest.approx(0.5 / 0.75)
    assert results["r05"] == "FAIL"
    assert evaluate.score(OUTCOMES[:2], ["L", "R"])["r05"] == "NO DATA"
    day_and_evening = OUTCOMES[:2] + [O(L("f9", "L", "occupied", "evening", False), "occupied", "occupied", 0.9)]
    assert evaluate.score(day_and_evening, ["L", "R"])["r05"] == "INCOMPLETE"


def test_r07_fails_only_on_wrong_confident_occluded_readings():
    results = evaluate.score(OUTCOMES, ["L", "R"])
    assert results["occluded"] == {"n": 3, "correct": 1, "unknown": 1, "wrong": 1, "false_free": 1, "verdict": "FAIL"}
    safe = [o for o in OUTCOMES if o.label.frame != "f3"]
    assert evaluate.score(safe, ["L", "R"])["r07"] == "PASS"


def test_review_threshold_effect_is_reported():
    assert evaluate.score(OUTCOMES, ["L", "R"])["low_confidence"] == {"n": 2, "raw_wrong": 1, "raw_false_free": 1}


def test_report_renders_the_verdict_table():
    report = evaluate.render_report(
        evaluate.score(OUTCOMES, ["L", "R"]),
        {"labels": "labels.csv", "note": "", "slot_ids": ["L", "R"], "model": "m", "overlap": 0.4, "review": 0.7},
    )
    assert "| R-01 | Per-slot accuracy | 57.1% (4/7) | >= 90% | **FAIL** (small sample: 7 rows) |" in report


def test_runs_are_added_to_the_accuracy_section_before_r04(tmp_path):
    doc = tmp_path / "evaluation.md"
    doc.write_text("# SlotSight Evaluation\n\n## R-04: Update interval\n\n### R-04 timing run\n", encoding="utf-8")
    evaluate.append_to_evaluation("\n### Accuracy run: A\n", doc)
    evaluate.append_to_evaluation("\n### Accuracy run: B\n", doc)
    text = doc.read_text(encoding="utf-8")
    assert text.index(evaluate.ACCURACY_SECTION) < text.index("run: A") < text.index("run: B") < text.index("## R-04")


# --- End to end with a fake detector ---------------------------------------------------


def test_end_to_end_run_scores_writes_predictions_and_caches(app, fake_detector, tmp_path, monkeypatch):
    app.save_slots(TWO_SLOTS)
    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "car.png"), blank_frame())
    cv2.imwrite(str(frames / "empty.png"), blank_frame(90))
    labels = write_labels(
        tmp_path / "labels.csv",
        [
            ["car.png", "L", "occupied", "daytime", "0"],
            ["car.png", "R", "vacant", "daytime", "0"],
            ["empty.png", "L", "vacant", "evening", "0"],
            ["empty.png", "R", "vacant", "evening", "1"],
        ],
    )
    boxes = {128: [(20, 20, 300, 460, 0.9)], 90: []}
    fake_detector.detect = lambda frame, **kwargs: boxes[int(frame[0, 0, 0])]
    cache, predictions = tmp_path / "cache.json", tmp_path / "pred.csv"
    monkeypatch.setattr(evaluate, "DEFAULT_CACHE", cache)
    args = ["evaluate.py", "--labels", str(labels), "--frames-dir", str(frames), "--predictions", str(predictions)]
    monkeypatch.setattr(sys, "argv", args)
    assert evaluate.main() == 0
    with predictions.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4 and all(row["correct"] == "1" for row in rows)
    assert len(json.loads(cache.read_text(encoding="utf-8"))) == 2

    # A second run with another threshold reuses cached detections: no inference at all.
    fake_detector.detect = lambda frame, **kwargs: pytest.fail("detector called despite the cache")
    monkeypatch.setattr(sys, "argv", args + ["--review-threshold", "0.95"])
    assert evaluate.main() == 2  # R-01 now fails: the 0.9 detection shows as unknown


def test_missing_frames_are_reported(app, fake_detector, tmp_path, monkeypatch, capsys):
    app.save_slots(TWO_SLOTS)
    labels = write_labels(tmp_path / "labels.csv", [["gone.png", "L", "vacant", "daytime", "0"]])
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--labels", str(labels), "--frames-dir", str(tmp_path)])
    assert evaluate.main() == 1
    assert "not found" in capsys.readouterr().err
