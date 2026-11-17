"""Slot classification: overlap ratio, occupied/vacant decisions and confidence (R-01, R-07)."""

import numpy as np
import pytest
from conftest import TWO_SLOTS, blank_frame

import app as slotsight


def hull(slot, width=640, height=480):
    import cv2

    return cv2.convexHull(slotsight.slot_polygon_pixels(slot, width, height)).reshape(-1, 2)


def classify(boxes, overlap_threshold=0.4):
    return {r["id"]: r for r in slotsight.classify_frame(blank_frame(), TWO_SLOTS, boxes, overlap_threshold)}


def test_overlap_ratio_full_and_none():
    left = hull(TWO_SLOTS[0])
    assert slotsight.overlap_ratio((0, 0, 320, 480), left) == pytest.approx(1.0)
    assert slotsight.overlap_ratio((400, 0, 600, 480), left) == 0.0


def test_overlap_ratio_uses_the_larger_share():
    # A small box fully inside a large slot covers little of the slot but is all inside it.
    left = hull(TWO_SLOTS[0])
    assert slotsight.overlap_ratio((10, 10, 60, 60), left) == pytest.approx(1.0)


def test_overlap_ratio_degenerate_box():
    assert slotsight.overlap_ratio((10, 10, 10, 50), hull(TWO_SLOTS[0])) == 0.0


def test_no_detections_means_every_slot_vacant_with_full_confidence():
    results = classify([])
    assert {r["status"] for r in results.values()} == {"vacant"}
    assert all(r["confidence"] == 1.0 for r in results.values())


def test_vehicle_in_slot_is_occupied_with_detection_confidence():
    results = classify([(20, 20, 300, 460, 0.83)])
    assert results["L"]["status"] == "occupied"
    assert results["L"]["confidence"] == 0.83
    assert results["R"]["status"] == "vacant"


def test_motorcycle_sized_box_still_occupies_a_slot():
    results = classify([(100, 200, 160, 280, 0.6)])
    assert results["L"]["status"] == "occupied"


def test_strongest_matching_detection_sets_confidence():
    results = classify([(20, 20, 300, 460, 0.5), (30, 30, 290, 450, 0.9)])
    assert results["L"]["confidence"] == 0.9


def test_partial_overlap_lowers_vacant_confidence():
    # Box 0-340: 20 px of it falls in slot R, a share below the 0.4 threshold.
    results = classify([(0, 0, 340, 480, 0.9)])
    right = results["R"]
    assert right["status"] == "vacant"
    assert right["confidence"] < 1.0


def test_weak_overlap_becomes_unknown_after_review_threshold():
    # Box x 100-420 straddles both slots: 100 of its 320 px (31%) fall in R, below the
    # 0.4 threshold. Doubt = 0.95 * (0.3125 / 0.4)^2 = 0.58, so R reads vacant at 0.42.
    results = classify([(100, 0, 420, 480, 0.95)])
    assert results["R"]["status"] == "vacant" and results["R"]["confidence"] < 0.7
    shown = slotsight.apply_review_threshold([dict(r) for r in results.values()], 0.7)
    assert {r["id"]: r["status"] for r in shown}["R"] == "unknown"


def test_low_confidence_detection_shows_unknown_not_occupied():
    results = classify([(20, 20, 300, 460, 0.3)])
    assert results["L"]["status"] == "occupied"
    shown = {r["id"]: r for r in slotsight.apply_review_threshold([dict(r) for r in results.values()], 0.7)}
    assert shown["L"]["status"] == "unknown"


def test_overlap_threshold_is_respected():
    box = [(0, 0, 400, 480, 0.9)]  # 80/320 = 25% of slot R, 20% of the box
    assert classify(box, overlap_threshold=0.4)["R"]["status"] == "vacant"
    assert classify(box, overlap_threshold=0.2)["R"]["status"] == "occupied"


def test_slot_points_are_scaled_to_the_frame():
    points = slotsight.slot_polygon_pixels({"points": [[50, 50], [100, 100], [0, 100]]}, 640, 480)
    assert np.allclose(points, [[320, 240], [640, 480], [0, 480]])
