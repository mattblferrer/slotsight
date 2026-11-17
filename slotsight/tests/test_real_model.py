"""One end-to-end check with the real default model, skipped when it isn't installed.

Uses the weights already in slotsight/models/ (never downloads) and Ultralytics'
bundled sample image of a bus.
"""

from pathlib import Path

import cv2
import pytest

import detectors

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SPEC = detectors.MODELS[detectors.DEFAULT_MODEL_ID]

pytestmark = pytest.mark.skipif(
    not (detectors.package_available(SPEC.provider) and detectors.is_installed(SPEC, MODELS_DIR)),
    reason="default model or its package is not installed",
)


def test_default_model_finds_the_bus_in_the_sample_image():
    import ultralytics

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    detector = detectors.load_detector(SPEC, MODELS_DIR)
    boxes = detector.detect(image)
    best = max(boxes, key=lambda box: box[4])
    x1, y1, x2, y2, confidence = best
    assert confidence > 0.8
    assert (x2 - x1) > image.shape[1] * 0.8  # the bus spans the image width
    assert all(box[4] >= detectors.DETECTION_CONFIDENCE_FLOOR for box in boxes)
