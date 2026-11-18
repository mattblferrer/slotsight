"""Regression tests for the 2026-11-18 review of the dashboard UI work (`cf93312..4c5eeba`).

The races are made deterministic by wrapping slot_probabilities (the classifier call
that runs outside state_lock) so the "meanwhile" change happens exactly in that gap.
The browser-side findings (1, 2, 4, 5, 7, 10) were checked in a browser instead.
"""

from conftest import TWO_SLOTS, blank_frame, png_upload
from test_classifier import save_random_weights

THIRD_SLOT = {"id": "M", "points": [[40, 40], [60, 40], [60, 60], [40, 60]]}


def process(client):
    return client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data").get_json()


def use_classifier(app, monkeypatch):
    """Weights present and a classifier that calls every slot vacant, whatever the layout."""
    app.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_random_weights(app.MODELS_DIR)
    monkeypatch.setattr(app.slot_classifier, "predict", lambda frame, slots: [(0.98, 0.02)] * len(slots))


def once_during_classifier_call(app, monkeypatch, action):
    """Run `action` once, right after the next classifier call outside the lock."""
    original = app.slot_probabilities
    done = []

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if not done:
            done.append(True)
            action()
        return result

    monkeypatch.setattr(app, "slot_probabilities", wrapped)


# 3. Responses that the page renders carry the camera check and the job.
def test_setting_and_upload_responses_carry_the_camera_status(client, app, fake_detector, two_slots):
    upload = process(client)
    assert {"camera", "job", "model_status"} <= upload.keys()
    for url, body in (("/api/settings", {"review_threshold": 0.6}), ("/api/review-threshold", {"review_threshold": 0.5})):
        data = client.post(url, json=body).get_json()
        assert {"camera", "job", "model_status"} <= data.keys(), url


# 6. A layout save keeps a setting saved while it was computing.
def test_a_layout_save_keeps_a_setting_saved_meanwhile(client, app, fake_detector, two_slots, monkeypatch):
    process(client)
    once_during_classifier_call(app, monkeypatch, lambda: app.save_settings(review_threshold=0.5))
    assert client.post("/api/config", json={"slots": two_slots}).status_code == 200
    assert app.load_settings()["review_threshold"] == 0.5


# 8. Switching the classifier on never applies output computed for another layout.
def test_classifier_output_for_an_old_layout_is_not_applied(client, app, fake_detector, two_slots, monkeypatch):
    use_classifier(app, monkeypatch)
    app.save_settings(slot_classifier=False)
    process(client)
    once_during_classifier_call(app, monkeypatch, lambda: app.save_slots(TWO_SLOTS + [THIRD_SLOT]))
    data = client.post("/api/settings", json={"slot_classifier": True}).get_json()
    assert [s["id"] for s in data["slots"]] == ["L", "R", "M"]
    assert all(s["method"] == "classifier" for s in data["slots"])
    # The cache holds output for the layout it was computed for.
    cached_slots, cached = app.latest_frame_probabilities
    assert cached_slots == TWO_SLOTS + [THIRD_SLOT] and len(cached) == 3


# 9. A frame classified just before the classifier is switched off is decided without it.
def test_a_frame_in_flight_is_not_decided_by_a_switched_off_classifier(client, app, fake_detector, two_slots, monkeypatch):
    use_classifier(app, monkeypatch)
    once_during_classifier_call(app, monkeypatch, lambda: app.save_settings(slot_classifier=False))
    data = process(client)
    assert all(s["method"] == "detector" for s in data["slots"])
