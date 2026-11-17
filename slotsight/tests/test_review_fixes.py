"""Regression tests for the 2026-11-17 code review (docs/PLAN.md review log, a1212e9..686fd1b)."""

import io
import json
import tokenize
from pathlib import Path

import numpy as np
import pytest
from conftest import TWO_SLOTS, blank_frame, history_rows, png_upload

import detectors

SLOTSIGHT_DIR = Path(__file__).resolve().parent.parent


# --- 1. Config reads never reset the files -------------------------------------------


def test_unreadable_settings_keep_the_last_good_values_and_the_file(app):
    app.save_settings(lot_name="My lot", review_threshold=0.55)
    app.SETTINGS_PATH.write_text('{"lot_name": "My l', encoding="utf-8")  # half-written
    settings = app.load_settings()
    assert settings["lot_name"] == "My lot" and settings["review_threshold"] == 0.55
    assert app.SETTINGS_PATH.read_text(encoding="utf-8") == '{"lot_name": "My l'


def test_a_locked_settings_file_is_not_overwritten(app, monkeypatch):
    app.save_settings(lot_name="My lot")
    before = app.SETTINGS_PATH.read_bytes()
    real_read = Path.read_text

    def locked(self, *args, **kwargs):
        if self == app.SETTINGS_PATH:
            raise PermissionError("locked by another process")
        return real_read(self, *args, **kwargs)

    # A scoped patch: undoing the whole monkeypatch would also restore the real config paths.
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", locked)
        assert app.load_settings()["lot_name"] == "My lot"
    assert app.SETTINGS_PATH.read_bytes() == before


def test_unreadable_settings_on_first_read_are_not_overwritten(app):
    app.SETTINGS_PATH.parent.mkdir(parents=True)
    app.SETTINGS_PATH.write_text("not json", encoding="utf-8")
    assert app.load_settings()["lot_name"] == app.DEFAULT_LOT_NAME
    assert app.SETTINGS_PATH.read_text(encoding="utf-8") == "not json"


def test_unreadable_slots_keep_the_last_good_layout(app):
    app.save_slots(TWO_SLOTS)
    assert app.load_slots() == TWO_SLOTS
    app.CONFIG_PATH.write_text("[{", encoding="utf-8")
    assert app.load_slots() == TWO_SLOTS


def test_saves_leave_no_temporary_files(app):
    app.save_settings(lot_name="A")
    app.save_slots(TWO_SLOTS)
    assert sorted(path.name for path in app.CONFIG_PATH.parent.iterdir()) == ["settings.txt", "slots.json"]
    assert json.loads(app.SETTINGS_PATH.read_text(encoding="utf-8"))["lot_name"] == "A"


# --- 2. Zero-area slots ------------------------------------------------------------------


def test_zero_area_slot_reads_unknown_never_vacant(app):
    slots = [{"id": "Z", "points": [[10, 10], [20, 20], [30, 30]]}]
    results = app.classify_frame(blank_frame(), slots, [(0, 0, 640, 480, 0.95)], 0.4)
    assert results == [{"id": "Z", "status": "unknown", "confidence": 0.0}]


@pytest.mark.parametrize(
    ("slots", "message"),
    [
        ([{"id": "Z", "points": [[10, 10], [20, 20], [30, 30]]}], "no area"),
        ([{"id": "Z", "points": [[10, 10], [10, 10], [10, 10]]}], "no area"),
        ([{"id": "Z", "points": [[10, 10], [10.5, 10], [10.5, 10.5]]}], "no area"),
        ([{"id": "A", "points": [[0, 0], [9, 0], [9, 9]]}, {"id": "A", "points": [[20, 0], [29, 0], [29, 9]]}], "used twice"),
        ([{"id": 5, "points": [[0, 0], [9, 0], [9, 9]]}], "text id"),
        ([{"id": "x" * 41, "points": [[0, 0], [9, 0], [9, 9]]}], "text id"),
        ([{"id": "A", "points": [[0, 0], [9, 0]]}], "three"),
        ([{"id": "A", "points": [[0, 0], [9, 0], [9, 101]]}], "outside"),
        (["not a slot"], "object"),
    ],
)
def test_config_rejects_unusable_slots(client, slots, message):
    response = client.post("/api/config", json={"slots": slots, "lot_name": "x", "review_threshold": 0.7})
    assert response.status_code == 400
    assert message in response.get_json()["error"]


def test_the_default_layout_passes_validation(app):
    assert app.slot_config_problem(app.DEFAULT_SLOTS) is None


# --- 3. A model whose download failed can't be selected -----------------------------------


def test_failed_download_stays_missing_and_cannot_be_selected(client, app):
    app.models._errors["rfdetr/nano"] = "Could not download rf-detr-nano.pth: network unreachable"
    status = app.models.status("rfdetr/nano")
    assert status["state"] == "missing"
    assert "network unreachable" in status["error"]
    assert client.post("/api/models/select", json={"id": "rfdetr/nano"}).status_code == 409
    assert app.load_settings()["model"] == app.DEFAULT_MODEL


# --- 4. Live model status on every poll ---------------------------------------------------


def test_status_reports_the_live_model_state_after_a_reading(client, fake_detector, two_slots, app, monkeypatch):
    client.post("/api/process", data=png_upload(blank_frame()), content_type="multipart/form-data")
    assert client.get("/api/status").get_json()["model_status"]["state"] == "ready"
    monkeypatch.setattr(app.models, "status", lambda model_id: {"id": model_id, "name": "Other", "state": "error", "message": "boom", "progress": None, "error": "boom"})
    data = client.get("/api/status").get_json()
    assert data["model_status"]["state"] == "error"
    assert data["model"] == "Fake detector"  # the reading still names the model that made it


# --- 5. Stopped or replaced sources don't store readings ---------------------------------------


class FakeWorker:
    stopping = False

    def stop(self):
        self.stopping = True


def test_frames_from_a_stopped_or_replaced_source_are_dropped(fake_detector, two_slots, app, monkeypatch):
    worker = FakeWorker()
    handler = app.frame_handler("camera")
    handler.owner.append(worker)
    monkeypatch.setattr(app, "active_source", worker)
    handler(blank_frame(), None)
    assert len(history_rows(app)) == 2
    worker.stopping = True
    handler(blank_frame(), None)
    assert len(history_rows(app)) == 2
    worker.stopping = False
    monkeypatch.setattr(app, "active_source", FakeWorker())  # replaced by a new source
    handler(blank_frame(), None)
    assert len(history_rows(app)) == 2


# --- 6. A broken model is an error, not demo data -------------------------------------------------


@pytest.mark.parametrize("state", ["error", "needs_package", "unknown"])
def test_broken_model_shows_unknown_slots_and_the_error(client, app, monkeypatch, state):
    monkeypatch.setattr(app.models, "status", lambda model_id: {"id": model_id, "name": "YOLO11 small", "state": state, "message": "weights corrupt", "progress": None, "error": "weights corrupt"})
    data = client.get("/api/status").get_json()
    assert data["source"] == "model error"
    assert {slot["status"] for slot in data["slots"]} == {"unknown"}
    assert data["model"] != "demo classifier"
    assert data["model_status"]["state"] == state


def test_demo_data_only_when_no_model_is_installed(client, app):
    data = client.get("/api/status").get_json()
    assert data["model_status"]["state"] == "missing"
    assert data["source"] == "demo"


# --- 7. Exports never write the history file ---------------------------------------------------------


def test_exports_do_not_create_the_history_file(client, app):
    assert client.get("/api/history.csv").status_code == 200
    assert client.get("/api/history.xlsx").status_code == 200
    assert not app.HISTORY_PATH.exists()


def test_exports_read_an_older_file_without_changing_it(client, app):
    app.HISTORY_PATH.parent.mkdir(parents=True)
    app.HISTORY_PATH.write_text("timestamp,slot_id,status,confidence,source\n2026-01-01T00:00:00+00:00,A01,vacant,0.9,x.png\n", encoding="utf-8")
    before = app.HISTORY_PATH.read_bytes()
    text = client.get("/api/history.csv").get_data(as_text=True)
    assert text.splitlines() == [",".join(app.HISTORY_COLUMNS), "2026-01-01T00:00:00+00:00,A01,vacant,0.9,x.png,,"]
    assert app.HISTORY_PATH.read_bytes() == before


# --- 8. Upload size limit ------------------------------------------------------------------------------


def test_oversized_upload_is_rejected_with_a_message(client, fake_detector, two_slots, app, monkeypatch):
    monkeypatch.setitem(app.app.config, "MAX_CONTENT_LENGTH", 10_000)
    response = client.post("/api/process", data={"source": (io.BytesIO(b"x" * 50_000), "big.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 413
    assert "smaller than" in response.get_json()["error"]
    assert history_rows(app) == []


def test_the_upload_limit_is_configured():
    import app as slotsight

    assert slotsight.app.config["MAX_CONTENT_LENGTH"] >= slotsight.MAX_UPLOAD_MB * 1024 * 1024


# --- 9. No raw HTML from data in the dashboard --------------------------------------------------------------


@pytest.mark.parametrize("script", ["app.js", "label.js"])
def test_dashboard_scripts_never_use_innerHTML(script):
    assert "innerHTML" not in (SLOTSIGHT_DIR / "static" / script).read_text(encoding="utf-8")


# --- 10. Code parses on Python before 3.12 -----------------------------------------------------------------


def python_files():
    return sorted(path for path in SLOTSIGHT_DIR.rglob("*.py") if ".venv" not in path.parts)


@pytest.mark.parametrize("path", python_files(), ids=lambda path: str(path.relative_to(SLOTSIGHT_DIR)))
def test_no_quote_reused_inside_an_f_string(path):
    """Reusing the enclosing quote inside an f-string (PEP 701) is a SyntaxError before 3.12."""
    if not hasattr(tokenize, "FSTRING_START"):
        pytest.skip("this Python tokenizes f-strings as plain strings, which already rules this out")
    open_quotes: list[str] = []
    with path.open("rb") as file:
        for token in tokenize.tokenize(file.readline):
            if token.type == tokenize.FSTRING_START:
                quote = token.string.lstrip("rRfFbBuU")[:1]
                assert quote not in open_quotes, f"{path.name}:{token.start[0]} reuses {quote} inside an f-string"
                open_quotes.append(quote)
            elif token.type == tokenize.FSTRING_END:
                open_quotes.pop()
            elif token.type == tokenize.STRING and open_quotes:
                quote = token.string.lstrip("rRbBuU")[:1]
                assert quote not in open_quotes, f"{path.name}:{token.start[0]} reuses {quote} inside an f-string"
