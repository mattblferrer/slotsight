"""History file format and CSV/Excel exports (R-08)."""

import csv
import io

from conftest import blank_frame, png_upload


def process_two_frames(client, detector):
    client.post("/api/process", data=png_upload(blank_frame(), "a.png"), content_type="multipart/form-data")
    detector.boxes = [(20, 20, 300, 460, 0.9)]
    client.post("/api/process", data=png_upload(blank_frame(), "b.png"), content_type="multipart/form-data")


def test_csv_export_has_every_processed_frame(client, fake_detector, two_slots, app):
    process_two_frames(client, fake_detector)
    response = client.get("/api/history.csv")
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
    assert list(rows[0]) == app.HISTORY_COLUMNS
    assert [(row["source"], row["slot_id"], row["status"]) for row in rows] == [
        ("a.png", "L", "vacant"),
        ("a.png", "R", "vacant"),
        ("b.png", "L", "occupied"),
        ("b.png", "R", "vacant"),
    ]
    assert all(row["timestamp"] for row in rows)


def test_excel_export_matches_the_csv(client, fake_detector, two_slots, app):
    from openpyxl import load_workbook

    process_two_frames(client, fake_detector)
    response = client.get("/api/history.xlsx")
    assert response.status_code == 200
    sheet = load_workbook(io.BytesIO(response.data)).active
    rows = [[cell if cell is not None else "" for cell in row] for row in sheet.iter_rows(values_only=True)]
    csv_rows = list(csv.reader(io.StringIO(client.get("/api/history.csv").get_data(as_text=True))))
    assert rows == csv_rows


def test_exports_work_before_anything_is_processed(client, app):
    response = client.get("/api/history.csv")
    assert response.status_code == 200
    assert response.get_data(as_text=True).strip() == ",".join(app.HISTORY_COLUMNS)
    assert client.get("/api/history.xlsx").status_code == 200


def test_older_history_files_are_upgraded_in_place(app):
    app.HISTORY_PATH.parent.mkdir(parents=True)
    app.HISTORY_PATH.write_text("timestamp,slot_id,status,confidence,source\n2026-01-01T00:00:00+00:00,A01,vacant,0.9,x.png\n", encoding="utf-8")
    app.ensure_history()
    with app.HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == app.HISTORY_COLUMNS
    assert rows[0]["slot_id"] == "A01" and rows[0]["frame_time_s"] == ""
