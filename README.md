# SlotSight

A lightweight parking occupancy dashboard for one fixed camera and one lot. It supports slot polygons, live/demo status updates, image processing, confidence-aware statuses, and CSV/Excel history export.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

Without a detection model, the dashboard runs in demo mode with sample data, which is never logged. To enable vehicle detection, install `ultralytics` and place a compatible YOLO `.pt` model in the project root. `model_path` in `settings.txt` selects the model; the `SLOTSIGHT_MODEL` environment variable overrides it. Relative paths resolve against the project folder. With a model loaded, slots show as unknown until a frame is processed, and a frame that can't be analysed returns an error instead of demo data.

The detector treats common vehicle classes (car, motorcycle, bus, truck) as detections. A slot is occupied when a detection overlaps it by at least `overlap_threshold`: the larger of the share of the slot covered by the box and the share of the box inside the slot. Its confidence is the detection's score. A vacant slot's confidence drops when weaker or partial detections overlap it. Any slot below `review_threshold` confidence shows as unknown.

Slot polygons are stored in `slots.json` after using **Configure slots**. Coordinates are normalized from 0 to 100 so a configuration remains portable across image sizes. Lot name, review threshold, overlap threshold, and model path are stored together in `settings.txt`. Each processed frame appends one row per slot to `occupancy_history.csv` (viewing the dashboard does not); the dashboard provides CSV and optional Excel downloads.

## Scope notes

The prototype intentionally excludes accounts, payments, reservations, plate recognition, multi-camera stitching, and unattended deployment. Video stream integration can reuse the `/api/process` adapter by decoding frames before calling `snapshot()`.
