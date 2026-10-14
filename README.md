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

The dashboard runs immediately in demo mode. To enable vehicle detection, install `ultralytics`, place a compatible YOLO `.pt` model in the project root, and set `SLOTSIGHT_MODEL` to its filename. The current adapter treats common vehicle classes (car, motorcycle, bus, truck) as detections and assigns them to a slot when the detection center falls inside its polygon.

Slot polygons are stored in `slots.json` after using **Configure slots**. Coordinates are normalized from 0 to 100 so a configuration remains portable across image sizes. Lot name, review threshold, and model path are stored together in `settings.txt`. Processed results append to `occupancy_history.csv`; the dashboard provides CSV and optional Excel downloads.

## Scope notes

The prototype intentionally excludes accounts, payments, reservations, plate recognition, multi-camera stitching, and unattended deployment. Video stream integration can reuse the `/api/process` adapter by decoding frames before calling `snapshot()`.
