# SlotSight

A lightweight parking occupancy dashboard for one fixed camera and one lot. It classifies each user-defined slot as occupied, vacant or unknown from images, recorded video or a live camera, shows the result on a live map, and exports the history as CSV or Excel.

## Run locally

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r slotsight/requirements.txt
python slotsight/app.py
```

Open http://127.0.0.1:5000.

Without a detection model, the dashboard runs in demo mode with sample data, which is never logged. To enable vehicle detection, install `ultralytics` and place a compatible YOLO `.pt` model in the `slotsight/` folder. `model_path` in `slotsight/settings.txt` selects the model; the `SLOTSIGHT_MODEL` environment variable overrides it. Relative paths resolve against the `slotsight/` folder. With a model loaded, slots show as unknown until a frame is processed, and a frame that can't be analysed returns an error instead of demo data.

## Repository layout

```text
README.md, .gitignore
.claude/CLAUDE.md          instructions for Claude Code
slotsight/
  app.py                   Flask backend: detection, classification, history, API
  sources.py               video file, stream and camera reader
  requirements.txt
  settings.txt             lot name, thresholds, sample interval, model path (JSON)
  slots.json               slot polygons
  yolo11s.pt               detection model
  static/, templates/      dashboard UI
  scripts/                 measurement tools
  docs/                    requirements.md, PLAN.md, evaluation.md
```

## Inputs

- **Images**: upload one in the **Camera & footage** panel. It is classified immediately.
- **Recorded video** (up to 100 MB): upload it in the same panel. It is processed in the background, one frame every `sample_interval_s` seconds of footage, with progress shown on the dashboard. For longer recordings, enter the file's path in **Live source** instead of uploading it.
- **Live source**: enter an RTSP/HTTP stream URL, a webcam index such as `0`, or a video file path, then press **Start**. Live sources are sampled every `sample_interval_s` seconds of wall-clock time and reconnect automatically after dropouts. Tick **Play video files at recorded speed** to make a recording behave like a live camera.
- **Footage recorded at** (optional): the real-world start time of an uploaded image or video. History rows then use the recording start plus each frame's offset as their timestamp, instead of the processing time.

Only one source runs at a time. Starting a new one stops the previous one.

## How slots are classified

The detector treats common vehicle classes (car, motorcycle, bus, truck) as detections. A slot is occupied when a detection overlaps it by at least `overlap_threshold`: the larger of the share of the slot covered by the box and the share of the box inside the slot. Its confidence is the detection's score. A vacant slot's confidence drops when weaker or partial detections overlap it. Any slot below `review_threshold` confidence shows as unknown.

## Configuration and data

All in `slotsight/`:

- `slots.json`: slot polygons, edited with **Configure slots** or **Edit layout**. Coordinates are normalised from 0 to 100, so a configuration works at any image size.
- `settings.txt`: lot name, `review_threshold`, `overlap_threshold`, `sample_interval_s` (default 2 s) and `model_path`.
- `occupancy_history.csv`: one row per slot for every processed frame (viewing the dashboard doesn't add rows). Columns: `timestamp, slot_id, status, confidence, source, frame_time_s`, where `frame_time_s` is the frame's position in a video file. The dashboard offers CSV and Excel downloads.

The dashboard polls every 2 s. With 2 s sampling, a change reaches the screen within about 4 s (see [`slotsight/docs/evaluation.md`](slotsight/docs/evaluation.md)).

## Measuring update speed

```powershell
python slotsight/scripts/measure_update_interval.py path\to\footage.mp4 --seconds 120 --write
```

This plays the source at recorded speed through the full pipeline, reports processing time and the update interval against R-04's 5 s target, and with `--write` appends the result to `slotsight/docs/evaluation.md`. Readings go to a temporary history file.

## Project documents

- [Product requirements](slotsight/docs/requirements.md) (R-01 to R-08)
- [Plan](slotsight/docs/PLAN.md): requirement status, roadmap, review log and decisions
- [Evaluation](slotsight/docs/evaluation.md): measured evidence

## Scope notes

The prototype intentionally excludes accounts, payments, reservations, plate recognition, multi-camera stitching, and unattended deployment.
