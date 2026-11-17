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

Open <http://127.0.0.1:5000>.

On first run the app creates `slotsight/config/slots.json` (a 12-slot sample layout) and `settings.txt` from its built-in defaults. It also downloads the default detection model, YOLO11 small (19 MB), from Ultralytics' GitHub release into `slotsight/models/`. The dashboard shows the download's progress, and slots show as unknown until the model is ready. This needs an internet connection once. If the download fails, the dashboard stays in a clearly labelled demo mode and shows the error under **Configure slots > Detection model**, where you can retry.

## Detection models

SlotSight can use COCO-trained detectors from four providers: Ultralytics YOLO (YOLO11 and YOLO26, AGPL-3.0), Hugging Face RT-DETRv2, Roboflow RF-DETR, and an OpenCV-only YOLOX (all three Apache-2.0). Choose one under **Configure slots > Detection model**, where **Install** downloads a model and **Use** switches to it. You can also do this from a terminal:

```powershell
python slotsight/scripts/models.py list
python slotsight/scripts/models.py install opencv/yolox-s
python slotsight/scripts/models.py use opencv/yolox-s
```

Downloads are checked against published hashes and saved in `slotsight/models/` (not tracked by git). The RT-DETRv2 and RF-DETR models also need their Python package (`pip install transformers` or `pip install rfdetr`); the dashboard tells you which. The active model is the `model` setting in `settings.txt` (e.g. `"ultralytics/yolo11s"`), and the `SLOTSIGHT_MODEL` environment variable overrides it. Older `model_path` settings are converted automatically. See [`slotsight/models/README.md`](slotsight/models/README.md) for the full list, sizes and licences.

Without a usable model, the dashboard runs in demo mode with sample data, which is never logged. With a model, slots show as unknown until a frame is processed, and a frame that can't be analysed returns an error instead of demo data.

## Repository layout

```text
README.md, .gitignore
.claude/CLAUDE.md          instructions for Claude Code
slotsight/
  app.py                   Flask backend: classification, history, API
  detectors.py             detection models from several providers: registry, install, load
  sources.py               video file, stream and camera reader
  requirements.txt         Python dependencies
  config/                  files you edit (gitignored; created on first run)
    settings.txt           lot name, thresholds, sample interval, model (JSON)
    slots.json             slot polygons
  data/                    generated output (gitignored)
    occupancy_history.csv  one row per slot per processed frame
    eval_frames/           labeled test frames (stay local)
  models/                  downloaded detection models (gitignored); README lists licences
  static/, templates/      dashboard UI
  scripts/                 model installer (models.py) and the R-04 timing test
  tests/                   pytest suite (python -m pytest slotsight/tests)
  eval/                    accuracy evaluation: evaluate.py and labels.csv
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

All paths are under `slotsight/`:

Your configuration stays on your machine: git ignores both config files, and the app recreates them from its defaults if they're missing.

- `config/slots.json`: slot polygons, edited with **Configure slots** or **Edit layout**. Coordinates are normalised from 0 to 100, so a configuration works at any image size.
- `config/settings.txt`: lot name, `review_threshold`, `overlap_threshold`, `sample_interval_s` (default 2 s) and `model` (default `ultralytics/yolo11s`).
- `data/occupancy_history.csv`: one row per slot for every processed frame (viewing the dashboard doesn't add rows). Columns: `timestamp, slot_id, status, confidence, source, frame_time_s`, where `frame_time_s` is the frame's position in a video file. The dashboard offers CSV and Excel downloads. Older history files are kept beside it as `occupancy_history-<date>.csv`.
- `models/`: downloaded detection models.

On the dashboard map, each slot is filled green (available), red (occupied) or yellow with a dashed edge (unknown), with a pinned label showing its ID and a status symbol (○ ● ?). The time of the last update is in the map's corner, and hovering over or tabbing to a slot shows its status, confidence and update time.

The dashboard polls every 2 s. With 2 s sampling, a change reaches the screen within about 4 s (see [`slotsight/docs/evaluation.md`](slotsight/docs/evaluation.md)).

## Running the tests

```powershell
pip install -r slotsight/requirements-dev.txt
python -m pytest slotsight/tests
```

The suite takes about 10 seconds and needs no network. It uses a fake detector and temporary folders, so it never touches your config, history or models. One test runs the real default model on a sample image, and is skipped if that model isn't installed.

## Measuring update speed

```powershell
python slotsight/scripts/measure_update_interval.py path\to\footage.mp4 --seconds 120 --write
```

This plays the source at recorded speed through the full pipeline, reports processing time and the update interval against R-04's 5 s target, and with `--write` appends the result to `slotsight/docs/evaluation.md`. Readings go to a temporary history file.

## Measuring accuracy

```powershell
python slotsight/eval/evaluate.py --note "first labeled set" --write
```

This scores the dashboard's readings against hand-labeled frames for R-01 (per-slot accuracy), R-02 (free-count match), R-05 (evening and rain against daytime) and R-07 (occluded slots must be correct or unknown, never a wrong reading). Put the frame images in `slotsight/data/eval_frames/` (not tracked) and label them in `slotsight/eval/labels.csv`, one row per slot per frame:

```text
frame,slot_id,true_status,condition,occluded
day/0001.jpg,A01,occupied,daytime,0
day/0001.jpg,A02,vacant,daytime,1
```

`true_status` is `occupied` or `vacant`, `condition` is `daytime`, `evening` or `rain`, and `occluded` is `1` for a partly hidden slot. Label every slot in a frame so it counts towards R-02. An "unknown" reading counts as wrong for accuracy, and coverage is reported beside it. Detections are cached, so re-running with `--overlap-threshold` or `--review-threshold` to compare settings is fast. Per-row results go to `slotsight/data/eval_predictions.csv`, and `--write` adds the report to `slotsight/docs/evaluation.md`.

## Project documents

- [Product requirements](slotsight/docs/requirements.md) (R-01 to R-08)
- [Plan](slotsight/docs/PLAN.md): requirement status, roadmap, review log and decisions
- [Evaluation](slotsight/docs/evaluation.md): measured evidence

## Licences

SlotSight's default detector uses [Ultralytics](https://github.com/ultralytics/ultralytics), whose code and YOLO weights are licensed under **AGPL-3.0**. You can run SlotSight for yourself freely. If you distribute it, or let others use it over a network, AGPL-3.0 requires you to make its complete source available under the same licence, unless you have an Ultralytics Enterprise licence. To avoid AGPL code, use one of the Apache-2.0 models (RT-DETRv2, RF-DETR or OpenCV YOLOX) and uninstall `ultralytics`. Details, with each model's licence, are in [`slotsight/models/README.md`](slotsight/models/README.md). SlotSight itself does not yet have a licence of its own.

## Scope notes

The prototype intentionally excludes accounts, payments, reservations, plate recognition, multi-camera stitching, and unattended deployment.
