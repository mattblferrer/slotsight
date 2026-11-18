# SlotSight

A lightweight parking occupancy dashboard for one fixed camera and one lot. It classifies each user-defined slot as occupied, vacant or unknown from images, recorded video or a live camera, shows the result on a live map, and exports the history as CSV or Excel.

## Run locally

Requires Python 3.11 or newer (tested on 3.14). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r slotsight/requirements.txt
python slotsight/app.py
```

Open <http://127.0.0.1:5000>. The first visit in a browser opens a short guide (`/guide`) to setting up the lot and reading the dashboard; it is linked from the dashboard's top bar afterwards. To work on the code with Flask's debugger and auto-reload, set `SLOTSIGHT_DEBUG=1` first (`$env:SLOTSIGHT_DEBUG = "1"` in PowerShell). It is off by default, because the debugger can run code sent from the browser.

On first run the app creates `slotsight/config/slots.json` (a 12-slot sample layout) and `settings.txt` from its built-in defaults. It also downloads the default detection model, YOLO11 small (19 MB), from Ultralytics' GitHub release into `slotsight/models/`. The dashboard shows the download's progress, and slots show as unknown until the model is ready. This needs an internet connection once. If the download fails, the dashboard stays in a clearly labelled demo mode and shows the error under **Settings > Detection model and resolution**, where you can retry.

## Detection models

SlotSight can use COCO-trained detectors from four providers: Ultralytics YOLO (YOLO11 and YOLO26, AGPL-3.0), Hugging Face RT-DETRv2, Roboflow RF-DETR, and an OpenCV-only YOLOX (all three Apache-2.0). Choose one under **Settings > Detection model and resolution**, where **Install** downloads a model and **Use** switches to it. You can also do this from a terminal:

```powershell
python slotsight/scripts/models.py list
python slotsight/scripts/models.py install opencv/yolox-s
python slotsight/scripts/models.py use opencv/yolox-s
```

Downloads are checked against published hashes and saved in `slotsight/models/` (not tracked by git). The RT-DETRv2 and RF-DETR models also need their Python package (`pip install transformers` or `pip install rfdetr`); the dashboard tells you which. The active model is the `model` setting in `settings.txt` (e.g. `"ultralytics/yolo11s"`), and the `SLOTSIGHT_MODEL` environment variable overrides it. Older `model_path` settings are converted automatically. See [`slotsight/models/README.md`](slotsight/models/README.md) for the full list, sizes and licences.

Only when no model is installed does the dashboard show demo data, and it is labelled as such and never logged. If the selected model is installed but can't be used (for example, it fails to load or its Python package is missing), the dashboard shows the error with every slot unknown, never demo data. With a working model, slots show as unknown until a frame is processed, and a frame that can't be analysed returns an error.

## Repository layout

```text
README.md, .gitignore
.claude/CLAUDE.md          instructions for Claude Code
slotsight/
  app.py                   Flask backend: classification, history, API
  detectors.py             detection models from several providers: registry, install, load
  tracking.py              smoothing over frames and how long each status has lasted
  appearance.py            "doesn't look empty" safeguard for cars the detector misses
  classifier.py            per-slot classifier (the primary decision; the detector votes)
  camera.py                camera-shift check against a reference frame
  trends.py                free slots over time and busy hours, from the history
  sources.py               video file, stream and camera reader
  requirements.txt         Python dependencies
  config/                  files you edit (gitignored; created on first run)
    settings.txt           lot name, thresholds, sample interval, model (JSON)
    slots.json             slot polygons
    reference_frame.jpg    camera-shift reference (saved with the layout)
  data/                    generated output (gitignored)
    occupancy_history.csv  one row per slot per processed frame
    eval_frames/           labeled test frames (stay local)
  models/                  downloaded detection models (gitignored), the slot classifier's
                           base weights (tracked); README lists licences
  static/, templates/      dashboard, guide (/guide) and label page; fonts/ holds the
                           self-hosted Overpass fonts (SIL OFL)
  scripts/                 model installer (models.py) and the R-04 timing test
  tests/                   pytest suite (python -m pytest slotsight/tests)
  labeling.py              reading and saving hand labels (the /label page)
  eval/                    accuracy evaluation: extract_frames.py, evaluate.py, labels.csv
  docs/                    requirements.md, PLAN.md, evaluation.md, recording-guide.md
```

## Inputs

- **Images**: drop one on, or choose one in, the **Camera and footage** section and press **Check this file**. It is classified immediately.
- **Recorded video** (up to 100 MB): upload it in the same panel. It is processed in the background, one frame every `sample_interval_s` seconds of footage, with progress shown on the dashboard. For longer recordings, enter the file's path in **Live camera** instead of uploading it.
- **Live camera**: enter an RTSP/HTTP stream URL, a webcam index such as `0`, or a video file path, then press **Connect** (**Disconnect** stops it). Live sources are sampled every `sample_interval_s` seconds of wall-clock time and reconnect automatically after dropouts. Tick **Play video files at recorded speed** to make a recording behave like a live camera.
- **Recorded at** (optional): the real-world start time of an uploaded image or video. History rows then use the recording start plus each frame's offset as their timestamp, instead of the processing time.

Only one source runs at a time. Starting a new one stops the previous one.

## Reading the dashboard

- **Sign:** the top of the dashboard shows the lot's free count in a lit window, with the taken and not-sure counts beside it. The number is lit only while it is a current reading; for demo data, no frame, a model error, a source that stopped sending, a disconnected camera or paused updates it turns grey, and the line beneath says why.
- **Map:** each slot is filled green (free), red (taken) or yellow with a dashed edge (not sure; never counted as free). Hover over or tab to a slot to see its status, how long it has had it, its confidence and the time of the reading.
- **Detections:** the frame on the map shows every vehicle detection in blue with its confidence. A solid box marks a slot as occupied. A dashed box is weak or only partly in a slot; those are the detections that turn slots "not sure".
- **Smoothing (videos and live sources):** a slot shows "occupied" after 2 agreeing frames, "vacant" only after 3, and "unknown" at once. A car driving past doesn't flicker a free slot, and a slot never reads free on a single frame. A slot whose readings keep flipping shows unknown. While a change is being confirmed, the tooltip shows the latest frame's result. Single uploaded images are shown as read.
- **How long:** **Recent changes** (newest first) and the tooltips show how long each slot has had its status ("Taken · 14m"). "≥" means the status was already there when monitoring started, so it may be longer. For recorded footage, durations are measured in footage time.
- **"Doesn't look empty" check (videos and live sources):** each slot learns what it looks like when empty. If a slot reads vacant but looks too different from that (a car the detector missed, say), it shows unknown with the reason in the tooltip. It never marks a slot occupied. `empty_check_threshold` in `settings.txt` sets how different is too different (default 0.8; 0 turns it off).
- **Detection resolution:** in **Settings > Detection model and resolution**, 640 px is the fastest. 960 or 1280 px finds small or distant cars better at a CPU cost (Ultralytics models only; the others have a fixed input size). Re-run the timing test after changing it.
- **Camera check:** the frame the layout was saved on is kept as a reference (`config/reference_frame.jpg`); without one, the first processed frame of a video or live source is used. About every 10th frame of a video or live source is compared with it. If the slots look more than 2% of the frame width out of place in 3 checks in a row, a warning asks you to check the layout. Saving a changed layout sets a new reference. When a frame has too few features to compare (night, fog), the map footer says it can't verify, rather than raising an alarm. Nothing is moved automatically.
- **Settings:** every change is saved as you make it. **Appearance** chooses System, Light or Dark for this browser.
- **Trends:** free slots over the last 24 hours or 7 days of readings, and the average share of slots occupied for each hour of the day, with the three busiest hours marked. They are drawn from the history and refresh every minute. Hover or use the arrow keys for values, or open the table.

## How slots are classified

Two models look at every frame:

- **The slot classifier** (`classifier.py`) decides. A small neural network (about 61k parameters) looks at each slot's own image, straightened to a 48×48 patch, and gives the probability that it's occupied. It ships with base weights trained on the public PKLot dataset, and `python slotsight/eval/train_classifier.py` fine-tunes them on this lot's labeled tune sessions in a minute or two on the CPU.
- **The vehicle detector** (the selected detection model) votes. A slot reads **vacant** only if the classifier says so *and* the detector sees no vehicle there. Otherwise it reads **unknown**, so a disagreement never shows as free. The detector's boxes are also what you see on the preview.

A slot whose classifier confidence is below `review_threshold` shows as unknown. Without classifier weights, or with **Use the slot classifier** switched off in Settings, the detector decides alone: a slot is occupied when a detection overlaps it by at least `overlap_threshold` (the larger of the share of the slot covered by the box and the share of the box inside the slot).

On PKLot (other lots, so proxy evidence only), this combination scored 97.8 % per-slot on camera views it was never trained on, against 52.7 % for the detector alone. See [`slotsight/docs/evaluation.md`](slotsight/docs/evaluation.md).

## Configuration and data

All paths are under `slotsight/`:

Your configuration stays on your machine: git ignores both config files, and the app recreates them from its defaults if they're missing.

- `config/slots.json`: slot polygons, edited with **Edit layout** on the map, or as JSON under **Settings > Bay outlines as JSON**. Coordinates are normalised from 0 to 100, so a configuration works at any image size.
- `config/settings.txt` (if it can't be read, for example because of a typo, the app keeps working with the last good settings and refuses to save over it until it's fixed): lot name, `review_threshold`, `overlap_threshold`, `sample_interval_s` (default 2 s), `inference_size` (640, 960 or 1280; default 640), `empty_check_threshold` (default 0.8; 0 = off), `slot_classifier` (default `true`) and `model` (default `ultralytics/yolo11s`).
- `data/occupancy_history.csv`: one row per slot for every processed frame (viewing the dashboard doesn't add rows). Columns: `timestamp, slot_id, status, confidence, source, frame_time_s, raw_status`. `status` is what the dashboard showed (smoothed for videos and live sources), `raw_status` is the single-frame result and `confidence` belongs to it, and `frame_time_s` is the frame's position in a video file. Older files gain the new column automatically. The dashboard offers CSV and Excel downloads. Older history files are kept beside it as `occupancy_history-<date>.csv`.
- `models/`: downloaded detection models.

On the dashboard map, each slot is filled green (free), red (taken) or yellow with a dashed edge (not sure), with a pinned label showing its ID and a status symbol (○ ● ?). The time of the last update is in the map's corner, and hovering over or tabbing to a slot shows its status, confidence and update time.

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

The full procedure, from recording the lot to reporting the evidence, is in [`slotsight/docs/recording-guide.md`](slotsight/docs/recording-guide.md). In short:

```powershell
python slotsight/eval/extract_frames.py lot-morning.mp4 --condition daytime        # frames for labeling
python slotsight/eval/extract_frames.py lot-rain.mp4 --condition rain --set tune   # kept apart for tuning
# label them at http://127.0.0.1:5000/label
python slotsight/eval/evaluate.py --only "tune/*" --overlap-threshold 0.5          # tune
python slotsight/eval/evaluate.py --exclude "tune/*" --note "test sessions" --write # report
```

While your own lot's footage was pending, the same classification code was run once on a subset of the public PKLot dataset (CC BY 4.0) as proxy evidence; the results are in [`slotsight/docs/evaluation.md`](slotsight/docs/evaluation.md). Those numbers come from other lots and camera views, so they don't show that this lot meets R-01, R-02 or R-05.

The evaluation scores the classifier against hand-labeled frames for R-01 (per-slot accuracy), R-02 (free-count match), R-05 (evening and rain against daytime) and R-07 (occluded slots must be correct or unknown, never a wrong reading). Frames go in `slotsight/data/eval_frames/<set>/<session>/` (not tracked). The **Label frames** page (`/label`, linked under History) shows each frame with the slot layout: click a slot to set it occupied or vacant, shift-click to mark it partly hidden, and press Enter to save and move on. Labels are saved to `slotsight/eval/labels.csv`, one row per slot per frame:

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

SlotSight's default detector uses [Ultralytics](https://github.com/ultralytics/ultralytics), whose code and YOLO weights are licensed under **AGPL-3.0**. You can run SlotSight for yourself freely. If you distribute it, or let others use it over a network, AGPL-3.0 requires you to make its complete source available under the same licence, unless you have an Ultralytics Enterprise licence. To avoid AGPL code, use one of the Apache-2.0 models (RT-DETRv2, RF-DETR or OpenCV YOLOX) and uninstall `ultralytics`. Details, with each model's licence, are in [`slotsight/models/README.md`](slotsight/models/README.md). The dashboard's Overpass and Overpass Mono fonts are under the SIL Open Font License 1.1 (see `slotsight/static/fonts/`). SlotSight itself does not yet have a licence of its own.

## Scope notes

The prototype intentionally excludes accounts, payments, reservations, plate recognition, multi-camera stitching, and unattended deployment.
