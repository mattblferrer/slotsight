# CLAUDE.md

## Project

SlotSight is a prototype that counts parking occupancy from one fixed camera at one lot. A pretrained YOLO detector finds vehicles. Each one is matched against slot polygons defined by the user, and every slot is classified as occupied, vacant or unknown. A Flask dashboard shows the results and exports the history.

- Product requirements (the source of truth): [`slotsight/docs/requirements.md`](../slotsight/docs/requirements.md), IDs R-01 to R-08
- Working plan (requirement status, roadmap, review log, decisions): [`slotsight/docs/PLAN.md`](../slotsight/docs/PLAN.md), called `PLAN.md` below
- Measured evidence: [`slotsight/docs/evaluation.md`](../slotsight/docs/evaluation.md)

## Current goal and priorities

1. **Bring the project back to meeting R-01–R-08 first.** Work through Milestone 1 in `PLAN.md` in order (1A correctness, then 1B inputs and live updates, then 1C dashboard and evidence). Tick off checkboxes as items land.
2. **Only then** move on to Milestone 2 (hardening and improvements) in `PLAN.md`.
3. Don't add features outside the PRD scope while requirements are unmet. These are out of scope entirely: payments, reservations, plate reading, multi-camera stitching, user accounts.
4. When a change affects a requirement, name the requirement ID in the commit message or PR description, and update the requirement status table in `PLAN.md`. Record reviews in its review log and direction changes in its decisions log.
5. A requirement counts as met only when its acceptance criterion has evidence: evaluation output, a timing measurement or a demo. "Code exists" is not enough.

## Research and decisions

- **Research before designing.** Before building a significant piece (detection and matching logic, video/stream handling, evaluation, the dashboard map, smoothing over time), look at how comparable computer-vision parking-occupancy projects do it. Examples: open-source YOLO and OpenCV parking-spot detectors, PKLot/CNRPark-EXT papers and baselines, and Ultralytics' parking-management solution. Summarise what you took from them, and link the sources in the relevant `PLAN.md` decision entry. Don't copy code whose licence is incompatible.
- **Ask the user about important decisions. Don't decide silently.** This covers architecture (threading, storage, new modules or services), features (adding, removing or changing user-visible behaviour), requirement interpretations (what an acceptance criterion means in practice, defaults that affect R-01 to R-08), and anything hard to reverse (data formats, history schema, deleting data, repository layout). Give a recommended option and the trade-offs.
- **Log every answer in the `PLAN.md` decisions log** with the date, the decision, the reason, and who decided ("User", "User (Claude's recommendation)", or "Claude" for small reversible choices). Update the roadmap items it affects.
- Small, easily reversible implementation details don't need a question. Mention them in the summary instead.

## Repository layout

Only `README.md`, `.gitignore` and this `.claude/` folder sit at the top level. Everything else lives in `slotsight/`. The `.venv` virtual environment also stays at the top level (gitignored; venvs break if moved).

- `slotsight/app.py`: the backend (Flask routes, settings and slot persistence, classification, annotation, history and export). `BASE_DIR` is `slotsight/`. `CONFIG_DIR`, `DATA_DIR` and `MODELS_DIR` are its `config/`, `data/` and `models/` subfolders. `models` is the `ModelManager`; `detect_vehicles` loads the configured model through it.
- `slotsight/detectors.py`: the model registry (`MODELS`, ids like `ultralytics/yolo11s`, `huggingface/rtdetr-v2-r18`, `rfdetr/nano`, `opencv/yolox-s`, plus `ultralytics/file:<name>.pt`), hash-checked downloads, one adapter per provider (each returns vehicle boxes only, by COCO class name), and `ModelManager` (one loaded model, background install/load jobs, non-blocking `status()`). Provider packages are imported lazily.
- `slotsight/tracking.py`: `SlotTracker`, safe-side smoothing over consecutive frames (to vacant after 3 agreeing frames, to occupied after 2, to unknown at once) and when each status began. Only sequential sources (`smooth=True` in `store_reading`) are smoothed; a single image or a re-classification is shown as read and an image resets the tracker.
- `slotsight/classifier.py`: `SlotClassifier`, the primary per-slot decision (a 61k-parameter CNN on 48×48 warped slot patches). `combine()` is the decision rule: the classifier decides, a slot is vacant only if the detector agrees, and disagreement or low confidence gives unknown. `app.decide()` (classify_frame + combine) is shared by the dashboard and `evaluate.py`. Weights: `models/slot_classifier.pt` (fine-tuned, gitignored) if present, else `models/slot_classifier_base.pt` (PKLot base, the one tracked weights file). Missing weights, torch or `slot_classifier: false` fall back to the detector alone. For new frames, `slot_probabilities()` runs outside `state_lock`, next to detection, and under `inference_lock` (one model call at a time). Its output is cached with the latest frame (`latest_frame_probabilities`); a threshold change reuses it, and a config save computes the new layout's output before taking `state_lock`. Patches use `canonical_quad` (clockwise from the top-left), so corner order doesn't matter.
- Re-classifying the latest frame of a video or live source (`reclassify_latest_frame`) must keep the safeguards: the empty check re-runs with `learn=False` and the tracker uses `revise()`, never `update()`.
- `slotsight/eval/train_classifier.py`: trains the classifier. With no `--dataset` it fine-tunes the base weights on the lot's labeled `tune/*` sessions only. Never train on test sessions; that would contaminate the R-01/R-02/R-05 evidence. It records only folder names in the metadata, never local paths.
- `slotsight/appearance.py`: `EmptyReference`, the "doesn't look empty" safeguard. For sequential sources, after the review threshold and before smoothing, a vacant slot whose normalised crop differs from its learned empty reference by more than `empty_check_threshold` (default 0.8) becomes unknown with a `reason`. References learn from the detector's vacant readings with low edge density, are keyed by slot polygon, and reset when a new source starts. It never sets "occupied".
- `slotsight/camera.py`: `CameraMonitor`, an ORB + RANSAC comparison against `config/reference_frame.jpg` every 10th frame of a sequential source. "moved" after 3 checks over 2 % of the frame width; warns only. The reference is set on a layout save that changes the slots (or from the first frame of a sequential source).
- `slotsight/trends.py`: pure aggregation of `history_rows()` into free slots per interval (24h/7d, ending at the latest reading) and occupancy by local hour of day.
- `slotsight/sources.py`: `SourceWorker`, which reads video files, streams and cameras on a background thread and samples one frame per `sample_interval_s`
- `slotsight/static/app.js`: dashboard logic (2 s polling, map rendering, slot polygon editor, config dialog, uploads, live-source controls, job progress)
- `slotsight/static/style.css`, `slotsight/templates/index.html`: the UI (the "garage wayfinding sign" design: the sign with the LED free count leads; sign blue only on the sign, brand mark and primary button)
- `slotsight/templates/guide.html`, `slotsight/static/guide.js`: the guide at `/guide` (setup steps, reading the dashboard, shortcuts, limits) with a read-only live sign; `static/first-visit.js` (in the dashboard's `<head>`) opens it once per browser. Keep its wording in step with the dashboard and with what is measured (no accuracy claims before R-01/R-02/R-05 evidence)
- `slotsight/static/theme.js`: applies the viewer's theme (System, Light, Dark; kept in `localStorage`) before the page is drawn; loaded in `<head>` of both pages
- `slotsight/static/fonts/`: self-hosted Overpass and Overpass Mono (SIL OFL, licences and source in its `README.md`). Never load fonts or scripts from another site
- `slotsight/config/slots.json`: slot polygons, with coordinates **normalised 0–100** relative to frame width and height. Gitignored; created from `DEFAULT_SLOTS` on first run.
- `slotsight/config/settings.txt`: JSON despite the extension (`lot_name`, `review_threshold`, `overlap_threshold`, `sample_interval_s`, `inference_size`, `slot_classifier`, `empty_check_threshold`, `model`). Gitignored; created from the `app.py` defaults on first run. A legacy `model_path` is migrated to `model` automatically.
- `slotsight/data/occupancy_history.csv`: append-only log (`timestamp, slot_id, status, confidence, source, frame_time_s, raw_status`; `status` is smoothed for sequential sources, `raw_status` is the single frame's result). Gitignored, like the archived `occupancy_history-*.csv` files.
- `slotsight/models/`: downloaded weights, all gitignored except `README.md` (model list and licences, including Ultralytics' AGPL-3.0). The default `ultralytics/yolo11s` downloads on first run (`prepare_model()` in `__main__`, only in the reloader's serving process). Other models install from the dashboard or `scripts/models.py`.
- `slotsight/scripts/measure_update_interval.py`: the R-04 timing test
- `slotsight/scripts/models.py`: `list`, `install <id>`, `use <id>` for detection models
- `slotsight/labeling.py`: reading and saving `eval/labels.csv` for the `/label` page (atomic rewrite, one frame's rows replaced per save; frame paths confined to `data/eval_frames/`)
- `slotsight/eval/extract_frames.py`: frames every N seconds from a recording into `data/eval_frames/<test|tune>/<session>/` with a `session.json`; refuses to split a session between tune and test
- `slotsight/docs/recording-guide.md`: how to record, label, tune and report (the user-facing procedure for the remaining Milestone 1 evidence)
- `slotsight/eval/evaluate.py`: the accuracy harness (`--only` / `--exclude` frame-path patterns for the tune/test split) (R-01, R-02, R-05, R-07). Labels live in `slotsight/eval/labels.csv` (tracked). Frames go in `slotsight/data/eval_frames/`, and the detection cache and per-row results are `data/eval_cache.json` and `data/eval_predictions.csv`. All three are gitignored.
- `slotsight/tests/`: the pytest suite (`conftest.py` has the `app`, `fake_detector`, `two_slots` and `client` fixtures; every test uses temporary config/data/models folders). `slotsight/requirements-dev.txt` adds pytest.
- `slotsight/docs/`: `requirements.md`, `PLAN.md`, `evaluation.md`

API:
- `GET /api/status` (the snapshot plus `job`)
- `POST /api/process` (multipart `source`, optional `recorded_at`; images return immediately, videos return 202 and run as a job)
- `POST /api/source` (`target`, `realtime`, `recorded_at`), `POST /api/source/stop`, `GET /api/source`
- `POST /api/config` (`slots`, plus any settings), `POST /api/settings` (any of `lot_name`, `review_threshold`, `inference_size`, `slot_classifier`; never the layout; the Settings dialog saves each control through it), `POST /api/review-threshold`
- `GET /api/models` (catalogue, active id, running job), `POST /api/models/install` (`id`; 202, background download), `POST /api/models/select` (`id`; must be installed; saves `model` and loads in the background)
- `GET /guide` (guide page)
- `GET /label` (labeling page), `GET /api/label/frames`, `GET /api/label/image?frame=`, `GET /api/label/frame?frame=`, `POST /api/label/frame` (`frame`, `condition`, `slots`)
- `GET /api/history.csv`, `GET /api/history.xlsx`, `GET /api/trends?range=24h|7d` (read-only, cached by history size and modification time)
- `GET /api/status` also carries `camera` (camera-check state) and, per slot, `raw_status`, `changed_at` and `changed_since_start`

## Run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r slotsight/requirements.txt   # includes ultralytics for the default model
python slotsight/app.py                       # http://127.0.0.1:5000
```

Python 3.11 or newer (the "recorded at" times end in `Z`, which `datetime.fromisoformat` accepts from 3.11). Don't reuse an f-string's own quote inside it (that needs 3.12; a test checks). `SLOTSIGHT_DEBUG=1` turns on Flask's debugger and reloader; it is off by default.

The local `.venv` already has Flask, OpenCV, numpy, openpyxl, ultralytics, transformers and rfdetr installed (CPU-only torch; no CUDA on this laptop).

## Rules and pitfalls

- The repo lives in OneDrive and is open in VS Code. Files have been deleted or recreated outside Claude's commands after moves: an archive was sent to the Recycle Bin, and a stray `settings.txt` reappeared. After moving or deleting files, re-check with `git status` and `ls`, and never assume an untracked file is still where it was.
- **Never let demo data mix with real inference.** Demo output is only for when no frame exists and no model is installed (model state `missing`). While a model downloads or loads, slots are "unknown". When an installed model is broken (`error`, `needs_package`, `unknown`), slots are "unknown", `source` is `"model error"` and the dashboard shows the error, never demo data. An empty detection list on a real frame means every slot is vacant. A detector error must show up as an error, never as demo data.
- **GET endpoints must not write history.** Only processing a frame appends rows.
- "Unknown" must protect against false "free" readings (R-07). Don't hard-code confidence values for vacant slots.
- Keep defaults in sync across `app.py` constants (`DEFAULT_MODEL` comes from `detectors.DEFAULT_MODEL_ID`) and the UI defaults (slider, `POLL_MS` in `app.js`, `DASHBOARD_POLL_S` in the timing script). `settings.txt` is generated from them.
- Every registry download needs a published hash (`sha256` or `md5`) taken from the real file or the provider's own metadata, never typed in by guesswork.
- Resolve file paths against `BASE_DIR` (or `CONFIG_DIR`, `DATA_DIR`, `MODELS_DIR`), not the working directory. Editable config goes in `config/`, generated output in `data/`, and weights in `models/`. `sources.py` stays next to `app.py` because it's app code. `scripts/` is only for tools run by hand.
- Threading: run inference and model loading outside `state_lock` (status polls must never wait on them). `snapshot()` uses only the non-blocking `models.status()`. `models.require()` blocks while loading and is called only from `detect_vehicles` and scripts. Model calls are serialised by `inference_lock`. Lock order is source_lock → state_lock → ModelManager's internal lock (held briefly); never take state_lock while holding inference_lock or inside ModelManager.
- Build DOM from server or config data with `textContent` or element creation, not raw `innerHTML` interpolation.
- Write config files only with `write_atomic` and read them with `read_config`. Never write over a config file that couldn't be read; fall back to the last good copy. GET endpoints must not create or rewrite any file (exports read the history under `state_lock` and parse it outside).
- Labels are never pre-filled from the model (the labeling page starts every slot unset), so they stay independent of what is evaluated. Threshold tuning uses only `tune/` sessions; reported verdicts use `--exclude "tune/*"`.
- The PKLot results in `docs/evaluation.md` (a one-off run on public data) are proxy evidence only: never mark R-01, R-02 or R-05 met from them.
- The evaluation harness scores single frames (`classify_frame` + review threshold), not smoothed output: labeled frames are independent images. Smoothing is a dashboard layer on top.
- Validate slots with `slot_config_problem` (unique text ids, at least three points, at least `MIN_SLOT_AREA`). A zero-area slot must classify as unknown, never vacant.
- Scripts that run frames through the pipeline (timing, evaluation) must redirect `HISTORY_PATH` and replace `camera` with a `CameraMonitor` on a temporary path. Otherwise their first frame becomes the lot's camera reference (this happened once on 2026-11-17).
- In tests, scope temporary patches with `monkeypatch.context()`. `monkeypatch.undo()` also undoes the fixture's temporary paths and points the app at the user's real files.
- Don't overwrite the user's `slots.json`, `settings.txt` or `occupancy_history.csv` during tests. Point `app.CONFIG_PATH`, `app.SETTINGS_PATH` and `app.HISTORY_PATH` at a temporary directory and use `app.app.test_client()`. Add `slotsight/` to `sys.path` to import `app`. To test first-run downloads, also set `app.MODELS_DIR` and `app.models = detectors.ModelManager(<temp dir>)`. The `app` fixture also gives each test its own `tracker`, `empty_reference`, a `slot_classifier` with no weights (detector-only, unless a test adds weights), a `camera` whose reference frame is in the temporary folder, and temporary `LABELS_PATH` / `EVAL_FRAMES_DIR`, so tests never write the real `eval/labels.csv`.
- The user may have `python slotsight/app.py` running with Flask's debug reloader. Multi-file edits can restart it mid-change and stop it, so tell the user if their server on port 5000 has stopped.
- The target hardware is a student laptop, CPU-only. Keep the inference rate in mind (R-04: on-screen updates every ≤5 s, measured end to end).

## Verification

- Run `python -m pytest slotsight/tests` from the repository root after every change (about 10 s, no network). Add or update tests with each change, using the `app` and `fake_detector` fixtures, never the user's real files or downloads. The suite already covers: a blank frame gives all vacant, status polls never grow the history, and video input logs one set of rows per sampled frame. Keep those passing.
- For UI changes, run the app and check the map, the activity list, the source panel and the exports in a browser.
- Accuracy claims (R-01, R-02, R-05, R-07) need a run of `slotsight/eval/evaluate.py` on labeled lot frames. Timing claims (R-04) need a run of `slotsight/scripts/measure_update_interval.py`.
