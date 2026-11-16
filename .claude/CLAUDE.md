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

- `slotsight/app.py`: the backend (Flask routes, settings and slot persistence, detection, classification, annotation, history and export). `BASE_DIR` is `slotsight/`. `CONFIG_DIR`, `DATA_DIR` and `MODELS_DIR` are its `config/`, `data/` and `models/` subfolders.
- `slotsight/sources.py`: `SourceWorker`, which reads video files, streams and cameras on a background thread and samples one frame per `sample_interval_s`
- `slotsight/static/app.js`: dashboard logic (2 s polling, map rendering, slot polygon editor, config dialog, uploads, live-source controls, job progress)
- `slotsight/static/style.css`, `slotsight/templates/index.html`: the UI
- `slotsight/config/slots.json`: slot polygons, with coordinates **normalised 0–100** relative to frame width and height
- `slotsight/config/settings.txt`: JSON despite the extension (`lot_name`, `review_threshold`, `overlap_threshold`, `sample_interval_s`, `model_path`)
- `slotsight/data/occupancy_history.csv`: append-only log (`timestamp, slot_id, status, confidence, source, frame_time_s`). Gitignored, like the archived `occupancy_history-*.csv` files.
- `slotsight/models/`: model weights. `yolo11s.pt` is the default (COCO; vehicle classes 2 car, 3 motorcycle, 5 bus, 7 truck). Relative `model_path` values resolve here. Other `.pt` files are gitignored.
- `slotsight/scripts/measure_update_interval.py`: the R-04 timing test
- `slotsight/docs/`: `requirements.md`, `PLAN.md`, `evaluation.md`

API:
- `GET /api/status` (the snapshot plus `job`)
- `POST /api/process` (multipart `source`, optional `recorded_at`; images return immediately, videos return 202 and run as a job)
- `POST /api/source` (`target`, `realtime`, `recorded_at`), `POST /api/source/stop`, `GET /api/source`
- `POST /api/config`, `POST /api/review-threshold`
- `GET /api/history.csv`, `GET /api/history.xlsx`

## Run

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r slotsight/requirements.txt
pip install ultralytics          # optional; enables real detection
python slotsight/app.py          # http://127.0.0.1:5000
```

The local `.venv` already has Flask, OpenCV, numpy, openpyxl and ultralytics installed (CPU-only torch; no CUDA on this laptop).

## Rules and pitfalls

- The repo lives in OneDrive and is open in VS Code. Files have been deleted or recreated outside Claude's commands after moves: an archive was sent to the Recycle Bin, and a stray `settings.txt` reappeared. After moving or deleting files, re-check with `git status` and `ls`, and never assume an untracked file is still where it was.
- **Never let demo data mix with real inference.** Demo output is only for when no frame or model is present. An empty detection list on a real frame means every slot is vacant. A detector error must show up as an error, never as demo data.
- **GET endpoints must not write history.** Only processing a frame appends rows.
- "Unknown" must protect against false "free" readings (R-07). Don't hard-code confidence values for vacant slots.
- Keep defaults in sync across `app.py` constants, `settings.txt` and the UI defaults (slider, `POLL_MS` in `app.js`, `DASHBOARD_POLL_S` in the timing script).
- Resolve file paths against `BASE_DIR` (or `CONFIG_DIR`, `DATA_DIR`, `MODELS_DIR`), not the working directory. Editable config goes in `config/`, generated output in `data/`, and weights in `models/`. `sources.py` stays next to `app.py` because it's app code. `scripts/` is only for tools run by hand.
- Threading: run inference outside `state_lock` (status polls must never wait on it). Model calls are serialised by `inference_lock`. Lock order is source_lock → state_lock → model_lock; never take state_lock while holding inference_lock.
- Build DOM from server or config data with `textContent` or element creation, not raw `innerHTML` interpolation.
- Don't overwrite the user's `slots.json`, `settings.txt` or `occupancy_history.csv` during tests. Point `app.CONFIG_PATH`, `app.SETTINGS_PATH` and `app.HISTORY_PATH` at a temporary directory and use `app.app.test_client()`. Add `slotsight/` to `sys.path` to import `app`.
- The target hardware is a student laptop, CPU-only. Keep the inference rate in mind (R-04: on-screen updates every ≤5 s, measured end to end).

## Verification

- There is no test suite yet (adding a pytest suite is planned). Until then, check changes with the Flask test client against temporary config copies. At minimum, cover: a blank frame gives all vacant, `GET /api/status` doesn't grow the history, and video input logs one set of rows per sampled frame.
- For UI changes, run the app and check the map, the activity list, the source panel and the exports in a browser.
- Accuracy claims (R-01, R-02, R-05) need the evaluation harness described in `PLAN.md` Milestone 1C. Timing claims (R-04) need a run of `slotsight/scripts/measure_update_interval.py`.
