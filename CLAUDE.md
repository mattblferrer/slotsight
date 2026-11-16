# CLAUDE.md

## Project

SlotSight is a prototype that counts parking occupancy from one fixed camera at one lot. A pretrained YOLO detector finds vehicles. Each one is matched against slot polygons defined by the user, and every slot is classified as occupied, vacant or unknown. A Flask dashboard shows the results and exports the history.

- Product requirements (the source of truth): [`requirements.md`](requirements.md), IDs R-01 to R-08
- Working plan (requirement status, roadmap, review log, decisions): [`PLAN.md`](PLAN.md)

## Current goal and priorities

1. **Bring the project back to meeting R-01–R-08 first.** Work through Milestone 1 in `PLAN.md` in order (1A correctness, then 1B inputs and live updates, then 1C dashboard and evidence). Tick off checkboxes as items land.
2. **Only then** move on to Milestone 2 (hardening and improvements) in `PLAN.md`.
3. Don't add features outside the PRD scope while requirements are unmet. These are out of scope entirely: payments, reservations, plate reading, multi-camera stitching, user accounts.
4. When a change affects a requirement, name the requirement ID in the commit message or PR description, and update the requirement status table in `PLAN.md`. Record reviews in its review log and direction changes in its decisions log.
5. A requirement counts as met only when its acceptance criterion has evidence: evaluation output, a timing measurement or a demo. "Code exists" is not enough.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install ultralytics   # optional; enables real detection
python app.py              # http://127.0.0.1:5000
```

The local `.venv` already has Flask, OpenCV, numpy, openpyxl and ultralytics installed.

## Layout

- `app.py`: the whole backend (Flask routes, settings and slot persistence, detection, classification, annotation, history and export)
- `static/app.js`: dashboard logic (polling, map rendering, slot polygon editor, config dialog, uploads)
- `static/style.css`, `templates/index.html`: the UI
- `slots.json`: slot polygons, with coordinates **normalised 0–100** relative to frame width and height
- `settings.txt`: JSON despite the extension (`lot_name`, `review_threshold`, `model_path`)
- `occupancy_history.csv`: append-only log (`timestamp, slot_id, status, confidence, source`). It is gitignored.
- `yolo11s.pt`: the model weights (COCO; vehicle classes 2 car, 3 motorcycle, 5 bus, 7 truck)

API: `GET /api/status`, `POST /api/process` (multipart `source`), `POST /api/config`, `POST /api/review-threshold`, `GET /api/history.csv`, `GET /api/history.xlsx`.

## Rules and pitfalls

- **Never let demo data mix with real inference.** Demo output is only for when no frame or model is present. An empty detection list on a real frame means every slot is vacant. A detector error must show up as an error, never as demo data.
- **GET endpoints must not write history.** Only processing a frame appends rows.
- "Unknown" must protect against false "free" readings (R-07). Don't hard-code confidence values for vacant slots.
- Keep defaults in sync across `app.py` constants, `settings.txt` and the UI slider defaults.
- Resolve file paths against `BASE_DIR`, not the working directory.
- Build DOM from server or config data with `textContent` or element creation, not raw `innerHTML` interpolation.
- Don't overwrite the user's `slots.json`, `settings.txt` or `occupancy_history.csv` during tests. Point `app.CONFIG_PATH`, `app.SETTINGS_PATH` and `app.HISTORY_PATH` at a temporary directory and use `app.app.test_client()`.
- The target hardware is a student laptop, possibly CPU-only. Keep the inference rate in mind (R-04: updates every ≤5 s).

## Verification

- There is no test suite yet (adding a pytest suite is planned). Until then, check changes with the Flask test client against temporary config copies. At minimum, cover: a blank frame gives all vacant, `GET /api/status` doesn't grow the history, and video input logs several frames.
- For UI changes, run the app and check the map, the activity list and the exports in a browser.
- Accuracy claims (R-01, R-02, R-05) need the evaluation harness described in `PLAN.md` Milestone 1C.
