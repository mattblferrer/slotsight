# SlotSight Plan

The working plan for SlotSight: where the project stands against its requirements, what's next, and why. Keep it current. Update it whenever work lands, a review happens, or priorities change.

- Requirements (source of truth): [`requirements.md`](requirements.md), IDs R-01 to R-08
- Agent and project instructions: [`CLAUDE.md`](CLAUDE.md)

## How to use this document

- **Requirement status** shows the current verdict for each requirement. Update a row only when evidence changes (an evaluation run, a timing test, a demo).
- **Roadmap** lists the work, grouped into milestones. Tick the checkboxes as items land, and add new items under the right milestone, including the requirement IDs they affect.
- **Review log** records dated findings from code reviews. Add a new dated entry for each review instead of editing old ones.
- **Decisions log** records choices that change direction or scope, with the reasons.

## Requirement status

Last updated: 2026-11-17, after Milestone 1A (correctness fixes)

| ID | Requirement | Status | Current gap |
| --- | --- | --- | --- |
| R-01 | Classify each slot occupied/vacant (≥90% per-slot accuracy) | ⚠️ Partial | Classification is now real (1A fixed the fake-results bug and the overlap check). No ground-truth evaluation yet (1C). |
| R-02 | Report total free slots (matches ground truth in ≥90% of frames) | ⚠️ Partial | Count is computed from real classifications. No evaluation yet (1C). |
| R-03 | Accept a video file or live stream plus a slot-polygon config, and output timestamped statuses | ⚠️ Partial | Polygon config works. Video is reduced to frame 1. The upload field only accepts images. No live stream. |
| R-04 | Status updates at least every 5 s | ⚠️ Partial | The UI polls every 5 s, but nothing processes new frames. No timing test. |
| R-05 | Evening/rain accuracy ≥90% of daytime | ❌ No evidence | No labeled test sets per condition. No evaluation. |
| R-06 | Lot map shows each slot's status and time of last update | ⚠️ Partial | Map polygons are transparent, so status shows only on hover or in the uploaded preview image. |
| R-07 | Low-confidence slots show "unknown" | ⚠️ Partial | Vacant confidence now drops with weak or partial detections (1A). Not yet tested on real occluded cases (1C). |
| R-08 | Export history as CSV/Excel with slot ID, status and timestamp | ✅ Met, with one issue | New rows come only from processed frames (1A). The existing `occupancy_history.csv` still holds old demo rows (1C). |

### Definition of done

| ID | Done when |
| --- | --- |
| R-01 | `eval/evaluate.py` reports per-slot accuracy ≥ 90% on the reference set |
| R-02 | Free-count match rate ≥ 90% of test frames |
| R-03 | A video file and a live stream both produce timestamped per-slot rows covering the full input |
| R-04 | The measured update interval on target hardware is ≤ 5 s and is recorded in `docs/evaluation.md` |
| R-05 | Evening and rain accuracy are each ≥ 90% of daytime accuracy, reported per condition |
| R-06 | The demo shows every slot coloured by status on the map, with a visible last-update time |
| R-07 | Low-confidence and occluded test cases display as "unknown" |
| R-08 | CSV and XLSX exports contain slot ID, status and timestamp for every processed frame, with no polling noise |

## Roadmap

### Milestone 1: Meet the requirements (current focus)

#### 1A. Correctness fixes (blocking for every requirement)

- [x] **Separate demo mode from real inference.** When a real frame is supplied, an empty detection list means every slot is vacant. The demo pattern runs only when `frame is None` and no model is loaded. A detector exception is reported (logged, plus an error status in the response), never replaced by demo data. With a model loaded and no frame yet, slots show as unknown. A frame uploaded with no model returns HTTP 503. *(R-01, R-02)*
- [x] **Make `GET /api/status` read-only.** Only processing a frame should append history. Demo snapshots are never recorded. Threshold changes, config saves and file-less `POST /api/process` calls don't record either. *(R-08)*
- [x] **Replace the corner-in-polygon check** with an overlap ratio, compared with a configurable `overlap_threshold` in `settings.txt` (default 0.4). The ratio is the larger of: intersection / slot area, and intersection / box area. The second term keeps small vehicles in large slots from being missed. It is computed with `cv2.intersectConvexConvex` against the slot's convex hull. README updated. *(R-01)*
- [x] **Give vacant slots a real confidence score.** Vacant confidence = 1 − max(detection confidence × (overlap / overlap_threshold)²) over detections that overlap but fall below the threshold. Occupied confidence is the strongest matching detection's score. The detector now runs with a 0.1 confidence floor, so weak detections push slots to "unknown" instead of "vacant". *(R-07)*
- [x] **Make defaults consistent** across `app.py`, `settings.txt` and the slider default (review 0.7, overlap 0.4, `yolo11s.pt`). Resolve `model_path` against `BASE_DIR`. Reload the model when the path changes. `SLOTSIGHT_MODEL` now overrides `settings.txt`, as the README describes.
- [x] **Keep the upload snapshot after a config save.** Re-classify `latest_frame` with the new slots instead of falling back to demo data. Cached detections are reused, so no new inference runs.

#### 1B. Inputs and live updates (R-03, R-04)

- [ ] **Process the whole video.** Sample frames every N seconds of video time, configured as `sample_interval_s` with a default of 1–5 s. Log one set of rows per sampled frame, with the frame's *video timestamp* and the processing timestamp.
- [ ] **Add a live source.** Add a background worker that reads an RTSP/HTTP stream or webcam index (`cv2.VideoCapture(url)`), processes a frame at least every 5 s, updates the latest snapshot and appends history. Expose start/stop through the API and a UI control.
- [ ] **Let the upload field accept video** (`accept="image/*,video/*"`) and show progress or a summary for multi-frame jobs.
- [ ] **Add a timing test.** Measure the time from frame capture to a dashboard-visible update on the target laptop (CPU-only and GPU if available). Record the results in `docs/evaluation.md`. *(R-04 evidence)*

#### 1C. Dashboard and evidence (R-01, R-02, R-05, R-06, R-07, R-08)

- [ ] **Colour map polygons by status.** Add a `status-<status>` class in `renderMap` and style it. Show each slot's ID and last-update time on the map or in a pinned label. *(R-06)*
- [ ] **Build an evaluation harness** (`eval/evaluate.py`). Inputs: labeled frames (CSV of `frame, slot_id, true_status, condition`) plus `slots.json`. Outputs:
  - per-slot accuracy
  - free-count match rate
  - accuracy per condition (daytime, evening, rain) and the evening/rain ratio to daytime
  - a breakdown of "unknown" results on low-confidence and occluded cases

  *(R-01, R-02, R-05, R-07 evidence)*
- [ ] **Collect and label a reference test set** from the target lot, covering daytime, evening and rain, and including partly hidden cases. Store the labels (not raw video) in the repo.
- [ ] **Tune `review_threshold` and `overlap_threshold`** on a held-out split and record the chosen values and their results.
- [ ] **Fix the R-08 data issue.** Clean or rotate the existing demo-polluted `occupancy_history.csv`. Make sure exports contain only processed-frame rows, or mark the source clearly. *(R-08)*

### Milestone 2: Hardening and improvements (after Milestone 1)

#### Hardening

- [ ] Escape slot IDs and other server data before inserting them as HTML (use `textContent` or DOM building).
- [ ] Set `app.config["MAX_CONTENT_LENGTH"]`. Make `debug` follow an environment variable.
- [ ] Add try/catch to the upload handler. Remove unused imports and the dead `.slot.*` CSS, or start using it.
- [ ] Draw detection boxes in `annotate_frame`, or remove the "vehicle boxes" wording from the UI.
- [ ] Move `yolo11s.pt` out of git (download on first run or use Git LFS) and document Ultralytics' AGPL licence.
- [ ] Add a pytest suite covering classification, the empty-frame case, config validation, history writes and exports.

#### Improvement ideas (backlog, not required by the PRD)

- [ ] Per-slot "last changed" time, not just last-processed time
- [ ] Occupancy trend charts and peak-time view
- [ ] Smoothing over time (e.g. majority vote over the last N frames) to cut status flicker
- [ ] Camera-shift detection that warns when slot polygons no longer line up

## Review log

### 2026-11-17: Full code review against requirements (commit `6356d42`)

#### Confirmed bugs

These were reproduced with the Flask test client, using YOLO `yolo11s.pt` with config files copied to a temporary folder.

1. **Fake results on real frames (critical).** `classify_frame` uses the detector only when at least one vehicle is found (`app.py:180`). Otherwise it falls back to the demo pattern (`app.py:188-192`). A plain gray image returned 4 of 9 slots "occupied", labeled `model: YOLO`. `detect_vehicles` also swallows every exception (`app.py:168`), which sends errors into the same fallback.
2. **`GET /api/status` writes history.** It calls `snapshot()` with `record=True` (`app.py:235, 257`). Two GETs added 18 rows. One open tab adds about 155k rows a day. The existing `occupancy_history.csv` is about 78% demo rows.
3. **Video uploads process only frame 1** (`app.py:336-338`).

#### Other findings

- `vehicle_overlaps_slot` (`app.py:141-148`) marks a slot occupied if *any bounding-box corner* falls inside it, which over-counts neighbouring slots. The README says the box centre is used, and the PRD calls for a tunable overlap threshold. `polygon_area` is unused.
- Saving config (`app.py:290`) clears the snapshot but keeps `latest_frame`. The dashboard switches back to demo data, and moving the threshold slider then brings the old frame back.
- Defaults disagree. Code: threshold 0.45, `yolo11n.pt` (`app.py:30-31`). `settings.txt`: 0.7, `yolo11s.pt`. Slider default: 45 (`templates/index.html:241`).
- The model path is checked relative to the working directory, not `BASE_DIR` (`app.py:154`). Running from elsewhere silently drops into demo mode.
- The model is cached globally and never reloaded when `model_path` changes.
- `.slot.occupied` / `.slot.unknown` CSS (`static/style.css:451+`) is never used. `.slot-hit` is transparent (`static/style.css:368-376`).
- `annotate_frame` ignores `detections`, but the UI copy promises "vehicle boxes" (`templates/index.html:177`).
- Security (local-only prototype, low risk):
  - Slot IDs are inserted as raw HTML through `innerHTML` (`static/app.js:101, 162`), so a crafted ID can inject script.
  - `debug=True` (`app.py:376`) turns on Flask's interactive debugger.
  - The upload size check doesn't work because `uploaded.content_length` is usually empty (`app.py:326`).
- The upload submit handler has no try/catch (`static/app.js:344-362`).
- `math` and `random` are imported but unused.
- The 19 MB `yolo11s.pt` is committed to git. Ultralytics weights are AGPL.
- There is no automated test suite.

## Decisions log

| Date | Decision | Reason |
| --- | --- | --- |
| 2026-11-17 | Meet R-01 to R-08 before working on improvements | The 2026-11-17 review found that no requirement is fully met, and the core classifier can return fake data |
| 2026-11-17 | Demo data appears only when no model is loaded and no frame is supplied, and is never logged | Fake readings must never be mistaken for real ones, on screen or in exports |
| 2026-11-17 | Overlap ratio = max(intersection / slot area, intersection / box area), default threshold 0.4 | Slot-only ratios miss motorcycles in full-size slots. Box-only ratios miss large vehicles over small, distant slots. Final values get tuned in 1C. |
| 2026-11-17 | Partial or weak detections lower vacant confidence instead of being ignored | Wrong "free" readings are the costly error (R-07), so doubt should show as "unknown" |
