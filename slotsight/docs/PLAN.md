# SlotSight Plan

The working plan for SlotSight: where the project stands against its requirements, what's next, and why. Keep it current. Update it whenever work lands, a review happens, or priorities change.

- Requirements (source of truth): [`requirements.md`](requirements.md), IDs R-01 to R-08
- Measured evidence: [`evaluation.md`](evaluation.md)
- Agent and project instructions: [`.claude/CLAUDE.md`](../../.claude/CLAUDE.md)

Paths in this document are relative to the `slotsight/` folder unless they start with `slotsight/` or `.claude/`.

## How to use this document

- **Requirement status** shows the current verdict for each requirement. Update a row only when evidence changes (an evaluation run, a timing test, a demo).
- **Roadmap** lists the work, grouped into milestones. Tick the checkboxes as items land, and add new items under the right milestone, including the requirement IDs they affect.
- **Review log** records dated findings from code reviews. Add a new dated entry for each review instead of editing old ones.
- **Decisions log** records choices that change direction or scope, with the reasons.

## Requirement status

Last updated: 2026-11-17, after the Milestone 1C map and evaluation harness work

| ID | Requirement | Status | Current gap |
| --- | --- | --- | --- |
| R-01 | Classify each slot occupied/vacant (≥90% per-slot accuracy) | ⚠️ Partial | Classification is real (1A). The harness (`eval/evaluate.py`) is ready, but no labeled reference set exists yet. |
| R-02 | Report total free slots (matches ground truth in ≥90% of frames) | ⚠️ Partial | Count is computed from real classifications. The harness measures the free-count match rate. Waiting on the reference set. |
| R-03 | Accept a video file or live stream plus a slot-polygon config, and output timestamped statuses | ✅ Met | Videos are sampled end to end and live streams/cameras are read continuously, both producing timestamped per-slot rows (1B). Tested with synthetic footage and a simulated stream; not yet with a real RTSP camera. |
| R-04 | Status updates at least every 5 s | ✅ Met (provisional) | Worst-case on-screen update gap 4.05 s on the target laptop, CPU only (`docs/evaluation.md`). Measured on synthetic 1080p footage; re-run with real lot footage. |
| R-05 | Evening/rain accuracy ≥90% of daytime | ❌ No evidence | The harness reports accuracy per condition and the evening/rain ratios to daytime. No footage recorded yet. |
| R-06 | Lot map shows each slot's status and time of last update | ✅ Met | Every slot is filled in its status colour and carries a pinned ID chip with a status symbol. The map corner shows the update time, and the hover/focus tooltip and activity list show each slot's time (1C). Checked in a browser on a sample frame; re-check in the demo with lot footage. |
| R-07 | Low-confidence slots show "unknown" | ⚠️ Partial | Vacant confidence drops with weak or partial detections (1A). The harness checks that occluded slots are never shown wrong. Waiting on labeled occluded cases. |
| R-08 | Export history as CSV/Excel with slot ID, status and timestamp | ✅ Met | New rows come only from processed frames (1A). The demo-polluted history was archived and a clean file started (1B). Video rows also carry `frame_time_s`, and a real-world time when a recording start is given. |

### Definition of done

| ID | Done when |
| --- | --- |
| R-01 | `eval/evaluate.py` reports per-slot accuracy ≥ 90% on the reference set |
| R-02 | Free-count match rate ≥ 90% of test frames |
| R-03 | A video file and a live stream both produce timestamped per-slot rows covering the full input |
| R-04 | The measured update interval on target hardware is ≤ 5 s and is recorded in `docs/evaluation.md` |
| R-05 | Evening and rain accuracy are each ≥ 90% of daytime accuracy, reported per condition |
| R-06 | The demo shows every slot coloured by status on the map, with a visible last-update time |
| R-07 | Low-confidence and occluded test cases display as "unknown" (scored as: no occluded case shows a wrong confident status; each is correct or unknown, see decisions log) |
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

- [x] **Process the whole video.** Frames are sampled every `sample_interval_s` seconds of video time (default 2 s) in a background job (`sources.py`). Each sampled frame logs one row per slot with its video offset (`frame_time_s`). With an optional "recorded at" start time, `timestamp` = start + offset.
- [x] **Add a live source.** `SourceWorker` reads an RTSP/HTTP URL, webcam index or local video path, drains frames continuously, processes one every `sample_interval_s` seconds, and reconnects up to 5 times after dropouts. The API is `POST /api/source`, `POST /api/source/stop` and `GET /api/source`, with matching UI controls. Stream credentials are removed from displayed and logged names. Inference runs outside the state lock, so status polls stay fast (≤20 ms measured during processing).
- [x] **Let the upload field accept video** (`accept="image/*,video/*"`). Job progress, frame count and per-frame processing time show under the source panel.
- [x] **Add a timing test.** `slotsight/scripts/measure_update_interval.py` runs a source at recorded speed through the full pipeline and reports the results to `docs/evaluation.md`. First run on this laptop (CPU only, synthetic 1080p footage): 0.09 s per frame, 4.05 s worst-case on-screen gap. The model now runs a warm-up inference when it loads, which removed a 1.7 s first-frame delay. *(R-04 evidence)*
- [ ] **Re-run the timing test with real lot footage** (and a real camera stream, if available) to confirm R-04 under real conditions. *(R-04)*

#### 1C. Dashboard and evidence (R-01, R-02, R-05, R-06, R-07, R-08)

- [x] **Colour map polygons by status.** `renderMap` draws each slot as an SVG polygon with a `status-<status>` class (green/red, plus yellow with a dashed edge for unknown). It adds a pinned chip with the slot ID and a status symbol (○ ● ?), so status doesn't rely on colour alone. The map corner shows "Updated hh:mm:ss", plus the capture time or video offset when known. The hover/focus tooltip and the activity list show each slot's status, confidence and update time. The backend adds a per-slot `updated_at`, which stays at the frame's processing time when a config or threshold change re-classifies it. The tooltip now follows each 2 s refresh instead of showing the old reading. The activity list is built with DOM calls instead of `innerHTML`. *(R-06)*
- [x] **Build an evaluation harness** (`eval/evaluate.py`). Inputs: `eval/labels.csv` (`frame, slot_id, true_status, condition, occluded`), frames in `data/eval_frames/`, and `slots.json`. It runs the dashboard's own detection, overlap and review-threshold code. Outputs, with PASS/FAIL/NO DATA verdicts:
  - per-slot accuracy (unknown counts as wrong) with coverage, a confusion table and a per-slot table
  - free-count match rate on fully labeled frames, plus mean error and over-reported frames
  - accuracy per condition (daytime, evening, rain) and the evening/rain ratio to daytime
  - occluded cases split into correct, unknown and wrong (including false "free"), plus how many wrong guesses the review threshold held back as unknown

  Detections are cached (`data/eval_cache.json`), so threshold comparisons don't repeat inference. `--write` adds the report to the accuracy section of `docs/evaluation.md`. The scoring logic, label validation and an end-to-end YOLO run were checked on hand-built and synthetic cases. *(R-01, R-02, R-05, R-07 evidence)*
- [ ] **Collect and label a reference test set** from the target lot, covering daytime, evening and rain, and including partly hidden cases. Store the labels (not raw video) in the repo. *Blocked: no footage recorded yet (2026-11-17).*
- [ ] **Tune `review_threshold` and `overlap_threshold`** on a held-out split and record the chosen values and their results. `evaluate.py --overlap-threshold/--review-threshold` re-scores from cached detections. *Blocked on the reference set.*
- [ ] **Re-check R-06 in the demo with lot footage** once the reference footage exists. *(R-06)*
- [x] **Fix the R-08 data issue.** The old file was archived as `occupancy_history-2026-11-17.csv` (gitignored) and a clean file in the new format starts on the next run. *(R-08)*

### Milestone 2: Hardening and improvements (after Milestone 1)

#### Hardening

- [ ] Escape slot IDs and other server data before inserting them as HTML (use `textContent` or DOM building).
- [ ] Set `app.config["MAX_CONTENT_LENGTH"]`. Make `debug` follow an environment variable.
- [ ] Add try/catch to the upload handler. Remove unused imports and the dead `.slot.*` CSS, or start using it.
- [ ] Draw detection boxes in `annotate_frame`, or remove the "vehicle boxes" wording from the UI.
- [x] Move `yolo11s.pt` out of git and document Ultralytics' AGPL licence. The weights and `config/slots.json` / `settings.txt` are no longer tracked (gitignored, with local copies kept). On first run the app creates the config from its defaults and downloads `yolo11s.pt` (hash-checked) in the background, showing progress on the dashboard. Offline, it stays in labelled demo mode with the error shown. The licence is documented in the README and `models/README.md`. `ultralytics` is now a regular requirement. Checked on a copy of a fresh clone.
- [x] **Detection models from several providers** (`detectors.py`): Ultralytics YOLO11/YOLO26, Hugging Face RT-DETRv2, Roboflow RF-DETR and OpenCV YOLOX behind one interface. Install and switch from **Configure slots > Detection model** or `scripts/models.py`. `settings.txt` uses `model` ids, and old `model_path` values migrate automatically. On this laptop (CPU, sample 810x1080 image): yolo11n 0.02 s, RF-DETR nano 0.09 s, RT-DETRv2 0.22 s and YOLOX-S 0.27 s per frame, all finding the sample bus. RF-DETR small has not been run yet (same code path as nano).
- [x] Purge `yolo11s.pt` from git history and force-push (2026-11-17). All local branches were rewritten with `git filter-branch`, and `main` was force-pushed with a lease (`f4cd09f` → `e2596cd`). The repo shrank from about 17 MB to about 100 KB. A pre-purge backup of every branch is in `slotsight-backup-2026-11-17.bundle` next to the repo folder.
- [ ] Re-run the R-04 timing test with any model other than yolo11s before relying on it (R-04 evidence covers yolo11s only). *(R-04)*
- [ ] Add a pytest suite covering classification, the empty-frame case, config validation, history writes and exports.

#### Improvement ideas (backlog, not required by the PRD)

- [ ] Per-slot "last changed" time, not just last-processed time
- [ ] Occupancy trend charts and peak-time view
- [ ] Smoothing over time (e.g. majority vote over the last N frames) to cut status flicker
- [ ] Camera-shift detection that warns when slot polygons no longer line up

## Review log

File and line references in entries dated before the 2026-11-17 reorganization point at the old top-level layout. Those files now live under `slotsight/` (documents under `slotsight/docs/`), and line numbers have shifted since.

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

| Date | Decision | Reason | Decided by |
| --- | --- | --- | --- |
| 2026-11-17 | Meet R-01 to R-08 before working on improvements | The 2026-11-17 review found that no requirement is fully met, and the core classifier can return fake data | User |
| 2026-11-17 | Demo data appears only when no model is loaded and no frame is supplied, and is never logged | Fake readings must never be mistaken for real ones, on screen or in exports | Claude (1A, before the ask-first rule) |
| 2026-11-17 | Overlap ratio = max(intersection / slot area, intersection / box area), default threshold 0.4 | Slot-only ratios miss motorcycles in full-size slots. Box-only ratios miss large vehicles over small, distant slots. Final values get tuned in 1C. | Claude (1A, before the ask-first rule) |
| 2026-11-17 | Partial or weak detections lower vacant confidence instead of being ignored | Wrong "free" readings are the costly error (R-07), so doubt should show as "unknown" | Claude (1A, before the ask-first rule) |
| 2026-11-17 | R-04 is measured end to end on screen: sampling + processing + dashboard polling. Sources sample every 2 s and the dashboard polls every 2 s. | This matches what the operator actually sees and the requirement's "near real-time" rationale. 4.05 s worst case measured on the target laptop. | User (Claude's recommendation) |
| 2026-11-17 | Uploads take an optional "recorded at" start time. History `timestamp` = start + frame offset; without one, it's the processing time. Live frames always use capture time. | R-08's peak-time analysis and R-05's per-condition test sets need real-world times for recorded footage | User (Claude's recommendation) |
| 2026-11-17 | Archive the demo-polluted history as `occupancy_history-2026-11-17.csv` and start a clean file with the new `frame_time_s` column. Older-format files are upgraded in place automatically. | Keeps old data without mixing about 78% demo rows into new exports | User (Claude's recommendation) |
| 2026-11-17 | Only one frame source runs at a time; starting a new one stops the previous one | PRD scope is one camera and one lot | Claude (small and reversible; noted in the summary) |
| 2026-11-17 | Reorganized the repository: only `README.md` and `.gitignore` (plus `.claude/` and the gitignored `.venv`) stay at the top level. Code, data and the model live in `slotsight/`, and documents in `slotsight/docs/`. `CLAUDE.md` moved to `.claude/CLAUDE.md`. | A cleaner top level. `.claude/CLAUDE.md` is still loaded automatically by Claude Code. The venv stays put because virtual environments break when moved. | User (layout options proposed by Claude) |
| 2026-11-17 | Inside `slotsight/`: `config/` holds `settings.txt` and `slots.json` (files you edit), `data/` holds the occupancy history and its archives (generated, gitignored), and `models/` holds model weights, where relative `model_path` values resolve. `sources.py` stays next to `app.py`, and `requirements.txt` stays in `slotsight/`. | Separates editable config from growing output. `scripts/` stays for tools run by hand. `requirements.txt` stays where tools and people expect it. | User (Claude's recommendations for the history file, `sources.py` and `requirements.txt`) |
| 2026-11-17 | Only the default `yolo11s.pt` is tracked in `models/`. Other downloaded models are gitignored. | Model files are large, and committing them by accident bloats the repo for good | Claude (small and reversible) |
| 2026-11-17 | The evaluation harness lives in `slotsight/eval/` (`evaluate.py` plus a tracked `labels.csv`). Labels have one row per slot per frame: `frame, slot_id, true_status, condition, occluded`. Frames stay out of git in `data/eval_frames/`. | Keeps the plan's `eval/evaluate.py` name and keeps the labels beside the tool. The per-row layout follows CNRPark-EXT (one free/busy label per slot image, with weather recorded for each). The `occluded` column gives R-07 its test cases. The PRD says to store labels, not raw footage. Research: [Ultralytics parking management](https://docs.ultralytics.com/guides/parking-management) ([source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/parking_management.py), AGPL-3.0, no code copied) has no per-slot output or evaluation; CNRPark-EXT layout per [a mirror of its README](https://blog.csdn.net/chengyq116/article/details/81318800) (the original site was unreachable); PKLot's weather split is unbalanced ([Voxel51](https://docs.voxel51.com/dataset_zoo/datasets_hf/pklot.html)), so the report shows rows per condition and flags small samples. | User (Claude's recommendation) |
| 2026-11-17 | Scoring: for R-01, R-02 and R-05, "unknown" counts as wrong, with coverage reported beside it. R-02 compares the shown free count (unknowns are not free) with the true free count, on frames where every slot is labeled. R-07 passes when no occluded case shows a wrong confident status (each is correct or unknown). Wrong "free" readings are reported separately. | A strict score can't be gamed by marking everything unknown, while R-07 still penalises only the dangerous outcome. Requiring correct occluded slots to show as unknown would punish correct answers. | User (Claude's recommendation) |
| 2026-11-17 | R-06 map: polygons filled by status, a pinned ID chip with a status symbol on each slot, and one "Updated" stamp in the map corner. Each slot's time is in the tooltip and the activity list. | Readable at 30 slots, and all slots currently share one update time. Ultralytics' solution colours only outlines and has no per-slot time, which isn't enough for R-06's "pinpointable" status and timestamp. | User (Claude's recommendation) |
| 2026-11-17 | A slot's `updated_at` is when its frame was processed. A config or threshold change re-classifies the same frame without changing that time. With a model loaded but no frame yet, no update time is shown. | The map must not make an old reading look new (R-06) | Claude (small and reversible) |
| 2026-11-17 | R-05's overall verdict is INCOMPLETE when every tested condition passes but evening or rain has no labeled frames. It is NO DATA without a daytime baseline. | The PRD allows that only conditions that occur during the project can be evaluated, and the report should say which ones were left out | Claude (small and reversible) |
| 2026-11-17 | No frame-extraction or labeling helper for now; build the harness first and test it on synthetic data | No lot footage has been recorded yet | User |
| 2026-11-17 | Stop tracking model weights. The default `ultralytics/yolo11s` downloads on first run from Ultralytics' GitHub release (hash-checked). Users can install other models from several providers. | The repo should not carry a 19 MB AGPL binary, and users should be able to choose detectors, including non-AGPL ones. Git LFS was rejected: it needs extra tooling, has bandwidth quotas, and would still redistribute the weights. | User (download on first run was Claude's recommendation; multiple providers was the user's addition) |
| 2026-11-17 | Providers: Ultralytics YOLO (AGPL-3.0), Hugging Face `transformers` RT-DETRv2 (`PekingU/rtdetr_v2_r18vd`, pinned revision), Roboflow RF-DETR nano/small (Apache-2.0 sizes only), OpenCV-zoo YOLOX-S via `cv2.dnn` (Apache-2.0, pinned revision). torchvision detectors are left out because their weights' licence terms aren't stated. Research: [RF-DETR](https://rfdetr.roboflow.com) ([PyPI](https://pypi.org/project/rfdetr/)), [RT-DETRv2 model card](https://huggingface.co/PekingU/rtdetr_v2_r18vd/discussions/1), [OpenCV YOLOX](https://huggingface.co/opencv/object_detection_yolox) (its Apache-2.0 reference pre/post-processing was followed), [torchvision weights licence question](https://discuss.pytorch.org/t/license-clarification-for-torchvision-resnet50-weights-and-fine-tuned-checkpoints/225542). | Different licences, sizes and CPU speeds, so users can trade accuracy, speed and licence | User |
| 2026-11-17 | The dashboard Configure dialog and a `scripts/models.py` CLI share one registry. Only one model stays loaded; installs and loads run on a background thread, and status polls never wait on them. | Approachable for first-time users, while the CLI lets offline machines be prepared ahead of time | User (Claude's recommendation) |
| 2026-11-17 | yolo11s stays the default, downloaded automatically on first run | It is the fastest option on CPU and the model behind the R-04 evidence | User (Claude's recommendation) |
| 2026-11-17 | `settings.txt` stores `model` as a registry id. Legacy `model_path` values (and `SLOTSIGHT_MODEL` paths) are converted; custom weights use `ultralytics/file:<name>.pt`. | An explicit provider avoids guessing from file names | User (Claude's recommendation) |
| 2026-11-17 | Gitignore `config/slots.json` and `config/settings.txt`. The app writes both from its built-in defaults when missing, so there are no tracked example files. | A lot's layout is local, and keeping defaults in one place (app.py) avoids a second copy to keep in sync | User (Claude's recommendation) |
| 2026-11-17 | Document the AGPL-3.0 implications in the README and `models/README.md`; add no LICENSE for SlotSight itself | Licensing SlotSight is the user's decision | User (Claude's recommendation) |
| 2026-11-17 | Purge `yolo11s.pt` from git history and force-push | Shrinks clones from about 17 MB, and stops the repo distributing the weights from old commits | User |
| 2026-11-17 | `ultralytics` became a regular requirement (other providers stay optional). Model states `downloading`/`loading` keep slots "unknown"; demo data appears only for `missing`/`needs_package`/`error`. | A first-time user who follows the README gets the default model working, and demo data never stands in for a model that is about to be ready | Claude (small and reversible) |
