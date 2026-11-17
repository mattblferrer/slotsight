# Recording and labeling guide

How to collect the lot footage and hand labels that the remaining Milestone 1 evidence needs. The footage serves R-01 (per-slot accuracy), R-02 (free count), R-04 (timing on real footage), R-05 (evening and rain against daytime), R-06 (the demo) and R-07 (partly hidden slots), and it also feeds threshold tuning.

The steps are: record, extract frames, label them on the dashboard's `/label` page, tune the thresholds on the tune sessions, then report on the test sessions.

## 1. What to record

| Condition | Sessions (recordings) | Notes |
| --- | --- | --- |
| Daytime | at least 3, on different days | Include a busy period, such as a morning or after-work peak |
| Evening | at least 2 | Dusk or dark, with the lot's own lighting |
| Rain | at least 2, if it rains during the project | Rain falling or a wet lens. If it doesn't rain, R-05 is reported as untested for rain. |

- **Length:** 20 to 30 minutes per session. The PRD plans for 1 to 2 hours in total.
- **Partly hidden slots (R-07):** make sure some frames show slots partly hidden by a large vehicle, a pillar or a tree. Aim for at least 30 partly hidden slot labels across the test sessions.
- **Variety:** a mix of full, half-full and nearly empty lots. A lot that never changes tests very little.

### How much is enough

With about 12 slots, extracting a frame every 30 seconds gives about 50 frames, or 600 slot labels, per 30-minute session. Measured accuracy has an uncertainty that shrinks with the number of labels. At 90% accuracy, the 95% interval is about:

| Labels (or frames, for R-02) | ± interval |
| --- | --- |
| 100 | 5.9 points |
| 200 | 4.2 points |
| 600 | 2.4 points |

So aim for these totals in the **test** sessions:

- at least 600 slot labels per condition (R-01, R-05)
- at least 150 fully labeled frames overall (R-02 counts whole frames)

Frames from the same session are similar to each other, so more sessions beat longer ones.

## 2. Setting up the camera

- **Same view every time.** The slot layout belongs to one camera position. Mark the tripod or mount spot (tape on the ground, a photo of the setup) so every session matches it. If the view changes, redraw and save the layout before labeling that session. The dashboard's camera check warns when the view has moved away from the saved reference.
- **Phone settings:**
  - landscape, 1080p at 30 fps, no zoom
  - lock focus and exposure (tap and hold, AE/AF lock), and turn off auto-HDR changes
  - airplane mode or do-not-disturb, so calls don't stop the recording
  - keep the phone on a power bank
- **View:** elevated and looking down at the lot, with every slot in frame. Avoid pointing into the sun, and wipe the lens before each session.
- **Note the start time** of each recording. Enter it as "Footage recorded at" when you run the video through the dashboard, so the history gets real-world times (R-08).

### Storage

A phone records 1080p30 at roughly 100 to 150 MB per minute, so a 30-minute session is 3 to 5 GB. The project budget is about 50 GB. Extract frames soon after recording. Keep the videos until the evaluation, the timing run (R-04) and the demo (R-06) are done, then delete them. Videos and frames stay on your machine: `data/` is gitignored, and only `eval/labels.csv` is committed.

### Permission and privacy

- Get permission from whoever runs the lot before recording, and put up a notice if they want one.
- Frame the lot, not the people. The project excludes plate reading, so don't zoom in on plates or faces.
- Never commit videos or frames. Don't share them beyond what the project needs.

## 3. Extracting frames

From the repository root, for each recording:

```powershell
python slotsight/eval/extract_frames.py D:\lot\2026-11-20-morning.mp4 --condition daytime
python slotsight/eval/extract_frames.py D:\lot\2026-11-21-evening.mp4 --condition evening
python slotsight/eval/extract_frames.py D:\lot\2026-11-22-rain.mp4 --condition rain --set tune
```

- **Output:** one frame every 30 seconds (change it with `--every`), saved to `slotsight/data/eval_frames/<set>/<session>/`. The session name defaults to the file name.
- **`--set tune`:** keeps a whole session apart for threshold tuning. Put about **one session per condition** in `tune` and the rest in `test` (the default). The script refuses to put the same session in both, because frames seconds apart would make the test score look better than it is.
- **`--start` / `--end`:** skip the minutes spent setting up the camera.

## 4. Labeling

Start the dashboard (`python slotsight/app.py`) and open <http://127.0.0.1:5000/label>, or use the link under **Export history**. Each frame is shown with your slot layout.

- **Click a slot** to cycle unset → occupied → vacant. **Shift-click** (or tick the box) to mark it **partly hidden**. **Unset → vacant** fills every slot you haven't set yet.
- Check the **condition** for each frame. It defaults to the session's `--condition`.
- **Enter** saves and jumps to the next unlabeled frame. **←** and **→** move between frames. Labels go to `slotsight/eval/labels.csv`, and saving a frame again replaces its labels.

Slots start unset on purpose: the model never suggests labels, so the labels stay independent of what is being evaluated.

### Labeling rules

Label what a driver arriving at that moment would find. Use these rules every time, so labels made on different days agree.

- **Occupied:** a parked vehicle (car, motorcycle, tricycle, van, truck) makes the slot unusable. A car parked across two slots makes **both** occupied.
- **Vacant:** a driver could park there. A car driving past, or someone walking through, leaves the slot vacant.
- **Partly hidden:** mark it when roughly a quarter or more of the slot is covered by something in front of it (another vehicle, a pillar, a tree), whatever the slot's status.
- **Can't tell** (too dark, glare, fully blocked): leave the slot unset rather than guess. That frame then doesn't count for R-02, which needs every slot labeled.
- **Condition:**
  - daytime: natural daylight
  - evening: dusk or dark, with artificial light
  - rain: rain is visible or the lens is wet, whatever the time; rain wins over evening

## 5. Tuning the thresholds (tune sessions only)

```powershell
python slotsight/eval/evaluate.py --only "tune/*"
python slotsight/eval/evaluate.py --only "tune/*" --overlap-threshold 0.3 --review-threshold 0.6
python slotsight/eval/evaluate.py --only "tune/*" --overlap-threshold 0.5 --review-threshold 0.75
```

Detections are cached after the first run, so each new combination takes seconds. Pick the overlap and review thresholds that give the best accuracy **without raising "occupied shown as free"**, the costly error (R-07). Then set them: the overlap threshold in `config/settings.txt`, and the review threshold with the dashboard slider. Record the chosen values and why in `docs/PLAN.md`.

## 6. Reporting the evidence (test sessions only)

```powershell
python slotsight/eval/evaluate.py --exclude "tune/*" --note "test sessions, Nov 20-27" --write
python slotsight/scripts/measure_update_interval.py D:\lot\2026-11-20-morning.mp4 --seconds 120 --note "lot footage" --write
```

- **The first command** adds the R-01, R-02, R-05 and R-07 verdicts to `docs/evaluation.md`.
- **The second** adds an R-04 timing run on real footage.
- **For the R-06 demo,** play a recording under **Live source** with "Play video files at recorded speed" ticked. Every slot should be coloured on the map, with its status and the time of the last update.

Finally, update the requirement status table in `docs/PLAN.md` from the new evidence. Requirements count as met only on that evidence.
