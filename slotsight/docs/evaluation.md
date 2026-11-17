# SlotSight Evaluation

Measured evidence for the requirements in `requirements.md`.

## R-01, R-02, R-05, R-07: Accuracy

Runs of `slotsight/eval/evaluate.py` against hand-labeled frames of the target lot.

No reference set has been labeled yet (2026-11-17), so there is no accuracy evidence. The harness itself was checked on hand-built cases and a two-frame synthetic set, which is not evidence for any requirement.

### PKLot proxy run: 2026-11-17 15:21

Public dataset, not the target lot: proxy evidence only (decided 2026-11-17). PKLot is daylight-only and has no partly-hidden labels, so it says nothing about evening (R-05) or R-07.

- Data: PKLot (CC BY 4.0, UFPR) via https://huggingface.co/datasets/Voxel51/PKLot; seeded subset of 603 frames, 32831 slot labels across 3 views
- Model: YOLO11 small; overlap threshold 0.4; review threshold 0.7 (not tuned)

| Measure | Pooled | PUCPR (100 slots) | UFPR04 (28) | UFPR05 (40) |
| --- | --- | --- | --- | --- |
| Per-slot accuracy (R-01 proxy) | 52.7% | 48.3% | 65.3% | 54.4% |
| Coverage (not unknown) | 80.7% | 89.3% | 68.8% | 68.4% |
| Accuracy where not unknown | 65.4% | 54.2% | 94.9% | 79.6% |
| Free-count exact match (R-02 proxy) | 43.7% (243/556) | 25.4% | 65.4% | 39.8% |
| Free count mean abs. error (slots) | | 40.8 | 1.0 | 5.8 |
| Occupied shown as free | 9167 | 7860 | 192 | 1115 |

| Condition | Labels | Accuracy | Coverage | Ratio to daytime (R-05 proxy) |
| --- | --- | --- | --- | --- |
| cloudy | 11251 | 61.8% | 83.1% | 120.4% |
| daytime | 10358 | 51.4% | 79.9% | 100.0% |
| rain | 11222 | 44.9% | 78.9% | 87.4% |

| True \ shown | occupied | vacant | unknown |
| --- | --- | --- | --- |
| occupied | 1718 | 9167 | 6300 |
| vacant | 6 | 15593 | 47 |

Threshold check (pooled per-slot accuracy / occupied shown as free; untuned defaults in bold):

| overlap \ review | 0.5 | 0.6 | 0.7 |
| --- | --- | --- | --- |
| 0.3 | 56.4% / 9103 | 54.5% / 9103 | 52.5% / 9100 |
| 0.4 | 56.5% / 9173 | 54.7% / 9172 | **52.7% / 9167** |
| 0.5 | 56.5% / 9201 | 54.7% / 9200 | 52.8% / 9195 |

#### Why the PKLot proxy scores are low (diagnosis, 2026-11-17)

- **Missed cars, not false alarms.** Of 17,185 occupied slot labels, 9,167 were shown as free and 6,300 as unknown. Of 15,646 vacant labels, only 53 were wrong. A car the detector misses leaves its slot looking confidently empty.
- **PUCPR (high, wide view of tightly packed rows):** the COCO-trained detectors find only about a quarter of the parked cars, so accuracy stays at 46-47 % for every model.
- **UFPR04/UFPR05 (closer, angled views):** most cars are found, but axis-aligned boxes over rotated slots, with modest confidence, push many occupied slots to unknown (coverage 68-69 %).
- **Inference size helps the closer views.** On a 63-frame sample (21 per view), running yolo11s at 1280 px instead of 640 raised UFPR05 from 48.8 % to 69.3 % and UFPR04 from 63.4 % to 73.4 %, and cut occupied-shown-free from 154 to 42. PUCPR stayed at 46.9 %, though its occupied-shown-free fell from 865 to 537.
- **No installed model does much better** (same sample, default thresholds):

| Model | PUCPR | UFPR04 | UFPR05 | Pooled | Occupied shown free | s/frame |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11 nano | 46.6 % | 57.8 % | 43.8 % | 47.8 % | 1,098 | 0.03 |
| YOLO11 small (default) | 46.7 % | 63.4 % | 48.8 % | 50.1 % | 1,019 | 0.06 |
| YOLO11 medium | 46.8 % | 62.0 % | 47.0 % | 49.5 % | 1,131 | 0.15 |
| YOLO26 nano | 46.7 % | 52.0 % | 37.8 % | 45.4 % | 1,101 | 0.03 |
| YOLO26 small | 46.7 % | 57.9 % | 42.9 % | 47.7 % | 1,091 | 0.06 |
| RT-DETRv2 R18 | 29.1 % | 30.3 % | 39.4 % | 31.8 % | 968 | 0.31 |
| RF-DETR nano | 46.2 % | 56.9 % | 42.1 % | 47.0 % | 137 | 0.17 |
| RF-DETR small | 46.0 % | 58.1 % | 43.5 % | 47.4 % | 137 | 0.32 |
| YOLOX-S (OpenCV) | 46.5 % | 56.2 % | 42.3 % | 47.2 % | 1,205 | 0.44 |

  RF-DETR shows 7-8x fewer occupied slots as free (it finds more cars but with low confidence, so they become unknown), which is safer for R-07, but it isn't more accurate.
- **What this means.** A general-purpose vehicle detector plus polygon overlap, as it stands, is far from 90 % on views like PKLot's. Published 95 %+ PKLot results use classifiers trained on per-slot image crops. Whether the target lot behaves like this depends on its view: a closer, lower camera with fewer, larger slots is the favourable case. The first labeled session from the target lot should be evaluated before anything else is tuned.

#### "Doesn't look empty" safeguard on the PKLot frames (one-off measurement, 2026-11-17)

The safeguard (`appearance.py`) turns a vacant reading into unknown when the slot looks too different from its learned empty appearance. It was run on the same 603 PKLot frames, in time order per view, with the default model and thresholds:

| Check threshold | Occupied shown as free | Missed cars caught | Truly empty slots turned unknown | Per-slot accuracy |
| --- | --- | --- | --- | --- |
| off | 9,167 | 0 | 0 of 15,646 | 52.7 % |
| **0.8 (default)** | **3,604 (-61 %)** | **5,563** | **628 (4.0 %)** | 50.8 % |
| 0.6 | 1,634 (-82 %) | 7,533 | 2,053 (13.1 %) | 46.5 % |

It improves safety (R-07), not accuracy: caught cars become unknown, which still counts as wrong for R-01. By view at 0.8, occupied shown as free drops from 7,860 to 2,666 (PUCPR), 192 to 104 (UFPR04), and 1,115 to 834 (UFPR05), at a cost of 449, 51 and 128 truly empty slots shown unknown. PKLot frames are hours apart; live sampling every 2 s gives the learned references far fresher lighting. Proxy evidence only.

#### Improving accuracy cheaply: what was tried on PKLot (one-off research, 2026-11-17)

Proxy evidence only (public data, not the target lot). Detector variants were scored on a 63-frame sample (21 per view) with yolo11s and the default thresholds:

| Approach | Pooled accuracy | Occupied shown as free | CPU per frame | Notes |
| --- | --- | --- | --- | --- |
| Detector, full frame, 640 px (before) | 50.1 % | 1,019 | 0.06 s | |
| Car matched to slot by box centre or ground-contact point instead of overlap | 49.7-50.1 % | 1,030-1,062 | 0.06 s | no gain |
| Crop to the slot area, 640 / 960 px | 50.5 % / 55.0 % | 1,152 / 918 | 0.04 / 0.08 s | small gain at 960 |
| Full frame, 1280 px | 56.9 % | 579 | 0.13 s | helps the closer views only |
| Slot area in 2×2 tiles (SAHI-style) | 57.1 % | 836 | 0.16 s | UFPR04 80 %, UFPR05 67 %; PUCPR unchanged |
| **Per-slot classifier (61k-parameter CNN)** | **97-99 %** on an unseen view | **-99 %** | **+0.016 s per 100 slots** | adopted |

The per-slot classifier is the approach of the PKLot/CNRPark literature: mAlexNet (Amato et al., 2017), 90-98 %; MobileNetV3 ([Yuldashev et al., Sensors 2023](https://www.mdpi.com/1424-8220/23/17/7642)), 98 %; and compressed MobileNet variants at 0.75 ms per slot on a Raspberry Pi 5. Cross-lot accuracy is known to vary: one distilled lightweight model reached 80.9 % on unseen datasets ([arXiv 2410.14705](https://arxiv.org/html/2410.14705v1)).

**The pipeline as built, leave-one-view-out.** For each view, the classifier was trained from scratch on the other two views, then the full dashboard pipeline (classifier, detector vote for "vacant", review threshold 0.7, optionally the "doesn't look empty" check) was scored on the unseen view. PUCPR is a different lot from UFPR04/UFPR05.

| Test view | Empty check | Per-slot accuracy | Coverage | Occupied shown as free |
| --- | --- | --- | --- | --- |
| PUCPR (unseen lot) | off | 97.5 % | 98.8 % | 27 of 9,332 |
| PUCPR (unseen lot) | 0.8 | 97.1 % | 98.3 % | 4 |
| UFPR04 | off | 98.2 % | 98.5 % | 1 |
| UFPR04 | 0.8 | 97.5 % | 97.7 % | 0 |
| UFPR05 | off | 97.3 % | 98.2 % | 20 |
| UFPR05 | 0.8 | 96.9 % | 97.7 % | 17 |
| **Pooled** | off | **97.6 %** | | 48 of 17,185 |
| **Pooled** | **0.8 (default)** | **97.1 %** | | **21** |

With the classifier in place, the empty check catches 27 more missed cars at a cost of 158 empty slots shown as unknown. It stays on by default (safety first, R-07); revisit with the target lot's data. With 10 labeled frames from the target view added to training, classifier accuracy rose by a further 0.4-1.2 points. Base weights for the app were then trained on all three views (`models/slot_classifier_base.pt`). PKLot is daylight-only, and its three views are elevated cameras over university lots, so the target lot's evening and rain frames and its own camera angle remain untested.

#### Re-measured after the 2026-11-18 review fixes

Slot patches are now cut with a fixed corner order (clockwise from the top-left) and the classifier is trained with quarter-turn augmentation; the base weights were retrained. Same leave-one-view-out protocol as above:

| Test view | Empty check | Per-slot accuracy | Coverage | Occupied shown as free |
| --- | --- | --- | --- | --- |
| PUCPR (unseen lot) | off / 0.8 | 98.2 % / 97.7 % | 99.0 % / 98.4 % | 26 / 4 |
| UFPR04 | off / 0.8 | 98.6 % / 97.9 % | 99.0 % / 98.3 % | 0 / 0 |
| UFPR05 | off / 0.8 | 96.4 % / 96.0 % | 97.8 % / 97.4 % | 17 / 14 |
| **Pooled** | **off / 0.8 (default)** | **97.8 % / 97.4 %** | | **43 / 18** of 17,185 |

At 0.8 the empty check caught 25 missed cars and turned 151 truly empty slots unknown.

#### R-02 proxy with the slot classifier (2026-11-18)

Same leave-one-view-out protocol; frames where every slot is labeled. Exact match is what R-02 asks for; "within 1" is context. Unknown slots count as not free.

| Test view (slots) | Frames | Exact match, check off / 0.8 | Within 1 slot | Mean abs. error (slots) | Frames showing too many free |
| --- | --- | --- | --- | --- | --- |
| PUCPR (100) | 189 | 47.6 % / 36.0 % | 67.7 % / 61.4 % | 1.66 / 1.99 | 6 / 2 |
| UFPR04 (28) | 191 | 80.1 % / 74.9 % | 93.7 % / 90.6 % | 0.35 / 0.54 | 0 / 0 |
| UFPR05 (40) | 176 | 71.0 % / 67.6 % | 86.4 % / 83.0 % | 1.23 / 1.38 | 10 / 8 |
| **Pooled** | 556 | **66.2 % / 59.4 %** | **82.6 % / 78.2 %** | | 16 / 10 |

With the detector alone the pooled exact match was 43.7 %. R-02 is far stricter than R-01: one wrong or unknown slot anywhere breaks the match, so the exact-match rate is roughly the per-slot accuracy raised to the number of slots. Reaching 90 % exact needs per-slot errors (wrong or unknown) of about 1 % for 10 slots, 0.5 % for 20 and 0.35 % for 30, against about 2.2 % here. The empty check lowers R-02 (more unknowns) while improving R-07 (fewer false "free" readings). Smoothing across frames, which this single-frame scoring leaves out, should help R-02 on live footage.

#### Low light: synthetic dusk and night (2026-11-18, weak evidence)

PKLot has no evening frames, so the unseen view's test patches were darkened: dusk at 45 % brightness with gamma 1.3 and mild noise; night at 20 % with gamma 1.6, more noise and an orange street-lamp tint. Classifier only (no detector vote), leave-one-view-out, review threshold 0.7. Real night footage (headlights, glare, pools of lamp light) is harder to model than this.

| Classifier training | Daylight | Dusk | Night |
| --- | --- | --- | --- |
| Before: brightness jitter only | 98.0 %, 44 occupied shown free | 89.0 %, 1,293 | **48.1 %, 16,501** (confidently wrong: coverage 99 %) |
| **Now: plus low-light augmentation** | **98.3 %, 41** | **94.4 %, 18** | 63.3 %, 238 (doubt shows as unknown: coverage 63 %) |

The earlier classifier called nearly every occupied slot "free" at night, the worst error for R-07. Training on darkened, noisy, tinted patches (half of each batch) costs nothing in daylight, makes dusk much better, and fails safe at night. Night accuracy still needs real night footage (R-05). The base weights were retrained this way. The full pipeline, leave-one-view-out in daylight: **97.8 % per-slot (97.4 % with the empty check), occupied shown as free 29 (13 with the empty check) of 17,185**.

## R-04: Update interval

### R-04 timing run: 2026-11-17 04:03

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (ultralytics bus.jpg sample), not lot footage; re-run with real lot footage
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: yolo11s.pt; frame size 1920x1080; 9 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.091 | 0.104 | 0.106 |
| Interval between readings (s) | 1.999 | 2.012 | 2.051 |

- Worst-case gap between dashboard updates: 4.05 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.16 s

### R-04 timing run: 2026-11-17 13:56

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO11 nano (ultralytics/yolo11n); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.051 | 0.065 | 0.112 |
| Interval between readings (s) | 1.997 | 2.013 | 2.019 |

- Worst-case gap between dashboard updates: 4.02 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.13 s

### R-04 timing run: 2026-11-17 13:57

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO11 medium (ultralytics/yolo11m); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.191 | 0.208 | 0.227 |
| Interval between readings (s) | 1.998 | 2.02 | 2.135 |

- Worst-case gap between dashboard updates: 4.13 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.36 s

### R-04 timing run: 2026-11-17 13:58

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO26 nano (ultralytics/yolo26n); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.059 | 0.061 | 0.103 |
| Interval between readings (s) | 1.998 | 2.01 | 2.016 |

- Worst-case gap between dashboard updates: 4.02 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.12 s

### R-04 timing run: 2026-11-17 14:00

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO26 small (ultralytics/yolo26s); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.095 | 0.1 | 0.137 |
| Interval between readings (s) | 1.997 | 2.014 | 2.016 |

- Worst-case gap between dashboard updates: 4.02 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.15 s

### R-04 timing run: 2026-11-17 14:01

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: RT-DETRv2 R18 (huggingface/rtdetr-v2-r18); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 1.185 | 1.227 | 1.991 |
| Interval between readings (s) | 1.994 | 2.084 | 2.715 |

- Worst-case gap between dashboard updates: 4.71 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 6.71 s

### R-04 timing run: 2026-11-17 14:02

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: RF-DETR nano (rfdetr/nano); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.595 | 0.608 | 0.926 |
| Interval between readings (s) | 1.987 | 2.054 | 2.253 |

- Worst-case gap between dashboard updates: 4.25 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 5.18 s

### R-04 timing run: 2026-11-17 14:03

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: RF-DETR small (rfdetr/small); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 1.064 | 1.106 | 1.37 |
| Interval between readings (s) | 1.988 | 2.035 | 2.103 |

- Worst-case gap between dashboard updates: 4.10 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 5.47 s

### R-04 timing run: 2026-11-17 14:05

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); model comparison on the target laptop, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLOX-S (OpenCV zoo) (opencv/yolox-s); frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.104 | 0.142 | 0.249 |
| Interval between readings (s) | 1.995 | 2.039 | 2.046 |

- Worst-case gap between dashboard updates: 4.05 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.29 s

### R-04 timing run: 2026-11-17 19:31

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); yolo11s + slot classifier (base) + doesn't-look-empty check, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO11 small (ultralytics/yolo11s); inference size 640; frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.12 | 0.203 | 0.427 |
| Interval between readings (s) | 1.995 | 2.126 | 2.2 |

- Worst-case gap between dashboard updates: 4.20 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.63 s

### R-04 timing run: 2026-11-18 02:27

- Source: footage_1080p.mp4 (file, played at recorded speed) - synthetic 1080p30 footage (bus.jpg sample, static); yolo11s + slot classifier (base, after the 2026-11-18 review fixes) + doesn't-look-empty check, not lot evidence
- Machine: AMD64 Family 26 Model 36 Stepping 0, AuthenticAMD / CPU only / Python 3.14.7
- Model: YOLO11 small (ultralytics/yolo11s); inference size 640; frame size 1920x1080; 6 slots
- Sample interval 2 s; dashboard poll 2 s; 31 readings over 60 s

| Measure | Mean | p95 | Max |
| --- | --- | --- | --- |
| Processing time per frame (s) | 0.108 | 0.179 | 0.383 |
| Interval between readings (s) | 1.995 | 2.087 | 2.163 |

- Worst-case gap between dashboard updates: 4.16 s (target <= 5 s): **PASS**
- Worst-case age of the displayed reading: 4.55 s
