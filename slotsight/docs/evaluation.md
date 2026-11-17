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
