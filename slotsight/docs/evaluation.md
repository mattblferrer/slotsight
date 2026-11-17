# SlotSight Evaluation

Measured evidence for the requirements in `requirements.md`.

## R-01, R-02, R-05, R-07: Accuracy

Runs of `slotsight/eval/evaluate.py` against hand-labeled frames of the target lot.

No reference set has been labeled yet (2026-11-17), so there is no accuracy evidence. The harness itself was checked on hand-built cases and a two-frame synthetic set, which is not evidence for any requirement.

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
