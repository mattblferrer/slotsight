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
