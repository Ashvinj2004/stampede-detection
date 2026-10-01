# Crowd Density Early-Warning System

**Real-time crowd monitoring with density forecasting and threshold-based alerting, built on YOLOv8 detection, CSRNet density estimation, and a from-scratch DeepSORT-style tracker — with the design of each stage decided by measurement.**

*Independent project · VIT-AP University · 2026*

---

## What this does

Crowd crush incidents rarely happen without warning: density builds before it becomes dangerous. This system monitors camera feeds, converts person counts into real-world density using an academic pedestrian-safety standard, forecasts when density will cross a danger threshold, and raises alerts before that point is reached.

Per camera it:

1. Detects people with YOLOv8
2. Tracks them across frames with a Kalman filter, Hungarian assignment, and an appearance descriptor
3. Converts count into density (people/m²) using a per-camera area profile
4. Classifies risk against **Fruin Level of Service** thresholds
5. Forecasts time-to-critical by extrapolating the smoothed density trend
6. Fires escalation alerts — on-screen banner, audible tone, timestamped log
7. Selectively invokes **CSRNet** in high-density conditions where detection degrades

Multiple cameras run concurrently in independent threads on a tiled dashboard.

---

## Honest scope

This is a **density early-warning system**, not a stampede predictor. It forecasts when crowd density will cross established safety thresholds. It does not predict panic, crowd turbulence, or crush events — those depend on behavioural and structural factors this system does not measure.

---

## Experiment 1 — Which counting model to trust

### Method

Five frames were sampled at even intervals from three clips representing different camera geometries. Each frame was manually annotated by hand-counting visible people. YOLOv8n (1280px input) and ShanghaiTech Part B–pretrained CSRNet were evaluated against those counts.

| Clip | Geometry | Typical count |
|---|---|---|
| Plaza | Elevated, open floor | ~35 |
| Corridor | Ground-level, chokepoint | ~16 |
| Aerial | High overhead, crossing | ~80 |

### Results

| Clip | MAE — YOLO | MAE — CSRNet | Detection rate — YOLO | — CSRNet |
|---|---|---|---|---|
| Plaza | **1.3** | 22.1 | 98.5% | 41.2% |
| Corridor | **1.2** | 3.0 | 99.6% | 120.9% |
| Aerial | 14.3 | **9.9** | 82.6% | 87.1% |

### Interpretation

**YOLO substantially outperforms CSRNet at low-to-moderate density.** On the plaza clip YOLO reached 98.5% of ground truth (MAE 1.3) while CSRNet found only 41%.

**The ranking reverses at high density.** On the aerial clip YOLO's detection rate fell to 82.6% as occlusion increased, while CSRNet reached 87.1% with lower absolute error. This crossover is the empirical basis for the fusion policy.

**Input resolution dominated model choice.** Raising YOLO's input from 640px to 1280px took detections on the aerial clip from 13 to 52 — roughly 4× with no model change. For small-object detection in dense scenes, resolution mattered more than model capacity.

**A note on the CSRNet failure.** The plaza result is consistent with domain shift — the weights come from street-level photography, and the plaza is a high overhead view of a bright, low-texture floor. This is a hypothesis that fits the evidence rather than a tested conclusion; notably, Shanghai Part B is specifically the *sparse* crowd subset, which sits awkwardly with CSRNet performing best on the densest clip. Testing it properly would need weights trained on a known-different distribution.

### Resulting fusion policy

```
if yolo_count >= 50:          →  activate CSRNet (hysteresis: deactivate below 40)
    fused = max(yolo, csrnet) →  both undercount at high density; take the higher
else:                         →  trust YOLO alone (98%+ accurate in this regime)
```

Separate activation (50) and deactivation (40) thresholds prevent chattering when counts oscillate around a single cutoff. CSRNet runs every 20th frame while active, with the result cached between runs — inference costs seconds per frame on CPU, and crowd density does not change meaningfully at sub-second scales.

---

## Experiment 2 — Does appearance-based tracking help?

The tracker was built from scratch rather than imported, in stages, with each stage verified before the next was added.

### Components

**Kalman filter (8-dimensional).** State is `[cx, cy, aspect, height]` plus a velocity for each. Aspect ratio and height are tracked instead of width and height because aspect stays roughly constant as a person walks toward or away from the camera, separating shape from scale. Process and measurement noise scale with box height, since a distant person's pixel uncertainty is genuinely smaller.

An earlier version estimated velocity by differencing consecutive positions. On a synthetic constant-velocity target it drifted — final position 201.7 against a true 195, with mean error 5.98 versus 6.39 for the raw noisy measurements, barely better than not filtering at all. Making velocity part of the state with its own uncertainty fixed it: final position 195.2, mean error **4.15**.

**Hungarian assignment.** Detections are matched to predicted track boxes by minimising total cost rather than greedily taking the cheapest pair. Greedy matching fails because a locally cheap pair can strand another track with a terrible one — and a bad assignment is an ID switch.

**Appearance descriptor.** An HSV colour histogram per detection, with a rolling gallery of the 30 most recent views per track and matching against the closest of them. Measured on the corridor clip, mean distance between views of the same person three frames apart was **0.08**, versus **0.78** between different people — a clear separation.

### Synthetic results

| Scenario | Geometry only | Geometry + appearance |
|---|---|---|
| Clean crossing | 0 switches | 0 switches |
| Crossing + full occlusion | 0 switches | 0 switches |
| Occlusion + direction reversal | **10 switches** (10/10 runs) | **0 switches** (10/10 runs) |

The first two rows are the more interesting finding: **a properly specified Kalman filter handles clean crossings and even total occlusion on its own**, because the velocity model correctly predicts where each person re-emerges. Appearance contributes nothing there. It only earns its place when motion becomes unpredictable — in the third scenario, two people reverse direction while hidden, the prediction points at the wrong person, and geometry confidently mismatches every time.

### Results on real footage

200 frames per clip, identical detections in both conditions:

| Clip | Unique IDs — geometry | — with appearance | Reduction | fps |
|---|---|---|---|---|
| Corridor (15.5 detections/frame) | 37 | **28** | 24.3% | 8.7 → 7.9 |
| Plaza (38.4 detections/frame) | 85 | **50** | 41.2% | 6.6 → 4.9 |

Appearance helps more on the denser clip (41% vs 24%) — more people means more crossings and more ambiguous geometry. The throughput cost scales the same way and for the same reason: one histogram per detection per frame.

**Caveat on the metric.** Unique-ID count conflates genuine identity switches with normal churn from people entering and leaving frame. It indicates reduced fragmentation, not a switch count, and is not comparable to MOTA or IDF1. Two clips, 200 frames, single run each.

---

## Architecture

```
stampede-detection/
│
├── config.py              # All settings + per-camera profiles
├── analytics.py           # FrameAnalyzer (per-camera state) + CameraWorker (threading)
├── main.py                # Orchestration + rendering
├── alerts.py              # Alert dispatch: log, sound, terminal
│
├── kalman_box.py          # 8-dimensional Kalman filter for bounding boxes
├── matching.py            # IoU and Hungarian assignment
├── appearance.py          # HSV histogram descriptor + feature gallery
├── tracker.py             # Track lifecycle and combined-cost association
│
├── csrnet_model.py        # CSRNet architecture
├── csrnet_estimator.py    # CSRNet inference wrapper
├── density_estimator.py   # Earlier texture-based estimator (superseded, retained)
│
├── experiment_csrnet.py   # Reproduces Experiment 1
└── measure_tracking.py    # Reproduces Experiment 2
```

**Design notes.** Each camera gets its own `FrameAnalyzer` and its own `Tracker`, so density history, forecast state, and track IDs stay independent between zones. Each `CameraWorker` also loads its own YOLO model — sharing one across threads corrupts initialization, a failure encountered and diagnosed during development.

Camera-specific values live entirely in `config.py`; adapting to a new camera means adding a profile entry, with no logic changes.

---

## Forecasting

Density history is kept in a fixed-length buffer. The trend is measured by comparing the mean of the older half against the newer half, giving a rate of change per second. Time-to-critical is the remaining gap to the Fruin threshold divided by that rate.

Two safeguards keep it honest. **Horizon cap:** predictions beyond 120 seconds are suppressed, because as the rate approaches zero the extrapolation becomes numerically unstable — "critical in 400s" is not meaningful precision. **Temporal smoothing:** the displayed ETA is averaged over recent forecasts to prevent flicker.

---

## Fruin Level of Service

| Density (persons/m²) | Level | Interpretation |
|---|---|---|
| < 0.43 | A/B | Free movement |
| 0.43 – 1.08 | C/D | Restricted movement |
| > 1.08 | E/F | Dangerous density |

Fruin, J.J. (1971), *Pedestrian Planning and Design*.

---

## Setup

```bash
pip install ultralytics opencv-python torch torchvision scipy numpy
```

CSRNet weights (MIT licensed, ShanghaiTech Part B, 65 MB):
→ https://huggingface.co/rootstrap-org/crowd-counting — download `weights.pth` into the project root.

```bash
python main.py                # dashboard (Q to quit)
python experiment_csrnet.py   # reproduce Experiment 1
python measure_tracking.py    # reproduce Experiment 2
```

---

## Known limitations

**CPU throughput.** 4.9–8.7 fps single-camera depending on crowd size; lower with multiple cameras and CSRNet active. Real-time deployment needs GPU acceleration.

**Appearance descriptor is colour-based.** An HSV histogram cannot distinguish people wearing similar colours — a real constraint in the corridor footage, where several people wear white shirts. A re-identification CNN would learn features beyond colour.

**Association gating is simpler than DeepSORT's.** Real DeepSORT uses Mahalanobis gating, where the Kalman covariance automatically widens the gate as a track's uncertainty grows. This implementation uses a fixed appearance threshold with a geometry penalty, which works but is less principled.

**No depth information.** Density depends on a manually estimated floor area. A single camera cannot recover real-world scale from pixels.

**Detection ceiling in dense crowds.** Even at 1280px, YOLOv8n found about 83% of ground truth on the densest footage.

**Small evaluation sets.** Fifteen hand-annotated frames for Experiment 1; two clips of 200 frames for Experiment 2.

---

## Future work

- [ ] Homography calibration to remove manual area estimation
- [ ] Mahalanobis gating in place of the fixed appearance threshold
- [ ] Re-identification CNN descriptor instead of colour histograms
- [ ] YOLOv8x fine-tuned on CrowdHuman (GPU required)
- [ ] Larger annotated evaluation set, and a proper MOTA/IDF1 tracking benchmark
- [ ] Flow-direction and chokepoint analysis; live camera streams

---

## References

1. Fruin, J.J. (1971). *Pedestrian Planning and Design.* Metropolitan Association of Urban Designers and Environmental Planners.
2. Li, Y., Zhang, X., & Chen, D. (2018). CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. *CVPR 2018*.
3. Zhang, Y., et al. (2016). Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. *CVPR 2016*.
4. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP 2017*.
5. Bewley, A., et al. (2016). Simple Online and Realtime Tracking. *ICIP 2016*.
6. Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*.

---

## Author

**Ashvin Jaison Olickal**
B.Tech Computer Science · VIT-AP University

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ashvinj2004)

---

*Every component was built and evaluated incrementally rather than assembled. Where a design decision could be settled by measurement, it was — including the two cases where measurement contradicted the expected answer.*
