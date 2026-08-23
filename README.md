# Crowd Density Early-Warning System

**Real-time crowd monitoring with density forecasting and threshold-based alerting, built on YOLOv8 detection and CSRNet density estimation with an empirically-derived fusion policy.**

---

## What this does

Crowd crush incidents rarely happen without warning — density builds before it becomes dangerous. This system monitors camera feeds, converts person counts into real-world density using an academic pedestrian-safety standard, forecasts when density will cross a danger threshold, and raises alerts before that point is reached.

Concretely, per camera it:

1. Detects people using YOLOv8 with persistent tracking IDs
2. Converts count into density (people/m²) using a per-camera area profile
3. Classifies risk using **Fruin Level of Service** thresholds
4. Forecasts time-to-critical by extrapolating the smoothed density trend
5. Fires escalation alerts — on-screen banner, audible tone, and timestamped log
6. Selectively invokes **CSRNet** in high-density conditions where detection degrades

Multiple cameras run concurrently in independent threads and render to a tiled dashboard.

---

## Honest scope

This is a **density early-warning system**, not a stampede predictor. It forecasts when crowd density will cross established safety thresholds. It does not predict panic, crowd turbulence, or crush events — those depend on factors (behavioural, structural, psychological) this system does not measure. The naming here is deliberate: the system does what it claims and nothing more.

---

## Experimental findings

The most substantive part of this project is not the pipeline — it's the evaluation that determined how the pipeline should behave.

### Method

Five frames were sampled at even intervals from three clips representing different camera geometries. Each frame was manually annotated by hand-counting visible people. YOLOv8n (at 1280px input) and ShanghaiTech-pretrained CSRNet were then evaluated against those counts.

| Clip | Geometry | Typical count |
|---|---|---|
| Plaza | Elevated, open floor | ~35 |
| Corridor | Ground-level, chokepoint | ~16 |
| Aerial | High overhead, crossing | ~80 |

### Results

| Clip | MAE — YOLO | MAE — CSRNet | Detection rate — YOLO | Detection rate — CSRNet |
|---|---|---|---|---|
| Plaza | **1.3** | 22.1 | 98.5% | 41.2% |
| Corridor | **1.2** | 3.0 | 99.6% | 120.9% |
| Aerial | 14.3 | **9.9** | 82.6% | 87.1% |

### Interpretation

**YOLO substantially outperforms CSRNet at low-to-moderate density.** On the plaza clip YOLO achieved 98.5% of ground truth (MAE 1.3), while CSRNet detected only 41% of people present.

**CSRNet's failure is attributable to domain shift.** These weights were trained on ShanghaiTech Part B — street-level photography of moderate crowds. The plaza footage is a high overhead view of a bright, low-texture stone floor, visually distant from that training distribution.

**The ranking reverses at high density.** On the aerial clip, YOLO's detection rate fell to 82.6% as occlusion increased, while CSRNet reached 87.1% with lower absolute error. This crossover is the empirical basis for the fusion policy below.

**Input resolution dominated model choice.** Raising YOLO's input from 640px to 1280px increased detections on the aerial clip from 13 to 52 — a 4× improvement with no model change. For small-object detection in dense scenes, resolution mattered more than model capacity.

### Limitations of this evaluation

Five frames per clip is a small sample — sufficient to observe a strong effect, insufficient to locate the crossover threshold precisely. Manual annotation carries its own error, particularly on the aerial frames where exact counts were ambiguous and recorded as ranges. Results are specific to these three clips and should not be treated as general benchmarks.

---

## Fusion policy

Derived directly from the results above:

```
if yolo_count >= 50:          →  activate CSRNet (hysteresis: deactivate below 40)
    fused = max(yolo, csrnet) →  both undercount at high density; take the higher
else:                         →  trust YOLO alone (98%+ accurate in this regime)
```

Two implementation details worth noting:

**Hysteresis.** Separate activation (50) and deactivation (40) thresholds prevent chattering when counts oscillate around a single cutoff — the same principle as a thermostat's dead band.

**Interval execution.** CSRNet runs every 20th frame when active, with the result cached between runs. CSRNet inference costs seconds per frame on CPU; running it per-frame would make the system unusable. Crowd density does not change meaningfully at sub-second scales, so the accuracy cost is negligible.

---

## Forecasting

Density history is retained in a fixed-length buffer. The trend is measured by comparing the mean of the older half against the newer half of that window, giving a rate of change per second. Time-to-critical is then the remaining gap to the Fruin threshold divided by that rate.

Two safeguards keep the forecast honest:

**Horizon cap.** Predictions beyond 120 seconds are suppressed. When the rate of change approaches zero, the extrapolated ETA becomes numerically unstable — small fluctuations in a near-zero denominator produce wild swings. A forecast of "critical in 400s" is not meaningful precision; the system reports "not meaningfully rising" instead.

**Temporal smoothing.** The displayed ETA is averaged over recent forecasts to prevent frame-to-frame flicker.

---

## Fruin Level of Service

Risk thresholds follow Fruin's pedestrian density standard rather than arbitrary counts:

| Density (persons/m²) | Level | Interpretation |
|---|---|---|
| < 0.43 | A/B | Free movement |
| 0.43 – 1.08 | C/D | Restricted movement |
| > 1.08 | E/F | Dangerous density |

Source: Fruin, J.J. (1971), *Pedestrian Planning and Design*.

---

## Architecture

```
stampede-detection/
│
├── config.py              # All settings + per-camera profiles
├── analytics.py           # FrameAnalyzer (per-camera state) + CameraWorker (threading)
├── csrnet_model.py        # CSRNet architecture definition
├── csrnet_estimator.py    # CSRNet inference wrapper
├── alerts.py              # Alert dispatch: log, sound, terminal
├── main.py                # Entry point: orchestration + rendering
│
├── experiment_csrnet.py   # Evaluation script used to produce the findings above
└── weights.pth            # CSRNet weights (downloaded, see Setup)
```

**Design notes.** Each camera receives its own `FrameAnalyzer` instance, so density history, forecast state, and risk transitions remain independent between zones. Each `CameraWorker` also loads its own YOLO model — sharing one model across threads corrupts both initialization and tracker state, since `.track(persist=True)` stores tracking state on the model object.

Camera-specific values live entirely in `config.py`. Adapting to a new camera means adding a profile entry; no logic changes are required.

---

## Setup

```bash
pip install ultralytics opencv-python torch torchvision lap numpy
```

CSRNet weights (MIT licensed, ShanghaiTech Part B):
→ https://huggingface.co/rootstrap-org/crowd-counting

Download `weights.pth` into the project root.

```bash
python main.py          # run the dashboard (Q to quit)
python experiment_csrnet.py   # reproduce the evaluation
```

---

## Configuring a camera

```python
CAMERA_PROFILES = {
    0: {
        "name": "Plaza Cam",
        "source": "crowd.mp4",   # file path, or an integer for a live device
        "area_m2": 700.0,        # visible ground area
        "coverage": 0.85,        # fraction actually standable (excludes fixtures)
    },
}
```

**Area estimation is currently manual.** A single camera cannot recover real-world scale from pixels alone; this is a fundamental limitation, not an implementation gap. Homography calibration (mapping four known ground points to real coordinates) would resolve it and is listed under future work.

---

## Known limitations

**CPU performance.** With two cameras at 1280px input plus periodic CSRNet inference, throughput is well below real-time. Practical deployment requires GPU acceleration.

**No depth information.** Density depends on a manually estimated area. Ground-level cameras with significant depth variation are particularly affected — texture-based estimation was found to overestimate substantially in that geometry.

**ID switching.** Tracking uses position-based matching only. When people cross paths, identities occasionally swap. Appearance-based tracking (DeepSORT) would reduce this.

**Detection ceiling in dense crowds.** Even at 1280px input, YOLOv8n captured ~83% of ground truth on the densest footage. A larger model fine-tuned on CrowdHuman would likely improve this but requires GPU resources.

**Small evaluation sample.** Fifteen manually annotated frames total.

---

## Future work

- [ ] Homography calibration to remove manual area estimation
- [ ] DeepSORT for appearance-based tracking, reducing ID switches
- [ ] YOLOv8x fine-tuned on CrowdHuman (GPU required)
- [ ] Expanded annotated evaluation set to locate the fusion crossover precisely
- [ ] Flow-direction and chokepoint analysis
- [ ] Live camera and network stream support

---

## References

1. Fruin, J.J. (1971). *Pedestrian Planning and Design.* Metropolitan Association of Urban Designers and Environmental Planners.
2. Li, Y., Zhang, X., & Chen, D. (2018). CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes. *CVPR 2018*.
3. Zhang, Y., Zhou, D., Chen, S., Gao, S., & Ma, Y. (2016). Single-Image Crowd Counting via Multi-Column Convolutional Neural Network. *CVPR 2016*.
4. Wojke, N., Bewley, A., & Paulus, D. (2017). Simple Online and Realtime Tracking with a Deep Association Metric. *ICIP 2017*.

---

## Author

**Ashvin Jaison Olickal**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ashvinj2004)

---
