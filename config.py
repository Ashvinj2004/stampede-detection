"""
Configuration for the crowd early-warning system.
change cameras/thresholds.
"""

# Detection settings
MODEL_PATH     = "yolov8n.pt"    # swap to yolov8x.pt when you have a GPU
CONF_THRESHOLD = 0.3             # confidence
DEVICE         = "cpu"           # change to "cuda" with a GPU

# Fruin thresholds (people per m²)
FRUIN_NORMAL_MAX  = 0.43
FRUIN_WARNING_MAX = 1.08

# Forecast settings
HISTORY_LENGTH   = 75            # frames of density memory (~3s at 25fps)
ETA_SMOOTHING    = 15            # frames to smooth the forecast number
FORECAST_HORIZON = 120           # don't predict further out than this (seconds)


# Each camera gets its own entry. Add as many as you like.
# swapping a profile adapts everything.
CAMERA_PROFILES = {
    0: {
        "name": "Plaza Cam",
        "source": "crowd.mp4",
        "area_m2": 700.0,
        "coverage": 0.85,
    },
    1: {
        "name": "Aerial Cam",
        "source": "crowd_aerial.mp4",     # same footage, different zone config
        "area_m2": 300.0,           # much smaller space → much higher density
        "coverage": 0.9,
    },
    # To add another camera, uncomment and fill in:
    # 2: {
    #     "name": "Entrance Cam",
    #     "source": "entrance.mp4",   # or a number like 0 for a webcam
    #     "area_m2": 40.0,
    #     "coverage": 0.9,
    # },
}

def usable_area(profile):
    """Compute the standable area for a camera profile."""
    return profile["area_m2"] * profile["coverage"]

# ── CSRNet fusion settings (thresholds derived from manual-annotation experiment) ──
CSRNET_ENABLED     = True
CSRNET_WEIGHTS     = "weights.pth"
CSRNET_TRIGGER_ON  = 50      # YOLO count above which CSRNet is invoked
CSRNET_TRIGGER_OFF = 40      # deactivate only below this (hysteresis)
CSRNET_INTERVAL    = 20       # run CSRNet every Nth frame when triggered (CPU cost)
YOLO_IMGSZ         = 1280    # higher res: quadrupled detections vs default 640

# ── Tracker settings ──
TRACKER_USE_APPEARANCE   = True   # flip to False to A/B test against geometry alone
TRACKER_MAX_AGE          = 15     # frames a track survives unmatched (~1.2s at 25fps)
TRACKER_N_INIT           = 3      # matches needed before a track is shown
TRACKER_APPEARANCE_WEIGHT = 0.5   # 0 = geometry only, 1 = appearance only
TRACKER_MAX_APP_DISTANCE  = 0.4   # above this, appearance rules a pairing out