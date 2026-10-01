"""
FrameAnalyzer — wraps detection, density, and forecasting for ONE camera.
Each camera gets its own instance, so their histories stay independent.
"""

from collections import deque
import numpy as np
from ultralytics import YOLO
import config
import threading
import cv2

from tracker import Tracker
from appearance import extract_feature

from alerts import fire_alert
from csrnet_estimator import CSRNetEstimator

# Rank risk levels so we can detect escalation
RISK_RANK = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}

def classify_density(density):
    """Density number -> (risk label, BGR colour)."""
    if density >= config.FRUIN_WARNING_MAX:
        return "CRITICAL", (0, 0, 255)
    elif density >= config.FRUIN_NORMAL_MAX:
        return "WARNING", (0, 165, 255)
    else:
        return "NORMAL", (0, 255, 0)

class FrameAnalyzer:
    """Analyzes frames from a single camera and tracks its state over time."""

    def __init__(self, profile, model, fps=25):
        # Data that belongs to THIS camera (sealed inside via self)
        self.profile = profile
        self.model = model
        self.fps = fps
        self.usable_area = config.usable_area(profile)

        self.tracker = Tracker(
            max_age=config.TRACKER_MAX_AGE,
            n_init=config.TRACKER_N_INIT,
            appearance_weight=config.TRACKER_APPEARANCE_WEIGHT,
            max_appearance_distance=config.TRACKER_MAX_APP_DISTANCE,
            use_appearance=config.TRACKER_USE_APPEARANCE,
        )

        # Each camera's own private memory
        self.density_history = deque(maxlen=config.HISTORY_LENGTH)
        self.eta_history = deque(maxlen=config.ETA_SMOOTHING)
        self.previous_risk = "NORMAL"

        # CSRNet — loaded lazily, only if enabled
        self.csrnet = None
        if config.CSRNET_ENABLED:
            self.csrnet = CSRNetEstimator(config.CSRNET_WEIGHTS)
        self.frame_counter = 0
        self.last_csrnet_count = None
        self.csrnet_active = False

    def _forecast(self):
        """Estimate seconds until critical from this camera's density trend."""
        history = self.density_history
        if len(history) < 30:
            return None
        half = len(history) // 2
        older_avg = np.mean(list(history)[:half])
        newer_avg = np.mean(list(history)[half:])
        rate = (newer_avg - older_avg) / ((len(history) / 2) / self.fps)
        if rate <= 0:
            return None
        gap = config.FRUIN_WARNING_MAX - newer_avg
        if gap <= 0:
            return 0
        eta = gap / rate
        if eta > config.FORECAST_HORIZON:
            return None
        return eta

    def analyze(self, frame):
        """
        Run the full pipeline on one frame.
        Returns a dict of everything the display/alert layers need.
        """
        results = self.model.predict(
            frame, classes=[0], conf=config.CONF_THRESHOLD,
            imgsz=config.YOLO_IMGSZ, verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()
        person_count = len(boxes)

        # Appearance features for each detection, then run our own tracker
        features = [extract_feature(frame, b) for b in boxes]
        tracked = self.tracker.update(list(boxes), features)
            
        # ── Adaptive fusion ──
        # Below the trigger, YOLO was ~98% accurate in testing, so we trust it.
        # Above it, detection degrades and CSRNet becomes the better estimator.
        self.frame_counter += 1
        fused_count = person_count
        used_csrnet = False

        if self.csrnet is not None:
            # Hysteresis: separate on/off thresholds prevent chattering
            if not self.csrnet_active and person_count >= config.CSRNET_TRIGGER_ON:
                self.csrnet_active = True
            elif self.csrnet_active and person_count < config.CSRNET_TRIGGER_OFF:
                self.csrnet_active = False
                self.last_csrnet_count = None

            if self.csrnet_active:
                if self.frame_counter % config.CSRNET_INTERVAL == 0 or self.last_csrnet_count is None:
                    self.last_csrnet_count = self.csrnet.count(frame)
                if self.last_csrnet_count is not None:
                    fused_count = max(person_count, int(self.last_csrnet_count))
                    used_csrnet = True

        density = fused_count / self.usable_area
        risk_label, colour = classify_density(density)

        # Update this camera's memory + forecast
        self.density_history.append(density)
        raw_eta = self._forecast()
        if raw_eta is not None and raw_eta > 0:
            self.eta_history.append(raw_eta)
        eta = np.mean(self.eta_history) if len(self.eta_history) > 0 else raw_eta

        # Detect escalation
        escalated = RISK_RANK[risk_label] > RISK_RANK[self.previous_risk]
        self.previous_risk = risk_label

        return {
            "boxes": boxes,
            "tracked": tracked,
            "person_count": fused_count,
            "density": density,
            "risk_label": risk_label,
            "colour": colour,
            "eta": eta,
            "escalated": escalated,
            "camera_name": self.profile["name"],
            "used_csrnet": used_csrnet,
            "YOLO_count": person_count,
        }

class CameraWorker:
    """
    Runs one camera in its own thread.
    Continuously reads frames, analyzes them, and stores the latest result
    so the main loop can grab it without waiting.
    """

    def __init__(self, profile, model_path):
        self.profile = profile
        self.cap = cv2.VideoCapture(profile["source"])
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        model = YOLO(model_path)
        self.analyzer = FrameAnalyzer(profile, model, fps)

        # Shared state — written by the worker thread, read by the main thread
        self.latest_frame = None
        self.latest_data = None
        self._lock = threading.Lock()      # the "dnd" sign
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        self.cap.release()

    def _loop(self):
        """The loop runs continuously in its own thread."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop the video
                continue

            data = self.analyzer.analyze(frame)

            if data["escalated"]:
                fire_alert(data["risk_label"], data["density"], data["camera_name"])

            # Store the result — lock so the main thread can't read mid-write
            with self._lock:
                self.latest_frame = frame
                self.latest_data = data

    def get_latest(self):
        """Safely fetch the most recent frame + analysis."""
        with self._lock:
            if self.latest_frame is None:
                return None, None
            return self.latest_frame.copy(), self.latest_data