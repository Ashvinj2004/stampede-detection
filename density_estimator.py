"""
Lightweight crowd density estimator.

Estimates crowd size from visual texture rather than detecting individuals.
Works where YOLO fails: dense, occluded crowds.

more people -> more edges + more motion per unit area.
Calibrates itself against YOLO during sparse periods, when YOLO is reliable.
"""

import cv2
import numpy as np
from collections import deque


class TextureDensityEstimator:
    """Estimates crowd count from edge + motion texture."""

    def __init__(self, calibration_samples=100):
        self.prev_gray = None

        # Stores (texture_score, yolo_count) pairs collected when YOLO is trustworthy
        self.calibration_data = deque(maxlen=calibration_samples)
        self.scale_factor = None      # learned

    def _texture_score(self, frame):
        """Measure visual busyness: edge density + motion density."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Edge density ──
        # Canny finds sharp brightness changes. Result is a black image
        # with white pixels marking edges.
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size   # fraction of pixels that are edges

        # ── Motion density ──
        # Subtract previous frame: static things cancel out, movers light up.
        motion_ratio = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_ratio = np.count_nonzero(motion_mask) / motion_mask.size

        self.prev_gray = gray

        # Weighted blend. Edges carry more signal; motion confirms it's people.
        return (edge_ratio * 0.7) + (motion_ratio * 0.3)

    def update_calibration(self, frame, yolo_count, sparse_threshold=200):
        """
        Learn the texture->count relationship while the crowd is sparse
        (where YOLO is reliable). Returns the current texture score.
        """
        score = self._texture_score(frame)

        # Only learn from sparse scenes — YOLO is trustworthy there
        if 0 < yolo_count <= sparse_threshold and score > 0:
            self.calibration_data.append((score, yolo_count))

        # Recompute the scale factor from everything learned so far
        if len(self.calibration_data) >= 10:
            ratios = [count / s for s, count in self.calibration_data]
            self.scale_factor = float(np.median(ratios))   # median resists outliers

        return score

    def estimate(self, frame, yolo_count):
        """
        Estimate crowd count from texture.
        Returns (estimated_count, is_calibrated).
        """
        score = self.update_calibration(frame, yolo_count)

        if self.scale_factor is None:
            return None, False        # not calibrated yet — be honest about it

        return int(score * self.scale_factor), True
        