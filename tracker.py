"""
Multi-object tracker: Kalman prediction + Hungarian matching + appearance.

Each frame:
  1. every track predicts where it should now be
  2. predictions are matched to detections by combined geometry + appearance cost
  3. matched tracks are corrected; unmatched ones age out; new detections start tracks

Measured on synthetic scenarios: geometry alone handles clean crossings and even
full occlusion when motion stays predictable. It fails when people change
direction while hidden (10/10 runs produced an ID switch). Adding appearance
removed those entirely (0/10).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from Kalman_box import KalmanBoxTracker
from matching import iou
from appearance import FeatureGallery


class Track:
    """One tracked person: a Kalman filter, an ID, and a memory of appearances."""

    def __init__(self, track_id, bbox, feature, n_init=3, max_age=30):
        self.id = track_id
        self.kf = KalmanBoxTracker(bbox)
        self.gallery = FeatureGallery()
        if feature is not None:
            self.gallery.add(feature)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age

        # Tentative until matched n_init times. The Kalman filter needs a few
        # frames to estimate velocity, so a one-frame detection isn't yet
        # trustworthy enough to display.
        self.state = "tentative"

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, feature):
        self.kf.update(bbox)
        if feature is not None:
            self.gallery.add(feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == "tentative" and self.hits >= self.n_init:
            self.state = "confirmed"

    def mark_missed(self):
        if self.state == "tentative":
            self.state = "deleted"          # never established — discard
        elif self.time_since_update > self.max_age:
            self.state = "deleted"          # gone too long

    def box(self):
        return self.kf._to_box(self.kf.x)


class Tracker:
    def __init__(self, max_age=30, n_init=3,
                 appearance_weight=0.5, max_appearance_distance=0.4,
                 use_appearance=True):
        self.tracks = []
        self._next_id = 1
        self.max_age = max_age
        self.n_init = n_init
        self.w = appearance_weight
        self.max_app = max_appearance_distance
        self.use_appearance = use_appearance

    def _cost_matrix(self, tracks, detections, features):
        C = np.zeros((len(tracks), len(detections)))
        for i, t in enumerate(tracks):
            tb = t.box()
            for j, d in enumerate(detections):
                geo = 1 - iou(tb, d)

                if self.use_appearance and features is not None:
                    app = t.gallery.distance_to(features[j])
                    if app > self.max_app:
                        C[i, j] = 1e5          # appearance rules it out
                    elif geo >= 1.0:
                        # No overlap: the prediction drifted during occlusion.
                        # Appearance alone carries the match, at a penalty.
                        # Gating this out instead is what broke the first
                        # version — it rejected exactly the re-acquisitions
                        # appearance exists to make.
                        C[i, j] = self.w * app + (1 - self.w) * 0.95
                    else:
                        C[i, j] = self.w * app + (1 - self.w) * geo
                else:
                    C[i, j] = 1e5 if geo >= 1.0 else geo
        return C

    def update(self, detections, features=None):
        """
        Advance the tracker one frame.
        Returns [(track_id, box), ...] for confirmed tracks only.
        """
        for t in self.tracks:
            t.predict()

        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))

        if self.tracks and len(detections):
            C = self._cost_matrix(self.tracks, detections, features)
            rows, cols = linear_sum_assignment(C)
            matches = [(i, j) for i, j in zip(rows, cols) if C[i, j] < 1e4]
            matched_t = {m[0] for m in matches}
            matched_d = {m[1] for m in matches}
            unmatched_tracks = [i for i in range(len(self.tracks)) if i not in matched_t]
            unmatched_dets = [j for j in range(len(detections)) if j not in matched_d]

        for i, j in matches:
            self.tracks[i].update(detections[j],
                                  features[j] if features is not None else None)

        for i in unmatched_tracks:
            self.tracks[i].mark_missed()

        for j in unmatched_dets:
            self.tracks.append(Track(self._next_id, detections[j],
                                     features[j] if features is not None else None,
                                     self.n_init, self.max_age))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.state != "deleted"]
        
        # Only report tracks matched to a detection this frame. Unmatched tracks
        # stay alive internally so appearance can re-acquire them after occlusion,
        # but we don't draw a box where we have no current evidence of anyone.
        return [(t.id, t.box()) for t in self.tracks
                if t.state == "confirmed" and t.time_since_update == 0]