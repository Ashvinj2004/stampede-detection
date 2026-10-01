"""
8-dimensional Kalman filter for bounding boxes — the DeepSORT formulation.

State: [cx, cy, a, h, vcx, vcy, va, vh]
  cx, cy  centre of the box
  a       aspect ratio (width / height)
  h       height
  v*      the velocity of each

Aspect ratio and height are tracked instead of width and height because
aspect stays roughly constant as a person walks toward or away from the
camera, while height changes with distance. That separates shape from scale.

Same predict/update cycle as the 1D filter — just more dimensions.
"""

import numpy as np


class KalmanBoxTracker:
    def __init__(self, bbox):
        dt = 1.0

        # F: constant-velocity model. Each position gains its velocity each step.
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt

        # H: we measure position/shape only, never velocity.
        self.H = np.eye(4, 8)

        z = self._to_z(bbox)
        self.x = np.r_[z, np.zeros(4)]      # start with zero velocity

        # Initial uncertainty, scaled by box height.
        # Velocity uncertainty starts much higher — we have no idea yet.
        h = z[3]
        std = [2 * 0.05 * h, 2 * 0.05 * h, 1e-2, 2 * 0.05 * h,
               10 * 0.00625 * h, 10 * 0.00625 * h, 1e-5, 10 * 0.00625 * h]
        self.P = np.diag(np.square(std))

    @staticmethod
    def _to_z(b):
        """[x1,y1,x2,y2] -> [cx, cy, aspect, height]"""
        x1, y1, x2, y2 = b
        w, h = x2 - x1, y2 - y1
        return np.array([x1 + w / 2, y1 + h / 2, w / max(h, 1e-6), h])

    @staticmethod
    def _to_box(state):
        """[cx, cy, aspect, height, ...] -> [x1,y1,x2,y2]"""
        cx, cy, a, h = state[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        """Move the state forward one frame. Uncertainty grows."""
        h = max(self.x[3], 1.0)
        std = [0.05 * h, 0.05 * h, 1e-2, 0.05 * h,
               0.00625 * h, 0.00625 * h, 1e-5, 0.00625 * h]
        Q = np.diag(np.square(std))

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        return self._to_box(self.x)

    def update(self, bbox):
        """Correct the state with a new detection. Uncertainty shrinks."""
        h = max(self.x[3], 1.0)
        std = [0.05 * h, 0.05 * h, 1e-1, 0.05 * h]
        R = np.diag(np.square(std))

        z = self._to_z(bbox)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + R           # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain (8x4)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self._to_box(self.x)


if __name__ == "__main__":
    # Simulated person walking right and away from the camera
    np.random.seed(1)
    truth, meas = [], []
    for t in range(25):
        cx, cy = 200 + 6 * t, 300
        h = 160 - 1.5 * t          # shrinking = walking away
        w = h * 0.4
        b = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        truth.append(b)
        meas.append(b + np.random.normal(0, 5, 4))   # noisy "YOLO" boxes

    kf = KalmanBoxTracker(meas[0])
    est = [kf._to_box(kf.x)]
    for z in meas[1:]:
        kf.predict()
        est.append(kf.update(z))

    def mae(seq):
        return np.mean([np.mean(np.abs(p - t)) for p, t in zip(seq, truth)])

    print(f"MAE raw measurements : {mae(meas):.2f} px")
    print(f"MAE Kalman estimate  : {mae(est):.2f} px")
    print(f"final velocity [vcx, vcy, va, vh] = {np.round(kf.x[4:], 2)}"
          f"   (true vcx=6.0, vh=-1.5)")