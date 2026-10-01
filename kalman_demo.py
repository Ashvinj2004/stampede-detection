"""
Proper 1D Kalman filter — position AND velocity as state.

The earlier version estimated velocity by differencing positions, which
amplified noise and drifted. Here velocity is part of the state vector
with its own uncertainty, so the filter can detect and correct a bad
velocity estimate. This is the structure DeepSORT uses (in 8 dimensions).
"""

import numpy as np

# ── Ground truth: walking right at a steady 5 px/frame ──
truth = [100 + 5 * t for t in range(20)]

# ── What YOLO "sees": truth plus noise ──
np.random.seed(0)
measurements = [p + np.random.normal(0, 8) for p in truth]

# ── State: [position, velocity] as a column vector ──
x = np.array([[100.0],
              [0.0]])

# F: motion model. new_pos = pos + vel, new_vel = vel
F = np.array([[1.0, 1.0],
              [0.0, 1.0]])

# H: measurement model. We observe position only, not velocity.
H = np.array([[1.0, 0.0]])

# P: uncertainty in our state. Starts high — we know nothing.
P = np.diag([100.0, 100.0])

# Q: process noise. How much can reality deviate from constant velocity?
q = 0.01
Q = q * np.array([[0.25, 0.5],
                  [0.5,  1.0]])

# R: measurement noise. YOLO's 8px std, squared.
R = np.array([[64.0]])

I = np.eye(2)

print(f"{'t':>3} {'truth':>8} {'measured':>9} {'est pos':>9} {'est vel':>8} "
      f"{'err':>7} {'P_pos':>7} {'P_vel':>7}")
print("-" * 68)
estimates = []


for t, z in enumerate(measurements):
    # ── PREDICT ──
    x = F @ x                    # move state forward
    P = F @ P @ F.T + Q          # uncertainty grows

    # ── UPDATE ──
    y = np.array([[z]]) - H @ x  # innovation: measurement minus prediction
    S = H @ P @ H.T + R          # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)   # Kalman gain (2x1 — affects BOTH states)
    x = x + K @ y                # correct the state
    P = (I - K @ H) @ P          # uncertainty shrinks

    pos, vel = float(x[0, 0]), float(x[1, 0])
    err = abs(pos - truth[t])

    print(f"{t:>3} {truth[t]:>8.1f} {z:>9.1f} {pos:>9.1f} {vel:>8.2f} "
          f"{err:>7.2f} {P[0,0]:>7.2f} {P[1,1]:>7.2f}")

    estimates.append(pos)

errors = [abs(e - t) for e, t in zip(estimates, truth)]
mae_meas = sum(abs(m - t) for m, t in zip(measurements, truth)) / len(truth)
mae_est  = sum(errors) / len(errors)

print()
print(f"Mean abs error — raw measurements: {mae_meas:.2f}")
print(f"Mean abs error — Kalman estimate:  {mae_est:.2f}")