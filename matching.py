"""
Detection-to-track assignment.

Each frame: build a cost matrix between predicted track boxes and new
detections, then solve it globally with the Hungarian algorithm.

Greedy matching fails because a locally cheap pair can strand another
track with a terrible one — and a bad assignment is an ID switch.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    """Intersection over union of two [x1,y1,x2,y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = ((a[2]-a[0]) * (a[3]-a[1])
             + (b[2]-b[0]) * (b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def associate(track_boxes, detections, iou_threshold=0.3):
    """
    Match predicted track boxes to detections.
    Returns (matches, unmatched_tracks, unmatched_detections).
    """
    if len(track_boxes) == 0 or len(detections) == 0:
        return [], list(range(len(track_boxes))), list(range(len(detections)))

    # Cost = 1 - IoU. Perfect overlap costs 0.
    cost = np.zeros((len(track_boxes), len(detections)))
    for i, t in enumerate(track_boxes):
        for j, d in enumerate(detections):
            cost[i, j] = 1 - iou(t, d)

    rows, cols = linear_sum_assignment(cost)

    matches, used_t, used_d = [], set(), set()
    for i, j in zip(rows, cols):
        # Reject pairs that are technically optimal but still bad matches
        if cost[i, j] > 1 - iou_threshold:
            continue
        matches.append((i, j))
        used_t.add(i)
        used_d.add(j)

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in used_t]
    unmatched_dets = [j for j in range(len(detections)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets

if __name__ == "__main__":
    # Two tracks, two detections. Track A's box has drifted slightly.
    tracks = [np.array([100, 100, 150, 200]),
              np.array([300, 100, 350, 200])]
    dets   = [np.array([305, 102, 355, 202]),   # belongs to track B
              np.array([ 98, 101, 148, 201])]   # belongs to track A

    m, ut, ud = associate(tracks, dets)
    print("matches (track_idx, det_idx):", m)
    print("unmatched tracks:", ut)
    print("unmatched detections:", ud)

    # Greedy vs Hungarian on the case from before
    from scipy.optimize import linear_sum_assignment
    C = np.array([[0.20, 0.30],
                  [0.25, 0.90]])
    r, c = linear_sum_assignment(C)
    print(f"\nHungarian picks {[(int(i), int(j)) for i, j in zip(r, c)]} "
          f"at total cost {C[r, c].sum():.2f}")
    print("Greedy would pick [(0, 0), (1, 1)] at total cost 1.10")