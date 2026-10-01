"""
A/B measurement: tracker performance with and without appearance.

Counts total unique IDs issued over a clip (a proxy for fragmentation —
fewer IDs for the same scene means better identity continuity) and
measures throughput. Both conditions see identical detections.
"""

import time
import cv2
from ultralytics import YOLO

import config
from tracker import Tracker
from appearance import extract_feature


def run(video_path, use_appearance, max_frames=200):
    model = YOLO(config.MODEL_PATH)
    cap = cv2.VideoCapture(video_path)

    trk = Tracker(
        max_age=config.TRACKER_MAX_AGE,
        n_init=config.TRACKER_N_INIT,
        appearance_weight=config.TRACKER_APPEARANCE_WEIGHT,
        max_appearance_distance=config.TRACKER_MAX_APP_DISTANCE,
        use_appearance=use_appearance,
    )

    seen_ids = set()
    frames = 0
    detections_total = 0
    t0 = time.time()

    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1

        results = model.predict(frame, classes=[0], conf=config.CONF_THRESHOLD,
                                imgsz=config.YOLO_IMGSZ, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        detections_total += len(boxes)

        features = [extract_feature(frame, b) for b in boxes] if use_appearance else None
        for tid, _ in trk.update(list(boxes), features):
            seen_ids.add(tid)

    elapsed = time.time() - t0
    cap.release()

    return {
        "frames": frames,
        "unique_ids": len(seen_ids),
        "mean_detections": detections_total / max(frames, 1),
        "fps": frames / elapsed if elapsed > 0 else 0,
    }


if __name__ == "__main__":
    for clip in ["vid2.mp4", "crowd.mp4"]:
        print(f"\n=== {clip} ===")
        print(f"{'condition':<22} {'frames':>7} {'uniq IDs':>9} "
              f"{'mean dets':>10} {'fps':>7}")
        for label, ua in [("geometry only", False), ("geometry + appearance", True)]:
            r = run(clip, ua)
            print(f"{label:<22} {r['frames']:>7} {r['unique_ids']:>9} "
                  f"{r['mean_detections']:>10.1f} {r['fps']:>7.2f}")