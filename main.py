"""
Crowd Early-Warning System — main entry point.
Coordinates detection, forecasting, display, and alerting.

Run:  python main.py
Quit: press Q in the video window
"""

import numpy as np
import cv2
from ultralytics import YOLO

import config
from analytics import FrameAnalyzer, CameraWorker


def draw_overlay(frame, data):
    """Draw boxes, IDs, status panel, forecast, and critical banner onto a frame."""
    # Person boxes + track IDs
    for box, tid in zip(data["boxes"], data["track_ids"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if tid is not None:
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Status panel background
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (20, 20, 20), -1)

    # Stats line
    source_tag = " [CSRNet]" if data.get("used_csrnet") else ""
    cv2.putText(frame,
                f"{data['camera_name']}  |  People: {data['person_count']}{source_tag}  |  "
                f"Density: {data['density']:.3f}/m2",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Status line (coloured by risk)
    cv2.putText(frame, f"Status: {data['risk_label']}",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, data["colour"], 2)

    # Forecast line
    eta = data["eta"]
    if eta is None:
        forecast_text = "Forecast: stable / not rising"
    elif eta <= 0:
        forecast_text = "Forecast: CRITICAL NOW"
    else:
        forecast_text = f"Forecast: critical in ~{eta:.0f}s"
    cv2.putText(frame, forecast_text, (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Big red banner when critical
    if data["risk_label"] == "CRITICAL":
        cv2.rectangle(frame, (0, 120), (frame.shape[1], 170), (0, 0, 255), -1)
        cv2.putText(frame, "!!! CRITICAL DENSITY - ALERT SECURITY !!!",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


def build_dashboard(frames, cols=2, tile_size=(640, 360)):
    """Tile multiple camera frames into one grid image."""
    tiles = [cv2.resize(f, tile_size) for f in frames]

    # Pad with black tiles so the grid is always complete
    while len(tiles) % cols != 0:
        tiles.append(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))

    # Build each row, then stack the rows vertically
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    # Create and start one worker per camera
    workers = [CameraWorker(profile, config.MODEL_PATH)
               for profile in config.CAMERA_PROFILES.values()]
    for w in workers:
        w.start()

    names = ", ".join(w.profile["name"] for w in workers)
    print(f"[SYSTEM] Monitoring {len(workers)} zones: {names} — press Q to quit")

    try:
        while True:
            annotated_frames = []
            for w in workers:
                frame, data = w.get_latest()
                if frame is None:
                    # Worker hasn't produced anything yet — show a placeholder
                    placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Initializing...", (180, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    annotated_frames.append(placeholder)
                    continue
                annotated_frames.append(draw_overlay(frame, data))

            dashboard = build_dashboard(annotated_frames)
            cv2.imshow("Crowd Early-Warning System — Multi-Zone", dashboard)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()