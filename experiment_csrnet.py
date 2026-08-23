"""
Standalone experiment: CSRNet vs YOLO across three camera geometries.
Saves sample frames so they can be inspected and manually counted.
"""

import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from csrnet_model import CSRNet

# ── Setup ──
csrnet = CSRNet()
checkpoint = torch.load("weights.pth", map_location="cpu")
state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
csrnet.load_state_dict(state)
csrnet.eval()

yolo = YOLO("yolov8n.pt")

# CSRNet expects ImageNet-normalised input — these exact numbers matter,
# they're the mean/std of the ImageNet dataset the VGG frontend was trained on.
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def csrnet_count(frame):
    """Estimate crowd count from a frame using CSRNet."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0)      # add batch dimension
    with torch.no_grad():                       # no gradients needed for inference
        density_map = csrnet(tensor)
    return float(density_map.sum().item()), density_map


def yolo_count(frame, imgsz=1280):
    results = yolo.predict(frame, classes=[0], conf=0.3, imgsz=imgsz, verbose=False)
    return len(results[0].boxes.xyxy)


def run_clip(path, label, n_samples=5):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"[{label}] could not read video")
        return

    # Pick evenly spaced frames across the clip
    indices = np.linspace(0, total - 1, n_samples, dtype=int)

    print(f"\n{'='*60}")
    print(f"  {label}   ({total} frames)")
    print(f"{'='*60}")
    print(f"{'Frame':>8} | {'YOLO':>6} | {'CSRNet':>8}")
    print("-" * 30)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        y = yolo_count(frame)
        c, _ = csrnet_count(frame)

        print(f"{idx:>8} | {y:>6} | {c:>8.1f}")

        # Save the frame so you can count it manually
        cv2.imwrite(f"sample_{label}_{idx}.jpg", frame)

    cap.release()


if __name__ == "__main__":
    run_clip("crowd.mp4",        "plaza")
    run_clip("vid2.mp4",         "corridor")
    run_clip("crowd_aerial.mp4", "aerial")
    print("\nSample frames saved as sample_<clip>_<frame>.jpg — open and count them.")