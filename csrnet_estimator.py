"""
CSRNet wrapper — loads the pretrained model and estimates crowd count.

Used selectively: experiments showed CSRNet underperforms YOLO at
low/moderate density (domain shift from its ShanghaiTech training set)
but outperforms it in dense scenes where detection degrades.
"""

import cv2
import torch
from torchvision import transforms

from csrnet_model import CSRNet


class CSRNetEstimator:
    def __init__(self, weights_path):
        self.model = CSRNet()
        checkpoint = torch.load(weights_path, map_location="cpu")
        state = (checkpoint["state_dict"]
                 if isinstance(checkpoint, dict) and "state_dict" in checkpoint
                 else checkpoint)
        self.model.load_state_dict(state)
        self.model.eval()

        # ImageNet normalisation — required because the VGG frontend
        # was pretrained on ImageNet-scaled inputs.
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def count(self, frame):
        """Return estimated crowd count by summing the predicted density map."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(rgb).unsqueeze(0)
        with torch.no_grad():
            density_map = self.model(tensor)
        return float(density_map.sum().item())