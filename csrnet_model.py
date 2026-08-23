"""
CSRNet — Congested Scene Recognition Network

Architecture:
  Frontend — first 10 conv layers of VGG16 (standard feature extraction)
  Backend  — dilated convolutions (wide receptive field, no resolution loss)
  Output   — a density map; summing all its pixels gives the crowd count.

The network never detects individuals. It learns what "crowdedness"
looks like as texture, which is why it works where detectors fail.
"""

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()

        # ── Frontend: VGG16's first 10 conv layers ──
        # 'M' means a MaxPool layer (halves the image size).
        # Three pools total -> output is 1/8 the input resolution.
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M',
                              256, 256, 256, 'M', 512, 512, 512]

        # ── Backend: dilated convolutions ──
        # dilation=2 means the 3x3 kernel samples pixels with a gap of 2,
        # covering a 5x5 area using only 3x3 worth of parameters —
        # a wider view WITHOUT shrinking the image further.
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Final 1x1 conv collapses 64 channels down to a single-channel density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            # Initialise frontend from ImageNet-pretrained VGG16
            vgg = models.vgg16(weights='IMAGENET1K_V1')
            frontend_dict = dict(self.frontend.state_dict())
            vgg_items = list(vgg.features.state_dict().items())
            for i, key in enumerate(frontend_dict.keys()):
                frontend_dict[key] = vgg_items[i][1]
            self.frontend.load_state_dict(frontend_dict)

    def forward(self, x):
        x = self.frontend(x)       # extract visual features
        x = self.backend(x)        # widen context via dilated convs
        x = self.output_layer(x)   # collapse to one density channel
        return x


def make_layers(cfg, in_channels=3, dilation=False):
    """Build a conv stack from a config list. 'M' = maxpool."""
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(in_channels, v, kernel_size=3,
                             padding=d_rate, dilation=d_rate)
            layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)