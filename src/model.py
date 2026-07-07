"""SRCNN - the classic three-layer super-resolution CNN (Dong et al., 2014).

It takes a bicubic-upsampled low-resolution image at the target size and learns
a residual-free mapping to the sharp high-resolution image: patch extraction
(9x9) -> non-linear mapping (5x5) -> reconstruction (5x5).
"""

import torch.nn as nn


class Sr_Cnn(nn.Module):
    def __init__(self, num_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.features(x)


class Sr_Cnn_Mf(nn.Module):
    """Multi-frame SRCNN with early fusion.

    Several bicubic-upsampled LR shots of the same face are stacked on the
    channel axis (num_channels * num_frames) and the widened 9x9 first layer
    fuses them; the 5x5 mapping and 5x5 reconstruction are unchanged from the
    single-image SRCNN. Only the input width differs, so the two models stay
    directly comparable.
    """

    def __init__(self, num_channels=3, num_frames=5):
        super().__init__()
        self.num_frames = num_frames
        self.features = nn.Sequential(
            nn.Conv2d(num_channels * num_frames, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.features(x)
