"""Primeira CNN própria de estimativa de pose do trackingCorporal.

A rede recebe uma imagem RGB 256x256 e produz 17 heatmaps 64x64, um para cada
keypoint corporal. Ela não usa backbone pré-treinado nem detector de pose pronto.
"""

from __future__ import annotations

import torch
from torch import nn

from src.pose.keypoints import NUM_KEYPOINTS


class ConvBlock(nn.Sequential):
    """Dois blocos Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class PoseNet(nn.Module):
    """CNN encoder-decoder pequena para estimativa de 17 heatmaps corporais."""

    def __init__(self, keypoint_count: int = NUM_KEYPOINTS) -> None:
        super().__init__()

        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64)
        self.encoder3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(128, 256)

        # Entrada 256 -> pools: 128 -> 64 -> 32; upsample final -> 64.
        self.upsample = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.refine = ConvBlock(128, 64)
        self.head = nn.Conv2d(64, keypoint_count, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(image)
        x = self.pool(x)

        x = self.encoder2(x)
        x = self.pool(x)

        x = self.encoder3(x)
        x = self.pool(x)

        x = self.bottleneck(x)
        x = self.upsample(x)
        x = self.refine(x)

        return self.head(x)
