"""Compact U-Net for AIR2 semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class AIR2UNet(nn.Module):
    """Small U-Net that predicts per-pixel AIR2 object classes."""

    def __init__(self, num_classes: int = 7, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = Down(base_channels, base_channels * 2)
        self.encoder3 = Down(base_channels * 2, base_channels * 4)
        self.encoder4 = Down(base_channels * 4, base_channels * 8)

        self.decoder3 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = Up(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        x = self.encoder4(skip3)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.head(x)


def build_model(num_classes: int = 7, in_channels: int = 3, base_channels: int = 32) -> AIR2UNet:
    return AIR2UNet(num_classes=num_classes, in_channels=in_channels, base_channels=base_channels)
