"""Segmentation models for AIR2.

Two options:
  AIR2UNet        — compact custom U-Net, trained from scratch (original)
  AIR2ResNetSeg   — ResNet-18 pretrained backbone + lightweight decoder (recommended)
"""

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

    def __init__(self, num_classes: int = 9, in_channels: int = 3, base_channels: int = 32):
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


def build_model(num_classes: int = 9, in_channels: int = 3, base_channels: int = 32) -> AIR2UNet:
    return AIR2UNet(num_classes=num_classes, in_channels=in_channels, base_channels=base_channels)


# ---------------------------------------------------------------------------
# ResNet-18 segmentation model
# ---------------------------------------------------------------------------

class _UpNoSkip(nn.Module):
    """Upsample without a skip connection (used for the final decoder step)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class AIR2ResNetSeg(nn.Module):
    """ResNet-18 encoder + lightweight decoder for AIR2 semantic segmentation.

    Pretrained ImageNet weights give robust features from the start.
    The encoder (enc0..enc4) is reused frozen in the BC/PPO policy.
    """

    ENCODER_DIM = 512  # ResNet-18 layer4 output channels

    def __init__(self, num_classes: int = 9, pretrained: bool = True):
        super().__init__()
        import torchvision.models as tvm
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = tvm.resnet18(weights=weights)

        # Encoder — ResNet-18 stages, kept as named attributes for easy extraction
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64, H/2
        self.pool = backbone.maxpool                                              # 64, H/4
        self.enc1 = backbone.layer1   # 64,  H/4
        self.enc2 = backbone.layer2   # 128, H/8
        self.enc3 = backbone.layer3   # 256, H/16
        self.enc4 = backbone.layer4   # 512, H/32

        # Decoder with skip connections
        self.dec4 = Up(512, 256, 256)
        self.dec3 = Up(256, 128, 128)
        self.dec2 = Up(128, 64, 64)
        self.dec1 = Up(64, 64, 64)
        self.dec0 = _UpNoSkip(64, 32)   # H/2 → H, no skip

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)           # 64,  H/2
        s1 = self.enc1(self.pool(s0))  # 64,  H/4
        s2 = self.enc2(s1)          # 128, H/8
        s3 = self.enc3(s2)          # 256, H/16
        x  = self.enc4(s3)          # 512, H/32
        x  = self.dec4(x, s3)       # 256, H/16
        x  = self.dec3(x, s2)       # 128, H/8
        x  = self.dec2(x, s1)       # 64,  H/4
        x  = self.dec1(x, s0)       # 64,  H/2
        x  = self.dec0(x)           # 32,  H
        return self.head(x)


def build_resnet_model(num_classes: int = 9, pretrained: bool = True) -> AIR2ResNetSeg:
    return AIR2ResNetSeg(num_classes=num_classes, pretrained=pretrained)
