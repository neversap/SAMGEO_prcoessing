from __future__ import annotations

import torch
import torch.nn as nn


def build_model(config: dict) -> nn.Module:
    model_config = config["model"]
    architecture = model_config.get("architecture", "unet").lower()
    if architecture in {"unet", "unet_effb3", "smp_unet"}:
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "U-Net + EfficientNet-B3 requires segmentation-models-pytorch. "
                "Install training dependencies with requirements-train.txt."
            ) from exc
        return smp.Unet(
            encoder_name=model_config.get("encoder", "efficientnet-b3"),
            encoder_weights=model_config.get("encoder_weights", "imagenet"),
            in_channels=int(model_config.get("in_channels", 3)),
            classes=int(model_config.get("num_classes", 3)),
            activation=None,
        )
    if architecture == "simple_unet":
        return SimpleUNet(
            in_channels=int(model_config.get("in_channels", 3)),
            num_classes=int(model_config.get("num_classes", 3)),
        )
    raise ValueError(f"Unknown model architecture: {architecture}")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))
        x = self.up3(b)
        x = self.conv3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([x, d1], dim=1))
        return self.head(x)
