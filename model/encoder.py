"""
CNN Encoder: Frame → Latent Token.

4-stage convolutional downsampler that compresses a single frame
(C, 64, 64) into a latent vector of dimension d_model.
Includes residual connections with 1×1 shortcut projections for
stable gradient flow through deeper networks.
"""

import torch
import torch.nn as nn


class ResConvBlock(nn.Module):
    """Conv2d → BatchNorm → SiLU with residual shortcut and optional dropout."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, dropout: float = 0.05):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        # 1x1 shortcut projection to match dimensions
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


class CNNEncoder(nn.Module):
    """
    Encodes a single frame into a latent vector.

    Input:  (B, C, 64, 64)
    Output: (B, d_model)

    Architecture:
        Stage 1: (C, 64, 64) → (32, 32, 32)
        Stage 2: (32, 32, 32) → (64, 16, 16)
        Stage 3: (64, 16, 16) → (128, 8, 8)
        Stage 4: (128, 8, 8) → (256, 4, 4)
        Pool:    (256, 4, 4) → (256,)
        Project: (256,) → (d_model,)
    """

    def __init__(
        self,
        in_channels: int = 1,
        channel_progression: tuple[int, ...] = (32, 64, 128, 256),
        d_model: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()

        channels = [in_channels] + list(channel_progression)

        # Build residual convolutional stages
        self.stages = nn.Sequential(*[
            ResConvBlock(channels[i], channels[i + 1], stride=2, dropout=dropout)
            for i in range(len(channel_progression))
        ])

        # Global average pooling → latent projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(channel_progression[-1], d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frame, shape (B, C, H, W).

        Returns:
            Latent vector, shape (B, d_model).
        """
        h = self.stages(x)              # (B, 256, 4, 4)
        h = self.pool(h)                # (B, 256, 1, 1)
        h = h.flatten(1)                # (B, 256)
        z = self.project(h)             # (B, d_model)
        return z

    def extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract intermediate features for perceptual loss.

        Returns features after each stage.
        """
        features = []
        h = x
        for stage in self.stages:
            h = stage(h)
            features.append(h)
        return features
