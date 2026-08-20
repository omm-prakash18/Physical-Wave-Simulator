"""
CNN Decoder: Latent Token → Frame.

Mirror of the encoder: linear projection → reshape → 4-stage
transposed-conv upsampler back to pixel space.
Includes residual connections with 1×1 transpose shortcuts.
"""

import torch
import torch.nn as nn


class ResConvTransposeBlock(nn.Module):
    """ConvTranspose2d → BatchNorm → SiLU with residual shortcut and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.05):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        # 1x1 transpose shortcut to match spatial + channel dims
        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


class CNNDecoder(nn.Module):
    """
    Decodes a latent vector back to a frame.

    Input:  (B, d_model)
    Output: (B, C, 64, 64)

    Architecture:
        Project: (d_model,) → (256 * 4 * 4,) → reshape (256, 4, 4)
        Stage 1: (256, 4, 4) → (128, 8, 8)
        Stage 2: (128, 8, 8) → (64, 16, 16)
        Stage 3: (64, 16, 16) → (32, 32, 32)
        Stage 4: (32, 32, 32) → (16, 64, 64)
        Output:  (16, 64, 64) → (C, 64, 64) via Conv + Tanh
    """

    def __init__(
        self,
        out_channels: int = 1,
        channel_progression: tuple[int, ...] = (256, 128, 64, 32),
        d_model: int = 256,
        init_spatial: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.init_channels = channel_progression[0]
        self.init_spatial = init_spatial

        # Latent → spatial grid
        self.project = nn.Sequential(
            nn.Linear(d_model, self.init_channels * init_spatial * init_spatial),
            nn.SiLU(inplace=True),
        )

        # Transposed conv upsampling stages with residual connections
        stages = []
        for i in range(len(channel_progression) - 1):
            stages.append(ResConvTransposeBlock(channel_progression[i], channel_progression[i + 1], dropout=dropout))

        # One more upsample to get from 32×32 to 64×64
        final_upsample_ch = 16
        stages.append(ResConvTransposeBlock(channel_progression[-1], final_upsample_ch, dropout=dropout))

        self.stages = nn.Sequential(*stages)

        # Final output convolution
        self.output = nn.Sequential(
            nn.Conv2d(final_upsample_ch, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector, shape (B, d_model).

        Returns:
            Reconstructed frame, shape (B, C, H, W).
        """
        h = self.project(z)  # (B, 256*4*4)
        h = h.view(-1, self.init_channels, self.init_spatial, self.init_spatial)  # (B, 256, 4, 4)
        h = self.stages(h)   # (B, 16, 64, 64)
        x = self.output(h)   # (B, C, 64, 64)
        return x
