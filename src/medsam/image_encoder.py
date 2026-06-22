"""A tiny stand in image encoder.

The real SAM image encoder is a heavy ViT that needs a large pretrained
checkpoint. For testing the finetuning architecture we only need something that
maps an image to a dense feature grid of the right shape, so this is a small
convolutional encoder that downsamples the input to a feature map. It is fully
trainable and deterministic, which is what the finetuning loop and the tests
need.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyImageEncoder(nn.Module):
    """Convolutional encoder producing a dense embedding grid.

    Args:
        in_channels: number of input image channels (1 for grayscale medical scans).
        embed_dim: channel width of the output feature grid.
        downsample: total spatial downsampling factor (power of two).
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 32, downsample: int = 4):
        super().__init__()
        if downsample not in (2, 4, 8, 16):
            raise ValueError("downsample must be one of 2, 4, 8, 16")
        self.embed_dim = embed_dim
        self.downsample = downsample

        layers: list[nn.Module] = []
        channels = in_channels
        current = 1
        width = max(8, embed_dim // 2)
        while current < downsample:
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(8, width), num_channels=width),
                nn.GELU(),
            ]
            channels = width
            current *= 2
        # final projection to embed_dim at the current resolution
        layers += [
            nn.Conv2d(channels, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(8, embed_dim), num_channels=embed_dim),
            nn.GELU(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: (B, C, H, W) input batch.
        Returns:
            (B, embed_dim, H/downsample, W/downsample) feature grid.
        """
        if images.dim() != 4:
            raise ValueError("images must be (B, C, H, W)")
        return self.net(images)
