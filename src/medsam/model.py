"""TinySam: image encoder, prompt encoder, and mask decoder wired together.

This is a faithful but small reproduction of the SAM promptable segmentation
flow. It takes an image and a point or box prompt and returns a mask. Everything
is trainable on CPU, which is what makes the finetuning tests possible without a
large checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .image_encoder import TinyImageEncoder
from .mask_decoder import MaskDecoder
from .prompt_encoder import PositionalEncodingRandom, PromptEncoder


class TinySam(nn.Module):
    """Promptable segmentation model.

    Args:
        in_channels: image channel count.
        embed_dim: shared embedding width across the three components.
        image_size: (H, W) of the input images.
        encoder_downsample: spatial downsampling of the image encoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 32,
        image_size: tuple[int, int] = (64, 64),
        encoder_downsample: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.image_encoder = TinyImageEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            downsample=encoder_downsample,
        )
        self.prompt_encoder = PromptEncoder(embed_dim=embed_dim, image_size=image_size)
        self.mask_decoder = MaskDecoder(embed_dim=embed_dim)

        # positional encoding for the dense image grid, shared with the decoder
        self.grid_pe = PositionalEncodingRandom(embed_dim)
        self.encoder_downsample = encoder_downsample

    def _image_positional_encoding(
        self, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        """Build a (1, embed_dim, height, width) positional grid for the features."""
        ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
        ys = ys / height
        xs = xs / width
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2) in [0,1]
        pe = self.grid_pe._encode(coords)  # (H, W, embed_dim)
        return pe.permute(2, 0, 1).unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict mask logits for a batch.

        Args:
            images: (B, C, H, W) input images.
            points: optional (coords (B, N, 2), labels (B, N)) point prompts.
            boxes: optional (B, 4) box prompts as (x0, y0, x1, y1).
        Returns:
            (B, 1, H, W) mask logits at the input resolution.
        """
        batch_size = images.shape[0]
        image_embedding = self.image_encoder(images)
        _, _, hf, wf = image_embedding.shape

        image_pe = self._image_positional_encoding(hf, wf, images.device)
        image_pe = image_pe.expand(batch_size, -1, -1, -1)

        sparse_embeddings = self.prompt_encoder(points=points, boxes=boxes)

        output_size = (images.shape[2], images.shape[3])
        mask_logits = self.mask_decoder(
            image_embedding=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            output_size=output_size,
        )
        return mask_logits

    @torch.no_grad()
    def predict_mask(
        self,
        images: torch.Tensor,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Return a boolean mask by thresholding the logits at `threshold`."""
        logits = self.forward(images, points=points, boxes=boxes)
        return logits > threshold
