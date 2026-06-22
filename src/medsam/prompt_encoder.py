"""Prompt encoder for points and boxes.

This mirrors the structure of the SAM prompt encoder. Sparse prompts (points and
boxes) are turned into positional embeddings using a random Fourier feature
projection of normalised coordinates, then added to learned type embeddings. The
result is a small set of embedding vectors that the mask decoder attends to.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncodingRandom(nn.Module):
    """Random Fourier feature positional encoding for 2D coordinates.

    Coordinates are expected in pixel space. They are normalised to [0, 1] by the
    image size, shifted to [-1, 1], scaled by a fixed random Gaussian matrix, then
    mapped through sin and cos. This is the same trick SAM uses so that nearby
    points get smoothly varying embeddings.
    """

    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0):
        super().__init__()
        if num_pos_feats % 2 != 0:
            raise ValueError("num_pos_feats must be even")
        # register as buffer so it moves with .to(device) and is saved in state
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn(2, num_pos_feats // 2),
        )

    def _encode(self, coords: torch.Tensor) -> torch.Tensor:
        # coords in [0, 1]; map to [-1, 1]
        coords = 2.0 * coords - 1.0
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2.0 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward_with_coords(
        self, coords: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        """Encode pixel coordinates.

        Args:
            coords: (..., 2) tensor of (x, y) pixel coordinates.
            image_size: (height, width) of the image the coords live in.
        Returns:
            (..., num_pos_feats) positional embeddings.
        """
        coords = coords.clone().float()
        coords[..., 0] = coords[..., 0] / image_size[1]
        coords[..., 1] = coords[..., 1] / image_size[0]
        return self._encode(coords)


class PromptEncoder(nn.Module):
    """Encode point and box prompts into sparse embeddings.

    Args:
        embed_dim: dimensionality of the embeddings the decoder consumes.
        image_size: (height, width) prompts are expressed in.
    """

    # point label conventions, matching SAM
    LABEL_NEGATIVE = 0
    LABEL_POSITIVE = 1
    LABEL_BOX_TOP_LEFT = 2
    LABEL_BOX_BOTTOM_RIGHT = 3
    LABEL_PADDING = -1

    def __init__(self, embed_dim: int = 32, image_size: tuple[int, int] = (64, 64)):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.pe_layer = PositionalEncodingRandom(embed_dim)

        # one learned embedding per point type: negative, positive, box corners,
        # and a "not a point" embedding for padded slots.
        self.num_point_embeddings = 4
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        )
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(
        self, points: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        # shift coords to pixel centres to match SAM
        points = points + 0.5
        point_embedding = self.pe_layer.forward_with_coords(points, self.image_size)

        # add the learned per-type embedding
        out = point_embedding.clone()
        out[labels == self.LABEL_PADDING] = 0.0
        out[labels == self.LABEL_PADDING] += self.not_a_point_embed.weight
        for label_value in range(self.num_point_embeddings):
            mask = labels == label_value
            out[mask] += self.point_embeddings[label_value].weight
        return out

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        # boxes: (B, 4) as (x0, y0, x1, y1). Represent as two labelled corner points.
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[self.LABEL_BOX_TOP_LEFT].weight
        corner_embedding[:, 1, :] += self.point_embeddings[
            self.LABEL_BOX_BOTTOM_RIGHT
        ].weight
        return corner_embedding

    def forward(
        self,
        points: tuple[torch.Tensor, torch.Tensor] | None = None,
        boxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build sparse prompt embeddings.

        Args:
            points: optional (coords, labels) where coords is (B, N, 2) pixel xy and
                labels is (B, N) with values from the LABEL_* constants.
            boxes: optional (B, 4) tensor of (x0, y0, x1, y1) boxes.
        Returns:
            (B, num_tokens, embed_dim) sparse embeddings. At least one of points or
            boxes must be provided.
        """
        if points is None and boxes is None:
            raise ValueError("Provide at least one of points or boxes")

        batch_size = self._infer_batch_size(points, boxes)
        sparse = torch.empty(
            (batch_size, 0, self.embed_dim), device=self._device()
        )

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels)
            sparse = torch.cat([sparse, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse = torch.cat([sparse, box_embeddings], dim=1)

        return sparse

    def _infer_batch_size(self, points, boxes) -> int:
        if points is not None:
            return points[0].shape[0]
        return boxes.shape[0]

    def _device(self) -> torch.device:
        return self.point_embeddings[0].weight.device
