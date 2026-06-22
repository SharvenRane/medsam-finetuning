"""Mask decoder head.

This follows the SAM idea of a lightweight two way transformer: a small set of
tokens (a learned mask token plus the sparse prompt embeddings) attend to the
flattened image embedding and back again. The updated mask token is projected
into a spatial filter that is applied to an upsampled image embedding to produce
a low resolution mask, which is then resized to the full image resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoWayAttentionBlock(nn.Module):
    """One block of cross attention in both directions plus an MLP."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

        self.cross_attn_image_to_token = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # self attention among tokens
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)

        # tokens attend to image (positional embedding added to image keys)
        q = tokens
        k = image_embedding + image_pe
        attn_out, _ = self.cross_attn_token_to_image(q, k, image_embedding)
        tokens = self.norm2(tokens + attn_out)

        # token MLP
        tokens = self.norm3(tokens + self.mlp(tokens))

        # image attends back to tokens
        q = image_embedding + image_pe
        attn_out, _ = self.cross_attn_image_to_token(q, tokens, tokens)
        image_embedding = self.norm4(image_embedding + attn_out)

        return tokens, image_embedding


class MaskDecoder(nn.Module):
    """Predict a segmentation mask from image and prompt embeddings.

    Args:
        embed_dim: width of the token and image embeddings.
        num_heads: attention heads in the two way transformer.
        num_blocks: number of two way attention blocks.
        mlp_dim: hidden width of the token MLP.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        mlp_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.mask_token = nn.Embedding(1, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TwoWayAttentionBlock(embed_dim, num_heads, mlp_dim)
                for _ in range(num_blocks)
            ]
        )
        self.final_norm_tokens = nn.LayerNorm(embed_dim)

        # upsample the image embedding by 4x before applying the mask filter
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(8, embed_dim // 2), num_channels=embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # project the mask token into the filter that is dotted with the upscaled grid
        self.mask_token_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 4),
        )

    def forward(
        self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Decode a mask.

        Args:
            image_embedding: (B, embed_dim, Hf, Wf) dense image features.
            image_pe: (B, embed_dim, Hf, Wf) positional encoding for the image grid.
            sparse_prompt_embeddings: (B, N, embed_dim) prompt tokens.
            output_size: (H, W) of the full resolution mask to return.
        Returns:
            (B, 1, H, W) mask logits at the requested resolution.
        """
        batch_size, channels, height, width = image_embedding.shape

        # build the token sequence: one mask token then the prompt tokens
        mask_tokens = self.mask_token.weight.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat([mask_tokens, sparse_prompt_embeddings], dim=1)

        # flatten image to (B, Hf*Wf, C)
        image_seq = image_embedding.flatten(2).permute(0, 2, 1)
        pe_seq = image_pe.flatten(2).permute(0, 2, 1)

        for block in self.blocks:
            tokens, image_seq = block(tokens, image_seq, pe_seq)
        tokens = self.final_norm_tokens(tokens)

        # the first token is the mask token after attention
        mask_token_out = tokens[:, 0, :]
        mask_filter = self.mask_token_mlp(mask_token_out)  # (B, embed_dim//4)

        # reshape attended image back to a grid and upscale it
        image_grid = image_seq.permute(0, 2, 1).reshape(
            batch_size, channels, height, width
        )
        upscaled = self.output_upscaling(image_grid)  # (B, embed_dim//4, 4Hf, 4Wf)

        # dot the per pixel feature with the mask filter -> (B, 1, 4Hf, 4Wf)
        b, c, h, w = upscaled.shape
        low_res_logits = (mask_filter.view(b, 1, c) @ upscaled.view(b, c, h * w)).view(
            b, 1, h, w
        )

        # resize to the requested full resolution
        mask_logits = F.interpolate(
            low_res_logits, size=output_size, mode="bilinear", align_corners=False
        )
        return mask_logits
