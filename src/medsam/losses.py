"""Segmentation losses and metrics.

Dice is the primary objective because medical masks are often small and
imbalanced, where pixel cross entropy alone underweights the foreground. A focal
term is offered as an optional complement.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_coefficient(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice coefficient in [0, 1], higher is better.

    Args:
        logits: (B, 1, H, W) raw mask logits.
        targets: (B, 1, H, W) float masks in {0, 1}.
        eps: smoothing constant.
    Returns:
        scalar mean Dice over the batch.
    """
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Soft Dice loss, equal to 1 minus the Dice coefficient."""
    return 1.0 - dice_coefficient(logits, targets, eps=eps)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss on a per pixel basis, averaged over the batch."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted sum of Dice and focal loss, the combination MedSAM trains with."""
    loss = dice_weight * dice_loss(logits, targets)
    if focal_weight > 0:
        loss = loss + focal_weight * sigmoid_focal_loss(logits, targets)
    return loss
