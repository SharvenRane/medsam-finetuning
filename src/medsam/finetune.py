"""Finetuning loop.

A standard supervised loop: encode the image and prompt, decode a mask, score it
against the ground truth with the Dice plus focal objective, and step an
optimiser. The function records the loss at every step so callers (and tests) can
confirm the loss goes down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from .dataset import (
    SyntheticBlobDataset,
    collate_box_batch,
    collate_point_batch,
)
from .losses import dice_coefficient, segmentation_loss
from .model import TinySam


@dataclass
class FinetuneResult:
    """Outcome of a finetuning run."""

    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    final_dice: float = 0.0
    model: TinySam | None = None


def _forward_batch(model, batch, prompt_kind):
    if prompt_kind == "point":
        images, masks, (coords, labels) = batch
        logits = model(images, points=(coords, labels))
    else:
        images, masks, boxes = batch
        logits = model(images, boxes=boxes)
    return logits, masks


def finetune(
    model: TinySam | None = None,
    dataset: SyntheticBlobDataset | None = None,
    prompt_kind: str = "point",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-3,
    dice_weight: float = 1.0,
    focal_weight: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> FinetuneResult:
    """Finetune a TinySam model on a promptable segmentation dataset.

    Args:
        model: model to train; a fresh TinySam is built if None.
        dataset: dataset to train on; a SyntheticBlobDataset is built if None.
        prompt_kind: "point" or "box", must match the dataset prompt.
        epochs: number of passes over the dataset.
        batch_size: batch size.
        lr: AdamW learning rate.
        dice_weight, focal_weight: loss weights.
        seed: torch seed for reproducible init and shuffling.
        device: device to train on.
    Returns:
        FinetuneResult with per step and per epoch losses, the final mean Dice,
        and the trained model.
    """
    if prompt_kind not in ("point", "box"):
        raise ValueError("prompt_kind must be 'point' or 'box'")

    torch.manual_seed(seed)
    device = torch.device(device)

    if dataset is None:
        dataset = SyntheticBlobDataset(
            num_samples=16, prompt=prompt_kind, seed=seed
        )
    if model is None:
        model = TinySam(image_size=dataset.image_size)
    model = model.to(device)

    collate = collate_point_batch if prompt_kind == "point" else collate_box_batch
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    result = FinetuneResult(model=model)

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in loader:
            batch = _move_batch(batch, device, prompt_kind)
            logits, masks = _forward_batch(model, batch, prompt_kind)
            loss = segmentation_loss(
                logits, masks, dice_weight=dice_weight, focal_weight=focal_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            result.step_losses.append(float(loss.detach().cpu()))
            epoch_loss += float(loss.detach().cpu())
            num_batches += 1
        result.epoch_losses.append(epoch_loss / max(1, num_batches))

    result.final_dice = _evaluate_dice(model, dataset, prompt_kind, batch_size, device)
    return result


def _move_batch(batch, device, prompt_kind):
    if prompt_kind == "point":
        images, masks, (coords, labels) = batch
        return (
            images.to(device),
            masks.to(device),
            (coords.to(device), labels.to(device)),
        )
    images, masks, boxes = batch
    return images.to(device), masks.to(device), boxes.to(device)


@torch.no_grad()
def _evaluate_dice(model, dataset, prompt_kind, batch_size, device) -> float:
    model.eval()
    collate = collate_point_batch if prompt_kind == "point" else collate_box_batch
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    total = 0.0
    count = 0
    for batch in loader:
        batch = _move_batch(batch, device, prompt_kind)
        logits, masks = _forward_batch(model, batch, prompt_kind)
        total += float(dice_coefficient(logits, masks).cpu())
        count += 1
    return total / max(1, count)
