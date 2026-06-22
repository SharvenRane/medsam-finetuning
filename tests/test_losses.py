import torch

from medsam.losses import (
    dice_coefficient,
    dice_loss,
    segmentation_loss,
    sigmoid_focal_loss,
)


def test_perfect_prediction_has_low_dice_loss():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    # strong logits that match the target
    logits = (target * 2 - 1) * 20.0
    loss = dice_loss(logits, target)
    assert loss.item() < 0.05


def test_dice_coefficient_in_unit_range():
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 1, 16, 16) > 0.5).float()
    d = dice_coefficient(logits, target)
    assert 0.0 <= d.item() <= 1.0


def test_better_prediction_has_higher_dice():
    target = torch.zeros(1, 1, 16, 16)
    target[:, :, 4:12, 4:12] = 1.0
    good = (target * 2 - 1) * 10.0
    bad = (target * 2 - 1) * -10.0  # inverted
    assert dice_coefficient(good, target) > dice_coefficient(bad, target)


def test_focal_loss_non_negative():
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    assert sigmoid_focal_loss(logits, target).item() >= 0.0


def test_segmentation_loss_combines_terms():
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    only_dice = segmentation_loss(logits, target, dice_weight=1.0, focal_weight=0.0)
    both = segmentation_loss(logits, target, dice_weight=1.0, focal_weight=1.0)
    assert both.item() > only_dice.item()
