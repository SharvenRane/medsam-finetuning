import torch

from medsam.dataset import SyntheticBlobDataset
from medsam.finetune import finetune
from medsam.model import TinySam
from medsam.prompt_encoder import PromptEncoder


def _trend(values, window=3):
    """Mean of the first and last `window` values."""
    start = sum(values[:window]) / window
    end = sum(values[-window:]) / window
    return start, end


def test_finetune_reduces_dice_loss_point_prompt():
    torch.manual_seed(0)
    ds = SyntheticBlobDataset(num_samples=16, prompt="point", seed=0)
    result = finetune(
        dataset=ds, prompt_kind="point", epochs=8, batch_size=4, lr=2e-3, seed=0
    )
    assert len(result.epoch_losses) == 8
    # the loss at the end of training should be clearly below the start
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    start, end = _trend(result.step_losses, window=4)
    assert end < start


def test_finetune_reduces_dice_loss_box_prompt():
    torch.manual_seed(0)
    ds = SyntheticBlobDataset(num_samples=16, prompt="box", seed=1)
    result = finetune(
        dataset=ds, prompt_kind="box", epochs=8, batch_size=4, lr=2e-3, seed=0
    )
    assert result.epoch_losses[-1] < result.epoch_losses[0]


def test_finetune_improves_dice_over_random_init():
    torch.manual_seed(0)
    ds = SyntheticBlobDataset(num_samples=16, prompt="point", seed=2)
    model = TinySam(image_size=ds.image_size)

    # Dice before any training
    model.eval()
    with torch.no_grad():
        s = ds[0]
        logits0 = model(
            s["image"][None],
            points=(s["point_coords"][None], s["point_labels"][None]),
        )
    from medsam.losses import dice_coefficient

    target = s["mask"][None]
    dice_before = float(dice_coefficient(logits0, target))

    result = finetune(
        model=model, dataset=ds, prompt_kind="point", epochs=10, batch_size=4,
        lr=2e-3, seed=0,
    )
    assert result.final_dice > dice_before


def test_finetune_runs_with_defaults():
    result = finetune(epochs=2, batch_size=4)
    assert result.model is not None
    assert len(result.step_losses) > 0
    assert all(loss == loss for loss in result.step_losses)  # no NaNs
