import torch

from medsam.dataset import (
    SyntheticBlobDataset,
    collate_box_batch,
    collate_point_batch,
)
from medsam.prompt_encoder import PromptEncoder


def test_point_dataset_shapes_and_determinism():
    ds = SyntheticBlobDataset(num_samples=4, image_size=(64, 64), prompt="point", seed=7)
    a = ds[0]
    b = SyntheticBlobDataset(num_samples=4, prompt="point", seed=7)[0]
    assert a["image"].shape == (1, 64, 64)
    assert a["mask"].shape == (1, 64, 64)
    assert a["point_coords"].shape == (1, 2)
    assert a["point_labels"].tolist() == [PromptEncoder.LABEL_POSITIVE]
    # deterministic given the seed
    assert torch.allclose(a["image"], b["image"])


def test_point_lands_inside_the_mask():
    ds = SyntheticBlobDataset(num_samples=8, prompt="point", seed=3)
    for i in range(len(ds)):
        s = ds[i]
        x, y = s["point_coords"][0].tolist()
        assert s["mask"][0, int(round(y)), int(round(x))] == 1.0


def test_box_contains_the_mask():
    ds = SyntheticBlobDataset(num_samples=8, prompt="box", seed=5)
    for i in range(len(ds)):
        s = ds[i]
        x0, y0, x1, y1 = s["box"].tolist()
        ys, xs = torch.where(s["mask"][0] > 0)
        assert xs.min().item() >= int(x0)
        assert ys.min().item() >= int(y0)
        assert xs.max().item() <= round(x1) + 1
        assert ys.max().item() <= round(y1) + 1


def test_collate_point_batch():
    ds = SyntheticBlobDataset(num_samples=6, prompt="point", seed=1)
    images, masks, (coords, labels) = collate_point_batch([ds[i] for i in range(3)])
    assert images.shape == (3, 1, 64, 64)
    assert masks.shape == (3, 1, 64, 64)
    assert coords.shape == (3, 1, 2)
    assert labels.shape == (3, 1)


def test_collate_box_batch():
    ds = SyntheticBlobDataset(num_samples=6, prompt="box", seed=1)
    images, masks, boxes = collate_box_batch([ds[i] for i in range(3)])
    assert images.shape == (3, 1, 64, 64)
    assert boxes.shape == (3, 4)
