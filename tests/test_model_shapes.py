import torch

from medsam.image_encoder import TinyImageEncoder
from medsam.model import TinySam
from medsam.prompt_encoder import PromptEncoder


def test_image_encoder_downsamples():
    enc = TinyImageEncoder(in_channels=1, embed_dim=32, downsample=4)
    x = torch.randn(2, 1, 64, 64)
    feat = enc(x)
    assert feat.shape == (2, 32, 16, 16)


def test_point_prompt_produces_full_resolution_mask():
    model = TinySam(in_channels=1, embed_dim=32, image_size=(64, 64))
    images = torch.randn(3, 1, 64, 64)
    coords = torch.tensor([[[32.0, 32.0]]]).expand(3, 1, 2).clone()
    labels = torch.full((3, 1), PromptEncoder.LABEL_POSITIVE)
    logits = model(images, points=(coords, labels))
    assert logits.shape == (3, 1, 64, 64)
    assert torch.isfinite(logits).all()


def test_box_prompt_produces_full_resolution_mask():
    model = TinySam(in_channels=1, embed_dim=32, image_size=(64, 64))
    images = torch.randn(2, 1, 64, 64)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [5.0, 8.0, 40.0, 60.0]])
    logits = model(images, boxes=boxes)
    assert logits.shape == (2, 1, 64, 64)
    assert torch.isfinite(logits).all()


def test_predict_mask_returns_boolean_of_right_shape():
    model = TinySam(image_size=(64, 64))
    images = torch.randn(1, 1, 64, 64)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    mask = model.predict_mask(images, boxes=boxes)
    assert mask.shape == (1, 1, 64, 64)
    assert mask.dtype == torch.bool


def test_non_square_image_size():
    model = TinySam(image_size=(48, 80))
    images = torch.randn(1, 1, 48, 80)
    coords = torch.tensor([[[40.0, 24.0]]])
    labels = torch.tensor([[PromptEncoder.LABEL_POSITIVE]])
    logits = model(images, points=(coords, labels))
    assert logits.shape == (1, 1, 48, 80)
