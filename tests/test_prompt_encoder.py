import torch

from medsam.prompt_encoder import PositionalEncodingRandom, PromptEncoder


def test_point_embedding_shape():
    enc = PromptEncoder(embed_dim=32, image_size=(64, 64))
    coords = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (B=1, N=2, 2)
    labels = torch.tensor([[PromptEncoder.LABEL_POSITIVE, PromptEncoder.LABEL_NEGATIVE]])
    out = enc(points=(coords, labels))
    assert out.shape == (1, 2, 32)
    assert torch.isfinite(out).all()


def test_box_embedding_is_two_tokens():
    enc = PromptEncoder(embed_dim=16, image_size=(64, 64))
    boxes = torch.tensor([[5.0, 5.0, 40.0, 50.0]])  # (B=1, 4)
    out = enc(boxes=boxes)
    # a box becomes two corner tokens
    assert out.shape == (1, 2, 16)


def test_points_and_boxes_concatenate():
    enc = PromptEncoder(embed_dim=16, image_size=(64, 64))
    coords = torch.tensor([[[10.0, 20.0]]])
    labels = torch.tensor([[PromptEncoder.LABEL_POSITIVE]])
    boxes = torch.tensor([[5.0, 5.0, 40.0, 50.0]])
    out = enc(points=(coords, labels), boxes=boxes)
    # 1 point token + 2 box corner tokens
    assert out.shape == (1, 3, 16)


def test_requires_a_prompt():
    enc = PromptEncoder()
    try:
        enc()
    except ValueError:
        return
    raise AssertionError("expected ValueError when no prompt is given")


def test_positional_encoding_distinct_for_distinct_points():
    pe = PositionalEncodingRandom(num_pos_feats=64)
    coords = torch.tensor([[0.1, 0.1], [0.9, 0.9]])
    enc = pe.forward_with_coords(coords, (64, 64))
    assert enc.shape == (2, 64)
    # different positions should map to different embeddings
    assert not torch.allclose(enc[0], enc[1])


def test_positive_and_negative_labels_differ():
    enc = PromptEncoder(embed_dim=32, image_size=(64, 64))
    coords = torch.tensor([[[10.0, 20.0]]])
    pos = enc(points=(coords, torch.tensor([[PromptEncoder.LABEL_POSITIVE]])))
    neg = enc(points=(coords, torch.tensor([[PromptEncoder.LABEL_NEGATIVE]])))
    # same coordinate but different learned type embedding
    assert not torch.allclose(pos, neg)
