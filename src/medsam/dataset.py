"""Synthetic segmentation data.

Each sample is a grayscale image with a bright elliptical blob on a noisy
background, plus the ground truth mask of that blob and a prompt that points at
it. This stands in for medical scans where a clinician clicks a lesion or draws a
box around it. The blobs are deterministic given a seed so tests are reproducible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .prompt_encoder import PromptEncoder


class SyntheticBlobDataset(Dataset):
    """Images with a single elliptical foreground blob and a matching prompt.

    Args:
        num_samples: number of items in the dataset.
        image_size: (H, W) of each image.
        prompt: "point" to emit a single positive point at the blob centre, or
            "box" to emit a bounding box around the blob.
        noise: standard deviation of background noise.
        seed: base seed; sample i uses seed + i so items are stable.
    """

    def __init__(
        self,
        num_samples: int = 16,
        image_size: tuple[int, int] = (64, 64),
        prompt: str = "point",
        noise: float = 0.1,
        seed: int = 0,
    ):
        if prompt not in ("point", "box"):
            raise ValueError("prompt must be 'point' or 'box'")
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompt = prompt
        self.noise = noise
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def _make_sample(self, index: int):
        rng = np.random.default_rng(self.seed + index)
        height, width = self.image_size

        # blob centre and radii kept inside a margin
        margin = 0.2
        cy = rng.uniform(margin, 1 - margin) * height
        cx = rng.uniform(margin, 1 - margin) * width
        ry = rng.uniform(0.12, 0.22) * height
        rx = rng.uniform(0.12, 0.22) * width

        ys = np.arange(height)[:, None]
        xs = np.arange(width)[None, :]
        ellipse = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
        mask = (ellipse <= 1.0).astype(np.float32)

        image = rng.normal(0.0, self.noise, size=(height, width)).astype(np.float32)
        image += mask * 1.0  # foreground is brighter
        image = np.clip(image, -3, 3)

        image_t = torch.from_numpy(image)[None, ...]  # (1, H, W)
        mask_t = torch.from_numpy(mask)[None, ...]  # (1, H, W)

        if self.prompt == "point":
            coords = torch.tensor([[cx, cy]], dtype=torch.float32)  # (1, 2) xy
            labels = torch.tensor([PromptEncoder.LABEL_POSITIVE], dtype=torch.long)
            prompt_data = {"point_coords": coords, "point_labels": labels}
        else:
            x0 = max(0.0, cx - rx)
            y0 = max(0.0, cy - ry)
            x1 = min(width - 1.0, cx + rx)
            y1 = min(height - 1.0, cy + ry)
            box = torch.tensor([x0, y0, x1, y1], dtype=torch.float32)  # (4,)
            prompt_data = {"box": box}

        return image_t, mask_t, prompt_data

    def __getitem__(self, index: int):
        image, mask, prompt_data = self._make_sample(index)
        sample = {"image": image, "mask": mask}
        sample.update(prompt_data)
        return sample


def collate_point_batch(samples: list[dict]):
    """Collate point prompt samples into batched tensors."""
    images = torch.stack([s["image"] for s in samples], dim=0)
    masks = torch.stack([s["mask"] for s in samples], dim=0)
    coords = torch.stack([s["point_coords"] for s in samples], dim=0)  # (B, 1, 2)
    labels = torch.stack([s["point_labels"] for s in samples], dim=0)  # (B, 1)
    return images, masks, (coords, labels)


def collate_box_batch(samples: list[dict]):
    """Collate box prompt samples into batched tensors."""
    images = torch.stack([s["image"] for s in samples], dim=0)
    masks = torch.stack([s["mask"] for s in samples], dim=0)
    boxes = torch.stack([s["box"] for s in samples], dim=0)  # (B, 4)
    return images, masks, boxes
