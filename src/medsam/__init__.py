"""medsam: a small finetuning harness for SAM style promptable segmentation.

The package keeps every component small enough to run on CPU with a tiny stand
in image encoder, so the architecture can be exercised end to end without
downloading a large checkpoint.
"""

from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .image_encoder import TinyImageEncoder
from .model import TinySam
from .losses import dice_loss, dice_coefficient, sigmoid_focal_loss, segmentation_loss
from .dataset import SyntheticBlobDataset
from .finetune import finetune

__all__ = [
    "PromptEncoder",
    "MaskDecoder",
    "TinyImageEncoder",
    "TinySam",
    "dice_loss",
    "dice_coefficient",
    "sigmoid_focal_loss",
    "segmentation_loss",
    "SyntheticBlobDataset",
    "finetune",
]
