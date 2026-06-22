# medsam-finetuning

A small finetuning harness for a SAM style promptable segmentation model adapted
to medical images. SAM takes an image plus a prompt (a click, a box) and returns
a mask for whatever the prompt points at. MedSAM is the line of work that adapts
that idea to scans, where a clinician marks a lesion and the model fills in the
boundary. This repo reproduces that flow at a size you can run and test on a CPU
in seconds, with no large checkpoint to download.

The trick that keeps it light is the image encoder. The real SAM encoder is a
heavy ViT that only works with a big pretrained weight file. Here it is swapped
for a tiny convolutional encoder that produces a feature grid of the same shape.
Everything around it is real: the prompt encoder, the two way attention mask
decoder, the Dice plus focal objective, and the training loop. So you can study
and exercise the architecture end to end, and the same code would accept a real
encoder if you wanted to plug one in.

## What is inside

`src/medsam/` holds the components:

- `prompt_encoder.py` turns point and box prompts into sparse token embeddings.
  Coordinates are normalised and passed through a random Fourier feature
  positional encoding, then added to a learned embedding per prompt type
  (positive click, negative click, box corners). A box becomes two labelled
  corner tokens, matching how SAM encodes it.
- `image_encoder.py` is the tiny stand in encoder. It downsamples a grayscale
  image to a dense feature grid and is fully trainable.
- `mask_decoder.py` is a lightweight two way transformer. A learned mask token
  and the prompt tokens attend to the flattened image features and back again,
  the updated mask token becomes a per pixel filter, and that filter applied to
  an upscaled feature grid gives a low resolution mask that is resized to full
  resolution.
- `model.py` wires the three together into `TinySam`, including the positional
  encoding grid for the dense image features.
- `losses.py` has soft Dice, a Dice coefficient metric, focal loss, and the
  combined objective that MedSAM trains with.
- `dataset.py` builds synthetic data: a bright elliptical blob on noisy
  background, the ground truth mask, and a prompt (a positive point at the
  centre, or a box around the blob). Blobs are deterministic given a seed.
- `finetune.py` is the supervised loop. It records the loss at every step and
  every epoch and reports the final mean Dice.

## Install

The code needs PyTorch and NumPy. With the project's virtual environment:

```
pip install -r requirements.txt
```

## Run the tests

```
python -m pytest tests/ -q
```

The suite checks behaviour, not just imports. A point prompt and a box prompt
each produce a mask at the right resolution. The prompt encoder gives different
embeddings to different positions and to positive versus negative clicks. The
synthetic point always lands inside its mask and the box always contains it.
Dice is well behaved (a perfect prediction scores near zero loss, a better
prediction scores higher Dice than a worse one). Finetuning drives the Dice loss
down for both prompt types and lifts the final Dice above the random init.

On this machine all 25 tests pass in a few seconds on CPU.

## A finetuning run

Training a fresh `TinySam` for eight epochs on sixteen synthetic samples
produced these numbers in one run:

| prompt | first epoch loss | last epoch loss | final mean Dice |
| ------ | ---------------- | --------------- | --------------- |
| point  | 0.947            | 0.080           | 0.943           |
| box    | 0.937            | 0.079           | 0.947           |

Your numbers will shift a little with seed and hardware, but the direction is
the point: the loss falls and the Dice climbs as the model learns to segment the
prompted blob.

```python
from medsam.dataset import SyntheticBlobDataset
from medsam.finetune import finetune

ds = SyntheticBlobDataset(num_samples=16, prompt="point", seed=0)
result = finetune(dataset=ds, prompt_kind="point", epochs=8, batch_size=4, lr=2e-3)
print(result.epoch_losses[0], result.epoch_losses[-1], result.final_dice)
```

## Predicting a mask

```python
import torch
from medsam.model import TinySam
from medsam.prompt_encoder import PromptEncoder

model = TinySam(image_size=(64, 64))
image = torch.randn(1, 1, 64, 64)

# a single positive click at (32, 32)
coords = torch.tensor([[[32.0, 32.0]]])
labels = torch.tensor([[PromptEncoder.LABEL_POSITIVE]])
mask = model.predict_mask(image, points=(coords, labels))   # (1, 1, 64, 64) bool

# or a box prompt as (x0, y0, x1, y1)
box = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
mask = model.predict_mask(image, boxes=box)
```

## Notes

The synthetic data and the tiny encoder are there so the whole thing trains
offline in seconds. The architecture is the real contribution: prompt encoding,
the two way attention decoder, and the Dice driven finetuning loop are the same
pieces you would use to finetune a real SAM checkpoint on a medical dataset.
