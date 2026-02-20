# Global Context for SegFormer Tile Segmentation

## Problem
Tiles are processed in isolation. A stone texture might be a wall or a floor —
you can only tell by seeing the surrounding map structure. The model needs
map-level awareness to disambiguate.

## Approach
Before processing tiles, downsample the full battlemap to 256×256 and run it
through a lightweight CNN to produce a **global context vector**. This vector
is then spatially broadcast and concatenated to the SegFormer decoder features
so every tile prediction is conditioned on the whole map.

```
Full image (any size) ──resize──> 256×256 ──GlobalEncoder──> 128-dim vector
                                                                 │
                                                      broadcast to (H, W)
                                                                 │
Tile (512×512) ──SegFormer encoder──> decoder features ──concat──> adjusted decoder ──> prediction
```

## Architecture: GlobalContextSegFormer

### GlobalEncoder (new, ~0.5M params)
Small CNN that compresses the full image to a fixed-length vector:
```
Conv2d(3, 32, 3, stride=2, padding=1)  → 128×128
Conv2d(32, 64, 3, stride=2, padding=1) → 64×64
Conv2d(64, 128, 3, stride=2, padding=1) → 32×32
AdaptiveAvgPool2d(1) → 128-dim vector
```
Each conv followed by BatchNorm + ReLU.

### Fusion
The SegFormer MLP decoder concatenates multi-scale features into a single
tensor of shape (B, decoder_dim, H/4, W/4). We:
1. Take the 128-dim global vector
2. Broadcast it to (B, 128, H/4, W/4)
3. Concatenate with the decoder features → (B, decoder_dim + 128, H/4, W/4)
4. Replace the final classifier conv to accept the wider input

### Why this works
- Global encoder sees the entire map layout at low resolution
- Learns "this map has stone dungeon corridors" vs "this is an outdoor path"
- Each tile gets that context for free via concatenation
- Minimal overhead: 256×256 global pass is fast, vector broadcast is free

## File Changes

### 1. `primary_model_training/model.py`
- Add `GlobalEncoder` class (small CNN → 128-dim vector)
- Add `GlobalContextSegFormer` class that wraps SegFormerWrapper:
  - `__init__`: creates GlobalEncoder + loads SegFormer, replaces classifier
  - `forward(tiles, global_image)`: runs both paths, fuses, predicts
  - The SegFormer decode_head concatenates features to 256 channels, then
    a classifier conv maps 256 → num_classes. We replace that classifier
    with one that maps (256 + 128) → num_classes.

### 2. `primary_model_training/dataset.py`
- `WallSegmentationDataset.__init__`: add `global_image_size: int = 256` param
- `__getitem__`: alongside the tile, also return `'global_image'` — the full
  source image resized to `global_image_size × global_image_size` and
  normalized (same norm as tiles)
- No change to tile extraction logic

### 3. `primary_model_training/config.py`
- Add `use_global_context: bool = False` — auto-set True when model_type is
  `"segformer_gc"`
- Add `global_image_size: int = 256`
- Add `global_context_dim: int = 128`

### 4. `primary_model_training/train.py`
- Add `--model segformer_gc` choice
- In training loop, pass `global_image` to model when config requires it
- Update `Trainer.__init__` to pass `global_image_size` to dataset

### 5. `primary_model_training/inference.py`
- `load_model`: handle `model_type="segformer_gc"`
- `predict_mask`: pass downscaled full image alongside each tile batch

## Training
- Global encoder trains from scratch (randomly initialized)
- SegFormer encoder+decoder keep pretrained ADE20K weights
- Both train jointly end-to-end with same optimizer
- The global encoder will learn quickly since it's small and gets gradient
  signal from every tile

## Verification
```
python -m primary_model_training.train --model segformer_gc --merge-terrain --epochs 5 --batch-size 4
```
Confirm: no shape errors, loss decreases, VRAM fits in 8GB.
