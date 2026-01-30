# Self-Supervised Pretraining for Battlemap Segmentation

This module implements multi-task self-supervised pretraining to learn general visual features from unlabeled battlemap images. The pretrained encoder can then be transferred to the segmentation model for improved generalization across different map styles.

## Overview

### The Problem
Models trained only on Watabou-style maps overfit to that specific visual style and fail on maps with different art styles (hand-drawn, 3D rendered, different color palettes, etc.).

### The Solution
Pretrain the encoder on **unlabeled** battlemaps using self-supervised tasks that teach general visual understanding:

1. **Edge Prediction**: Detect boundaries → transfers directly to wall detection
2. **Colorization**: Understand material differences → walls vs floors have different colors
3. **Masked Autoencoding**: Learn spatial structure → understand room layouts
4. **Jigsaw Puzzle**: Learn spatial relationships → how map elements connect

## Quick Start

### 1. Collect Unlabeled Battlemaps

Place battlemap images in a directory. No labels needed!

```
unlabeled_battlemaps/
├── reddit_maps/
├── generated_maps/
└── other_sources/
```

### 2. Run Pretraining

```bash
python train_pretrain.py \
    --data_dir ./unlabeled_battlemaps \
    --epochs 100 \
    --batch_size 8 \
    --output_dir ./pretrained_models
```

### 3. Fine-tune on Labeled Data

```bash
python train_finetune.py \
    --pretrained ./pretrained_models/encoder_only.pth \
    --train_images ./labeled_data/images \
    --train_masks ./labeled_data/masks \
    --num_classes 6
```

## Architecture

```
                         ┌─→ [Edge Decoder] ──────→ Edge Loss
                         │
[Battlemap] → [Shared Encoder] ─┼─→ [Color Decoder] ─→ Color Loss
                         │
                         ├─→ [MAE Decoder] ──────→ MAE Loss
                         │
                         └─→ [Jigsaw Head] ──────→ Jigsaw Loss

Total Loss = w_edge * L_edge + w_color * L_color + w_mae * L_mae + w_jigsaw * L_jigsaw
```

After pretraining, only the **Shared Encoder** is kept and transferred to the segmentation model.

## File Structure

```
SelfSupervisedPretraining/
├── models/
│   ├── encoder.py       # Shared encoder (pretrained)
│   ├── decoders.py      # Task-specific decoders
│   └── multitask.py     # Combined multi-task model
├── pretext_tasks/
│   ├── edge_prediction.py
│   ├── colorization.py
│   ├── masked_autoencoder.py
│   └── jigsaw.py
├── data/
│   └── dataset.py       # Dataset with on-the-fly label generation
├── train_pretrain.py    # Pretraining script
├── train_finetune.py    # Fine-tuning script
├── config.yaml          # Configuration
└── README.md            # This file
```

## Pretext Tasks

### Edge Prediction
- **Input**: RGB battlemap
- **Target**: Canny edge detection (auto-generated)
- **Learns**: Boundary detection across all art styles

Inspired by [auto-wall](https://github.com/ThreeHats/auto-wall) which uses Canny edges for wall detection.

### Colorization
- **Input**: Grayscale battlemap
- **Target**: Original RGB colors
- **Learns**: Material/texture differences (walls often differ from floors in color)

### Masked Autoencoding (MAE)
- **Input**: Image with 75% of patches masked
- **Target**: Reconstruct original image
- **Learns**: Spatial structure and coherence

### Jigsaw Puzzle
- **Input**: Image with patches shuffled
- **Target**: Classify which permutation was used
- **Learns**: Spatial relationships between map elements

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
loss_weights:
  edge: 0.3      # Weight for edge prediction
  color: 0.2     # Weight for colorization
  mae: 0.4       # Weight for masked autoencoding
  jigsaw: 0.1    # Weight for jigsaw puzzle
```

## Tips

### Data Collection
- More data = better generalization
- Aim for 5,000+ images for good results
- Include diverse styles: hand-drawn, digital, 3D, different color palettes

### Pretraining
- Start with default loss weights, adjust if one task dominates
- Use `--amp` for faster training with mixed precision
- Monitor individual task losses to ensure all are learning

### Fine-tuning
- Use lower learning rate for encoder (1e-5) than decoder (1e-4)
- Consider freezing encoder initially if limited labeled data
- Gradually unfreeze encoder layers for best results

## Integration with Existing Model

The pretrained encoder is compatible with `AttentionASPPUNet` from the Watabou model:

```python
from models.encoder import SharedEncoder
from train_finetune import create_segmentation_model

# Load pretrained encoder into segmentation model
model = create_segmentation_model(
    pretrained_path='./pretrained_models/encoder_only.pth',
    num_classes=6,  # wall, floor, terrain, door, window, secret_door
    freeze_encoder=False
)
```

## References

- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
- Multi-task: "Multi-Task Self-Supervised Visual Learning" (Doersch & Zisserman, 2017)
- auto-wall: https://github.com/ThreeHats/auto-wall (Canny edge approach)
