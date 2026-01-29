# Self-Supervised Pretraining for Battlemap Wall Detection

## Goal
Train a model that generalizes across different battlemap art styles, not just Watabou-style maps.

## Problem
Current model (trained on ~200 Watabou maps) overRenfits to that specific visual style:
- Specific line weights and colors
- Consistent wall thickness
- Similar textures and shading

When shown "in-the-wild" maps with different styles, it fails to detect walls accurately.

## Solution: Multi-Task Self-Supervised Pretraining

### Core Idea
1. Collect large dataset of **unlabeled** battlemaps (easy to scrape)
2. Pretrain encoder on multiple "pretext tasks" that teach general visual features
3. Fine-tune on smaller labeled dataset for wall segmentation

### Architecture

```
                         ┌─→ [Edge Decoder] ──────→ Edge Loss
                         │
[Battlemap] → [Shared UNet Encoder] ─┼─→ [Colorization Decoder] ─→ Color Loss
                         │
                         ├─→ [Reconstruction Decoder] → MAE Loss
                         │
                         └─→ [Jigsaw MLP Head] ────→ Jigsaw Loss

Total Loss = w_edge * L_edge + w_color * L_color + w_mae * L_mae + w_jigsaw * L_jigsaw
```

### Pretext Tasks

#### 1. Edge Prediction
- **Input**: Battlemap image
- **Target**: Canny/Sobel edge detection (auto-generated)
- **What it learns**: General boundary/edge detection across all styles
- **Loss**: Binary cross-entropy or dice loss

```python
def generate_edge_labels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges / 255.0
```

#### 2. Colorization
- **Input**: Grayscale battlemap
- **Target**: Original color image
- **What it learns**: Material differences (walls vs floors often have different colors/textures)
- **Loss**: L1 or perceptual loss

```python
def generate_color_labels(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_3ch = np.stack([gray, gray, gray], axis=-1)
    return gray_3ch, image  # input, target
```

#### 3. Masked Autoencoding (MAE)
- **Input**: Image with 75% of patches masked
- **Target**: Reconstruct original image
- **What it learns**: Spatial coherence, structure completion
- **Loss**: MSE on masked regions only

```python
def generate_mae_input(image, patch_size=16, mask_ratio=0.75):
    h, w = image.shape[:2]
    n_patches_h, n_patches_w = h // patch_size, w // patch_size
    n_patches = n_patches_h * n_patches_w
    n_masked = int(n_patches * mask_ratio)

    mask_indices = np.random.choice(n_patches, n_masked, replace=False)
    mask = np.ones((n_patches_h, n_patches_w), dtype=bool)
    mask.flat[mask_indices] = False

    # Apply mask to image
    masked_image = image.copy()
    for idx in mask_indices:
        i, j = idx // n_patches_w, idx % n_patches_w
        masked_image[i*patch_size:(i+1)*patch_size,
                     j*patch_size:(j+1)*patch_size] = 0

    return masked_image, image, mask
```

#### 4. Jigsaw Puzzle
- **Input**: Image split into grid, patches shuffled
- **Target**: Predict correct permutation
- **What it learns**: Spatial relationships, room structure
- **Loss**: Cross-entropy over permutation classes

```python
def generate_jigsaw_input(image, grid_size=3):
    # Split into 3x3 grid
    patches = split_into_grid(image, grid_size)

    # Define a set of permutations (e.g., 100 fixed permutations)
    perm_idx = np.random.randint(0, len(PERMUTATIONS))
    perm = PERMUTATIONS[perm_idx]

    shuffled_patches = [patches[i] for i in perm]
    shuffled_image = reassemble_grid(shuffled_patches, grid_size)

    return shuffled_image, perm_idx
```

### Implementation Plan

#### Phase 1: Data Collection
```
battlemaps_unlabeled/
├── reddit_battlemaps/      # Scrape from r/battlemaps, r/dndmaps
├── pinterest/              # Battlemap collections
├── google_images/          # "fantasy battlemap" search
├── patreon_free/           # Free tier maps from creators
└── procedural/             # Generated maps (Watabou, Donjon, etc.)

Target: 5,000-10,000 unlabeled images
```

#### Phase 2: Pretraining
```python
# Hyperparameters
PRETRAIN_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

# Loss weights (tune these)
LOSS_WEIGHTS = {
    'edge': 0.3,
    'color': 0.2,
    'mae': 0.4,
    'jigsaw': 0.1
}

# Training
for epoch in range(PRETRAIN_EPOCHS):
    for batch in unlabeled_dataloader:
        # Generate all pretext labels
        edge_targets = generate_edge_labels(batch)
        gray_inputs, color_targets = generate_color_labels(batch)
        mae_inputs, mae_targets, mae_masks = generate_mae_input(batch)
        jigsaw_inputs, jigsaw_targets = generate_jigsaw_input(batch)

        # Forward pass through shared encoder + task heads
        edge_pred = model(batch, task='edge')
        color_pred = model(gray_inputs, task='color')
        mae_pred = model(mae_inputs, task='mae')
        jigsaw_pred = model(jigsaw_inputs, task='jigsaw')

        # Compute losses
        loss = (LOSS_WEIGHTS['edge'] * edge_loss(edge_pred, edge_targets) +
                LOSS_WEIGHTS['color'] * color_loss(color_pred, color_targets) +
                LOSS_WEIGHTS['mae'] * mae_loss(mae_pred, mae_targets, mae_masks) +
                LOSS_WEIGHTS['jigsaw'] * jigsaw_loss(jigsaw_pred, jigsaw_targets))

        loss.backward()
        optimizer.step()
```

#### Phase 3: Fine-tuning
```python
# Load pretrained encoder
pretrained = MultiTaskPretrainer.load('pretrained_encoder.pth')
encoder = pretrained.encoder

# Build segmentation model with pretrained encoder
model = UNetSegmentation(encoder=encoder, num_classes=3)

# Fine-tune on labeled Watabou data (+ any other labeled data)
# Use lower learning rate for encoder, higher for new decoder
optimizer = Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4}
])

for epoch in range(FINETUNE_EPOCHS):
    for images, masks in labeled_dataloader:
        pred = model(images)
        loss = dice_loss(pred, masks) + ce_loss(pred, masks)
        loss.backward()
        optimizer.step()
```

### Model Architecture

```python
class MultiTaskPretrainer(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()

        # Shared encoder (this is what we keep after pretraining)
        self.encoder = nn.ModuleList([
            ConvBlock(3, base_channels),           # 64
            DownBlock(base_channels, base_channels*2),    # 128
            DownBlock(base_channels*2, base_channels*4),  # 256
            DownBlock(base_channels*4, base_channels*8),  # 512
            DownBlock(base_channels*8, base_channels*16), # 1024
        ])

        # Task-specific decoders (discarded after pretraining)
        self.edge_decoder = UNetDecoder(base_channels*16, out_channels=1)
        self.color_decoder = UNetDecoder(base_channels*16, out_channels=3)
        self.mae_decoder = UNetDecoder(base_channels*16, out_channels=3)

        # Jigsaw uses global features
        self.jigsaw_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels*16, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_PERMUTATIONS)
        )

    def encode(self, x):
        features = []
        for block in self.encoder:
            x = block(x)
            features.append(x)
        return features

    def forward(self, x, task):
        features = self.encode(x)

        if task == 'edge':
            return self.edge_decoder(features)
        elif task == 'color':
            return self.color_decoder(features)
        elif task == 'mae':
            return self.mae_decoder(features)
        elif task == 'jigsaw':
            return self.jigsaw_head(features[-1])
        else:
            raise ValueError(f"Unknown task: {task}")
```

### Expected Benefits

| Aspect | Without Pretraining | With Pretraining |
|--------|---------------------|------------------|
| Training data needed | Hundreds of labeled maps | Thousands unlabeled + hundreds labeled |
| Generalization | Poor (overfits to style) | Good (learns general features) |
| Edge detection | Style-specific | Style-agnostic |
| New art styles | Fails | Adapts with few examples |

### Potential Improvements

1. **Contrastive learning**: Add SimCLR-style contrastive loss to learn that augmented versions of the same map should have similar features

2. **Style augmentation**: Apply neural style transfer during pretraining to see the same map in many styles

3. **Curriculum learning**: Start with easy pretext tasks (edges), gradually add harder ones (jigsaw)

4. **Progressive resolution**: Pretrain at low resolution first, then increase

5. **Domain-specific augmentations**:
   - Grid overlay/removal
   - Lighting changes
   - Paper texture overlay
   - Color palette swaps

### Files to Create

```
self_supervised_pretraining/
├── data/
│   ├── scraper.py           # Download battlemaps from various sources
│   └── dataset.py           # PyTorch dataset with pretext task generation
├── models/
│   ├── encoder.py           # Shared encoder architecture
│   ├── decoders.py          # Task-specific decoders
│   └── multitask.py         # Combined multi-task model
├── pretext_tasks/
│   ├── edge_prediction.py
│   ├── colorization.py
│   ├── masked_autoencoder.py
│   └── jigsaw.py
├── train_pretrain.py        # Pretraining script
├── train_finetune.py        # Fine-tuning script
└── config.yaml              # Hyperparameters
```

### References

- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
- SimCLR: "A Simple Framework for Contrastive Learning" (Chen et al., 2020)
- Jigsaw: "Unsupervised Visual Representation Learning by Context Prediction" (Doersch et al., 2015)
- Multi-task: "Multi-Task Self-Supervised Visual Learning" (Doersch & Zisserman, 2017)
