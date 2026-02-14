# Combined Watabou + Foundry Training Plan

## Overview

Integrate 106 Watabou dungeon masks (edge + recessed variants) with ~38 Foundry map masks for a combined training run that balances domain exposure.

---

## 1. The Data Balance Problem

| Source | Maps | Effective Samples | Grid Size |
|--------|------|-------------------|-----------|
| Foundry | ~38 | 38 | 140-200px (varies) |
| Watabou | 106 | 212 (edge + recessed) | 70px |

Without balancing, the model sees ~5.5x more Watabou tiles than Foundry tiles per epoch, risking overfitting to Watabou's clean, procedural style at the expense of learning from the more diverse (and more representative of real-world maps) Foundry data.

### Why This Matters

- Foundry maps have **varied art styles** (hand-drawn, digital, detailed textures) - closer to what the model will encounter in production
- Watabou maps are **procedurally uniform** (same renderer, same line style, same color palette) - good for learning wall structure but poor for style diversity
- The model should treat Foundry maps as the "high-value" data and Watabou maps as supplementary structural training

---

## 2. Strategy: Epoch-Level Source Balancing with Variant Randomization

### Core Idea

Each epoch:
1. Include **all** Foundry maps (they're the scarce, high-value resource)
2. **Subsample** Watabou maps to roughly match the Foundry count
3. For each selected Watabou map, **randomly choose** between edge and recessed mask variants

### Why This Approach

- **Over many epochs**, the model sees every Watabou map multiple times in both variants
- **Each epoch** has roughly balanced source representation
- **Edge/recessed randomization** acts as a free augmentation - the model learns that both wall placement styles are valid representations of the same map
- **Simple to implement** - no weighted samplers or complex batch construction needed

### Sampling Parameters

```
foundry_maps_per_epoch: all (~38)
watabou_maps_per_epoch: ~38 (randomly sampled from 106)
watabou_include_probability: 0.36 (38/106, tunable)
tiles_per_image: 4 (same as current)
variant_selection: random per-map per-epoch (50/50 edge vs recessed)
```

**Effective per-epoch:**
- ~38 Foundry maps x 4 tiles = ~152 Foundry tiles
- ~38 Watabou maps x 4 tiles = ~152 Watabou tiles
- Total: ~304 tiles/epoch (vs current ~152 from Foundry only)

**Over 100 epochs:**
- Each Watabou map seen ~36 times (some edge, some recessed)
- Each Foundry map seen 100 times (as before)
- The Foundry:Watabou ratio stays ~50:50 per epoch

### Tunable Ratio

The `watabou_include_probability` parameter controls the balance. Reasonable range:

| Probability | Watabou/epoch | Ratio (F:W) | When to use |
|-------------|---------------|-------------|-------------|
| 0.20 | ~21 | 65:35 | If Watabou is hurting Foundry performance |
| 0.36 | ~38 | 50:50 | Default balanced start |
| 0.50 | ~53 | 42:58 | If model needs more structural training |
| 1.00 | 106 | 26:74 | Only if adding many more Foundry maps later |

Start with 0.36 (balanced) and adjust based on validation metrics.

---

## 3. Implementation Plan

### 3.1 Modify `WallSegmentationDataset` to Support Multiple Sources

The current dataset takes a single `image_dir` and `mask_dir`. Extend it to accept a list of data sources, each with its own configuration.

**New dataclass for source config:**

```python
@dataclass
class DataSource:
    name: str                    # e.g., "foundry", "watabou"
    image_dir: str               # Path to source images
    mask_dir: str                # Path to masks (primary)
    mask_dir_alt: str = None     # Path to alt masks (for variant randomization)
    mask_suffix: str = "_mask_lines"  # Suffix before .png in mask filenames
    grid_size: int = 140         # Default grid size for this source
    include_probability: float = 1.0  # Per-epoch inclusion probability
    metadata_file: str = None    # Optional per-image metadata
```

**Source definitions for this run:**

```python
sources = [
    DataSource(
        name="foundry",
        image_dir="data/foundry_to_mask/Map_Images",
        mask_dir="data/foundry_to_mask/line_masks",
        mask_suffix="_mask_lines",
        grid_size=140,  # Overridden per-map by metadata
        include_probability=1.0,  # Always include all
        metadata_file="data/foundry_to_mask/Map_Images/name_mapping.json",
    ),
    DataSource(
        name="watabou",
        image_dir="data/watabou_to_mask/watabou_images",
        mask_dir="data/watabou_to_mask/watabou_edge_mask",
        mask_dir_alt="data/watabou_to_mask/watabou_recessed_mask",
        mask_suffix="",  # Watabou masks match image name exactly
        grid_size=70,
        include_probability=0.36,
    ),
]
```

### 3.2 Dataset Behavior Changes

**Sample discovery** (`_find_samples`):
- Iterate over each source, finding image-mask pairs using source-specific logic
- Tag each sample with its source name, grid size, and alt mask path (if any)
- For Watabou: direct filename match (image `X.png` -> mask `X.png`)
- For Foundry: existing `name_mapping.json` + fuzzy matching logic (unchanged)

**Epoch resampling** (`_resample_for_epoch` - new method):
- Called at the start of each epoch (or via a `resample()` method the training loop calls)
- For each source:
  - If `include_probability < 1.0`: randomly subsample maps with that probability
  - If `mask_dir_alt` exists: randomly choose primary or alt mask for each selected map
- Store the active sample list for this epoch
- `__len__` returns `len(active_samples) * tiles_per_image`

**Tile extraction** (`__getitem__`):
- Uses the active sample list (not the full sample list)
- Each sample already has its mask path set (primary or alt) for this epoch
- Grid size comes from the sample's source config
- Everything else (tiling, augmentation) stays the same

### 3.3 Training Loop Changes (`train.py`)

Minimal changes needed:

```python
# In Trainer.__init__:
# Replace single image_dir/mask_dir with sources list from config

# In Trainer.train():
for epoch in range(self.config.epochs):
    # Resample watabou maps for this epoch
    self.train_dataset.resample_for_epoch()

    # ... rest of training loop unchanged ...
```

### 3.4 Config Changes (`config.py`)

Add new fields to `Config`:

```python
# Multi-source data
data_sources: Optional[List[dict]] = None  # List of DataSource configs
# If None, falls back to single image_dir/mask_dir (backwards compatible)

# Watabou-specific
watabou_include_probability: float = 0.36
```

Add CLI args to `train.py`:

```
--watabou-dir        Path to watabou data (shortcut for adding watabou source)
--watabou-prob       Watabou inclusion probability (default: 0.36)
```

### 3.5 Class Compatibility Check

Both sources use the same class encoding:

| Class | Value | Foundry | Watabou |
|-------|-------|---------|---------|
| Background | 0 | Yes | Yes |
| Wall | 50 | Yes | Yes |
| Terrain | 100 | Yes | **No** (none in Watabou) |
| Door | 150 | Yes | Yes |

This is fine - Watabou maps simply never have terrain pixels. The model will learn terrain exclusively from Foundry maps, which is expected since Watabou dungeons don't have terrain objects.

If training with `--merge-terrain` (3 classes), there's no issue at all since terrain merges into wall.

---

## 4. File Changes Summary

| File | Change |
|------|--------|
| `primary_model_training/dataset.py` | Add multi-source support, epoch resampling, variant randomization |
| `primary_model_training/config.py` | Add `data_sources`, `watabou_include_probability` fields |
| `primary_model_training/train.py` | Add `--watabou-dir`/`--watabou-prob` CLI args, call `resample_for_epoch()` |

### Backwards Compatibility

The existing single-source workflow (`--image-dir` / `--mask-dir`) continues to work unchanged. Multi-source is only activated when `--watabou-dir` is provided or `data_sources` is configured.

---

## 5. Training Run Plan

### Run Command

```bash
python -m primary_model_training.train \
    --image-dir data/foundry_to_mask/Map_Images \
    --mask-dir data/foundry_to_mask/line_masks \
    --watabou-dir data/watabou_to_mask \
    --watabou-prob 0.36 \
    --merge-terrain \
    --epochs 100 \
    --batch-size 8
```

### What to Monitor

1. **Per-source validation metrics** - Track IoU/accuracy separately for Foundry vs Watabou tiles to ensure neither source dominates or degrades
2. **Wall class performance** - Should improve with more structural examples from Watabou
3. **Door class performance** - Watabou doors are simple rectangles; watch for regression on Foundry's more varied doors
4. **Overall convergence** - With 2x the data per epoch, may need to adjust learning rate or epoch count

### Potential Adjustments

- If Foundry IoU drops: reduce `watabou_prob` to 0.20
- If model converges slowly: increase epochs to 150
- If Watabou is too easy (loss near zero early): consider adding more aggressive augmentation specifically for Watabou tiles (e.g., stronger color jitter since Watabou maps are monochrome)

---

## 6. Future Extensions

This multi-source framework naturally supports:
- Adding more data sources later (e.g., Dungeondraft exports, hand-labeled maps)
- Per-source augmentation profiles (e.g., heavier color jitter for Watabou since it's visually monotone)
- Source-aware validation splits
- Curriculum learning (start with one source, gradually introduce others)

---

*Document created: 2026-02-13*
*Project: InstaWall - Combined Training Pipeline*
