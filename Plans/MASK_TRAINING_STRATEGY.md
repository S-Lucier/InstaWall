# Mask Training Strategy for Universal Battlemap Model

## Overview

This document outlines the strategy for training a model to predict Foundry VTT wall segments from battlemap images.

---

## 1. Mask Type Decision: Line Segments

### Why Line Segments (Method A) Over Filled Regions (Methods B/C)

**The Problem:**
Different maps use different walling styles in Foundry:
- **Edge-traced**: Walls placed along the visual edge of thick dungeon walls (traditional dungeons)
- **Center-traced**: Walls placed down the middle of thin walls (modern maps, buildings)

For center-traced maps, flood-fill approaches (Methods B/C) incorrectly include wall texture as "floor" since the visual wall extends on both sides of the segment line.

**The Solution:**
Use line segment masks (Method A) universally. The model learns to predict segment positions directly, which:
- Works for any walling style
- Outputs what Foundry actually needs (wall segment data)
- Avoids flood-fill edge cases

### Wall Placement Heuristic

The model should learn to place walls appropriately based on wall thickness:

| Wall Thickness (relative to grid) | Placement Strategy |
|-----------------------------------|-------------------|
| < 1 grid cell | Single center line |
| >= 1 grid cell | Edge trace both sides |

This is learned implicitly from training data if labeled consistently.

### Edge Placement Tolerance

Walls do NOT need pixel-perfect edge alignment. Slightly inset from visual edge (~5-10%) is acceptable and arguably preferable:
- Players can see wall texture
- Vision/movement blocking still works correctly
- More robust (less sensitive to exact edge detection)

What matters:
- Blocks vision/movement correctly
- Doesn't cut into playable floor space

---

## 2. Scale and Grid Context

### The Problem

Maps come in various sizes and grid scales. A wall that appears "thin" might be:
- Actually thin (should be center-traced)
- A thick wall on a large map that's been shrunk (should be edge-traced)

The model needs context about the grid scale to make correct decisions.

### Solution: Tiled Processing at Consistent Scale

Process maps in tiles at a consistent scale relative to grid size:

1. User provides grid size (pixels per cell) for the map
2. Tiles are extracted at a fixed grid-cell size (e.g., 8x8 grid cells per tile)
3. Each tile is resized to the model's input resolution (e.g., 512x512)
4. Model sees walls at consistent relative thickness regardless of source map size
5. Predictions are stitched back together

**Benefits:**
- Consistent wall thickness representation
- Model learns one scale, generalizes to all map sizes
- Standard approach in segmentation tasks

**Tile Size Considerations:**
- Too small: Not enough context for wall decisions
- Too large: Loses detail, harder to train
- Recommended: 6-10 grid cells per tile (experiment to find optimal)

### Alternative: Grid Size as Model Input

Could condition the model with grid size metadata (extra channel or embedding), but tiling is:
- More straightforward to implement
- Battle-tested in segmentation
- Naturally handles varying map sizes

---

## 3. Tile Boundary Alignment

### The Problem

When processing tiles independently, wall predictions at tile boundaries may not align when stitched together, causing:
- Gaps in walls
- Misaligned segments
- Discontinuities

### Mitigation Strategies

#### Strategy 1: Overlapping Tiles with Center-Crop (Recommended)

Process tiles with overlap, but only keep predictions from the center region where the model has maximum context.

```
Tile Layout (1D example, 50% overlap):

Tile 1: [====KEEP====][discard]
Tile 2:        [discard][====KEEP====][discard]
Tile 3:                        [discard][====KEEP====]

Final:  [====KEEP====][====KEEP====][====KEEP====]
```

**Implementation:**
- Extract tiles with 50% overlap in both dimensions
- After prediction, crop to center 50% of each tile
- Stitch center crops together

**Benefits:**
- Each pixel predicted from tile where it's most centered
- Model has full context on all sides for kept region
- Simple, proven approach

#### Strategy 2: Global Context Conditioning

Provide global map awareness to each tile:

1. Create low-resolution encoding of full map (e.g., 64x64)
2. Pass as additional input channels to each tile
3. Model sees both local detail and global structure

**Benefits:**
- Tiles aware of broader context
- Can help with consistent style across map

**Drawbacks:**
- More complex architecture
- May not be necessary if overlap approach works

#### Strategy 3: Post-Processing Connection

After tiling, run wall connection logic:

1. Detect wall segments ending near tile boundaries
2. If two segments point toward each other across boundary, connect them
3. Snap endpoints to create continuous walls

**Implementation:**
```python
def connect_across_boundaries(segments, boundary_threshold=10):
    # Find segments ending within threshold of tile boundary
    # Match pairs that are collinear and pointing toward each other
    # Extend/connect matched pairs
```

**Benefits:**
- Fixes remaining gaps after overlap approach
- Can enforce connectivity constraints

#### Strategy 4: Boundary-Aware Training Augmentation

Train with randomly positioned tile boundaries:

1. During training, extract tiles at random positions (not grid-aligned)
2. Model learns to predict walls robustly regardless of crop position
3. Optionally: loss term penalizing discontinuities at boundaries

**Benefits:**
- Model learns not to "hedge" at edges
- More robust predictions overall

### Recommended Approach

1. **Primary:** Overlapping tiles with center-crop (Strategy 1)
2. **Secondary:** Post-processing connection (Strategy 3) if gaps remain
3. **Optional:** Boundary-aware augmentation during training (Strategy 4)

---

## 4. Training Data

### Priority Sources

1. **Naturally edge-traced maps** (dungeons with thick walls)
   - Crypt of Everflame style
   - Traditional dungeon crawl maps
   - These work correctly with current tooling

2. **Manually corrected maps** (if needed)
   - Re-wall thin-wall maps to edge-trace style
   - Only if model struggles to generalize

### Data Pipeline

```
Foundry JSON + Image
        |
        v
foundry_to_mask.py (Method A - lines only)
        |
        v
data/foundry_to_mask/line_masks/
        |
        v
Tile extraction (grid-aligned, with overlap)
        |
        v
Training dataset
```

### Labeling Consistency

For training data quality:
- Walls should be ~5% of grid size in thickness
- Consistent edge offset across maps
- Clear distinction between wall/terrain/door classes

---

## 5. Class Definitions

Using Method A (lines-only) class values:

| Class | Value | Description |
|-------|-------|-------------|
| Background | 0 | Unmarked pixels |
| Wall | 50 | Standard walls, secret doors, and windows |
| Terrain | 100 | Partial blockers (pillars, rubble) |
| Door | 150 | Normal doors only |

**Conversions:**
- **Secret doors** → Wall (visually indistinguishable from walls)
- **Windows** → Wall (model can't distinguish from wall gaps)
  - Normal window: `light=0, sight=0, move=20`
  - Proximity window: `light=30, sight=30, move=20`

**Note:** Secret doors still count as doors for room connectivity checks in hybrid mode (rooms behind secret doors are playable space, not enclosed wall).

Training code should normalize: `class_index = pixel_value // 50`

---

## 6. Model Architecture Considerations

### Input
- RGB image tile (e.g., 512x512)
- Optional: grid size embedding or global context channels

### Output
- Single-channel class mask (same resolution as input)
- Or: multi-head output for wall position + wall type

### Architecture Options
- UNet (proven for segmentation)
- SegFormer (transformer-based, good for fine details)
- Custom encoder-decoder

### Loss Function
- CrossEntropyLoss for class prediction
- Optional: line-aware loss that penalizes broken segments

---

## 7. Inference Pipeline

```
User Input: Battlemap image + grid size (pixels per cell)
                    |
                    v
            Calculate tile grid
            (e.g., 8x8 cells per tile, 50% overlap)
                    |
                    v
            Extract overlapping tiles
                    |
                    v
            Model prediction on each tile
                    |
                    v
            Center-crop each prediction
                    |
                    v
            Stitch into full mask
                    |
                    v
            Post-process: connect segments at boundaries
                    |
                    v
            Convert mask to Foundry wall JSON
                    |
                    v
Output: Foundry-compatible wall data
```

---

## 8. Open Questions

1. **Optimal tile size**: 6x6? 8x8? 10x10 grid cells? Needs experimentation.

2. **Overlap amount**: 50% is standard, but 25% might suffice with good training.

3. **Terrain vs Wall distinction**: Can model reliably distinguish these from visual appearance alone?

4. **Secret doors**: These are designed to look like walls - may need to omit from training or accept lower accuracy.

5. **Multi-floor maps**: Currently ignored. Future consideration.

---

*Document created: 2026-02-05*
*Project: InstaWall - Universal Battlemap Model*
