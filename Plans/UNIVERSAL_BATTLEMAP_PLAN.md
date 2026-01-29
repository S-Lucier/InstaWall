# Universal TTRPG Battlemap Segmentation Model

## Project Planning Document

---

## 1. Executive Summary

**Goal:** Create a segmentation model that can identify playable space, walls, and doors on *any* TTRPG battlemap, regardless of art style, perspective, or source.

**Current State:** 3-class AttentionASPPUNet trained on Watabou dungeon generator output. Achieves 0.92+ Dice on doors, 0.99+ on walls/floors for this specific domain.

**Challenge:** Watabou maps share a consistent visual style (top-down, clean lines, uniform colors). Real-world battlemaps have enormous variation in:
- Art styles (hand-drawn, digital, 3D rendered, photo-realistic)
- Perspectives (true top-down, isometric, oblique)
- Visual complexity (minimalist to highly detailed)
- Map types (dungeons, outdoor, urban, caves, ships, etc.)

---

## 2. Problem Analysis

### 2.1 Why This Is Hard

| Aspect | Watabou Maps | General Battlemaps |
|--------|--------------|-------------------|
| Wall appearance | Black lines/fills | Stone, wood, hedges, cliffs, force fields, shadows, etc. |
| Floor appearance | White/gray fill | Tiles, dirt, grass, water, lava, carpet, stone, etc. |
| Door appearance | Gray rectangles | Wooden doors, archways, portcullises, secret doors, gaps, stairs |
| Perspective | Strict top-down | Top-down, isometric, mixed |
| Grid | 70px uniform | Variable or none |
| Lighting | None | Shadows, highlights, ambient occlusion |
| Props/furniture | None | Tables, beds, chests (not walls but occlude space) |

### 2.2 Semantic Ambiguity

The core challenge: "playable space" is a **game concept**, not a visual one.

- A table is visually distinct from floor, but is it playable? (Depends on game rules)
- Water might be playable (swim), impassable (deep), or terrain (shallow)
- Elevation changes may or may not block movement
- Secret doors are intentionally hidden visually

**Key Insight:** We must define consistent labeling conventions and accept some ambiguity.

### 2.3 Proposed Class Definitions

**6 classes:**

| Class | Definition | Examples |
|-------|------------|----------|
| **Wall** | Impassable barriers (blocks movement + vision) | Walls, cliffs, columns, pillars, ship hulls, dense hedges |
| **Floor** | Clearly playable horizontal space | Tiles, ground, decks, paths, open terrain |
| **Terrain** | Solid obstacles you can't walk/jump over, for Foundry terrain walls (blocks movement; vision passes first layer but not second, so you see the object but not past it) | Boulders, rubble piles, large rocks, stalagmites |
| **Door** | Passable openings in walls | Doors, archways, gates, portcullises, hatches |
| **Window** | Visual openings that block movement | Windows, arrow slits, grates, bars, murder holes |
| **Secret Door** | Hidden passable openings in walls | Hidden doors, concealed passages, bookcase doors, fake walls |

*Notes:*
- *Terrain class converts to same mask format as walls but gets tagged as terrain walls in Foundry output.*
- *Furniture/props that can't be represented as any wall type are ignored (left as floor).*
- *Secret doors are inherently difficult to detect (they're designed to look like walls). If model performance on secret doors is unreliable, merge with walls class and skip detecting them entirely.*

---

## 3. Data Strategy

### 3.1 The Data Problem

This is the **critical bottleneck**. Options:

#### Option A: Manual Annotation (High Quality, Low Scale)
- Commission/crowdsource labeling of real battlemaps
- 500-1000 diverse maps minimum for baseline
- Expensive and time-consuming
- Best quality ground truth

#### Option B: Synthetic Generation (High Scale, Domain Gap)
- Procedurally generate diverse battlemaps with known masks
- Use multiple generators (Watabou, Dungeondraft exports, etc.)
- Risk: model overfits to synthetic patterns

#### Option C: Semi-Supervised / Self-Training
- Train on small labeled set
- Generate pseudo-labels on large unlabeled corpus
- Iterate with confidence thresholding

#### Option D: Hybrid Approach (Recommended)
1. **Foundation:** Synthetic data from multiple generators (1000s of maps)
2. **Anchoring:** 200-500 manually labeled real maps across styles
3. **Expansion:** Semi-supervised learning on scraped battlemaps

### 3.2 Data Sources

**Synthetic/Procedural:**
- Watabou (current) - dungeons, villages, cities
- Dungeondraft export parsing (if JSON available)
- Custom procedural generators
- Game engine exports (Unity, Unreal with ground truth)

**Real Battlemaps (need licensing consideration):**
- Reddit r/battlemaps, r/dndmaps (CC-licensed content)
- OpenGameArt.org
- Patreon map creators (with permission)
- Commissioned original content
- Public domain historical maps

**Annotation Tools:**
- Label Studio, CVAT, or custom tool
- Consider polygon annotation (not just pixel masks)
- Export to standard format (COCO, Pascal VOC)

### 3.3 Data Augmentation Strategy

Beyond current augmentations, add:

```
Style Transfer:
- Neural style transfer to simulate different art styles
- Color palette randomization
- Texture overlay augmentation

Geometric:
- Perspective warping (simulate non-top-down)
- Local elastic deformation
- Random cropping with context

Degradation:
- JPEG artifacts
- Resolution variation (train multi-scale)
- Partial occlusion (simulate props)

Domain Randomization:
- Random background textures for walls
- Random floor patterns
- Synthetic lighting/shadow overlay
```

---

## 4. Model Architecture

### 4.1 Current Architecture Review

AttentionASPPUNet (37.7M params) works well for Watabou because:
- Consistent input domain
- Clear semantic boundaries
- Uniform scale

For universal model, need:
- **Larger receptive field** (variable map scales)
- **Better feature extraction** (complex textures)
- **Multi-scale processing** (small doors to large rooms)

### 4.2 Architecture Options

#### Option 1: Scaled AttentionASPPUNet
- Increase depth (5-6 encoder levels)
- Wider feature channels [64, 128, 256, 512, 1024]
- Add more ASPP dilation rates
- ~80-100M parameters
- **Pro:** Incremental improvement, known behavior
- **Con:** May not generalize to very different styles

#### Option 2: Modern Segmentation Backbone
**Recommended: SegFormer or Mask2Former**

```
SegFormer (B3 or B5):
- Transformer-based encoder
- Hierarchical feature extraction
- Pre-trained on ImageNet/ADE20K
- Excellent generalization
- ~45-85M parameters

Mask2Former:
- State-of-the-art panoptic segmentation
- Instance-aware (can separate individual rooms)
- Heavier but more capable
- ~100M+ parameters
```

#### Option 3: Hybrid CNN-Transformer
- EfficientNet or ConvNeXt encoder
- Transformer decoder (SETR-style)
- Balance efficiency and global reasoning
- ~60-80M parameters

### 4.3 Recommended Architecture Evolution

```
Phase 1: Enhanced UNet (current approach extended)
- Deeper AttentionASPPUNet
- Add FPN-style multi-scale outputs
- ~50-60M parameters

Phase 2: Transformer Backbone
- SegFormer-B3 with custom decoder
- Pre-train on synthetic, fine-tune on real
- ~50M parameters

Phase 3: Specialized Model (if needed)
- Style-conditioned segmentation
- Separate heads per map type
- Ensemble approaches
```

---

## 5. Training Strategy

### 5.1 Pre-training Pipeline

```
Stage 1: Wall Detection Pre-training (YOUR EXISTING PLAN)
├── Binary segmentation: wall vs non-wall
├── Massive synthetic dataset (Watabou + others)
├── Goal: Learn "wall-ness" across styles
└── Freeze encoder, transfer to Stage 2

Stage 2: Multi-Domain Synthetic Training
├── 6-class segmentation (wall, floor, terrain, door, window, secret door)
├── Multiple synthetic generators
├── Domain randomization augmentation
└── Establish baseline multi-domain performance

Stage 3: Real Data Fine-tuning
├── Curated labeled real battlemaps
├── Lower learning rate (1e-5)
├── Careful validation on held-out real data
└── Prevent catastrophic forgetting of synthetic

Stage 4: Semi-Supervised Expansion
├── Pseudo-label confident predictions on unlabeled
├── Self-training loop
├── Active learning for ambiguous cases
└── Continuous improvement pipeline
```

### 5.2 Loss Function Considerations

```python
# Multi-component loss for robustness

loss = (
    0.4 * CrossEntropyLoss(class_weights) +  # Pixel classification
    0.3 * DiceLoss(per_class=True) +          # Region overlap
    0.2 * BoundaryLoss() +                     # Edge accuracy (critical for walls)
    0.1 * FocalLoss(gamma=2)                   # Hard example mining
)
```

**Boundary Loss** is crucial - walls are defined by their edges.

### 5.3 Class Imbalance Strategy

Expect significant imbalance across all 6 classes:
- Floors: ~50-70% of pixels (dominant class)
- Walls: ~20-40% of pixels
- Terrain: ~1-5% of pixels (map-dependent)
- Doors: <2% of pixels (rare, critical)
- Windows: <1% of pixels (very rare)
- Secret doors: <0.5% of pixels (rarest, hardest to detect)

Mitigation strategies:
- Inverse frequency class weighting (as in current model)
- Online hard example mining (OHEM) for rare classes
- Separate detection heads for doors/windows/secret doors with upweighting
- Focal loss to focus on hard-to-classify pixels
- Consider merging secret door → wall if secret door performance is poor (they look like walls)

---

## 6. Evaluation Framework

### 6.1 Metrics

**Primary:**
- Per-class Dice Score
- Mean IoU (Intersection over Union)
- Boundary F1 (edge accuracy)

**Secondary:**
- Pixel Accuracy (sanity check)
- Per-domain breakdown (track generalization)
- Door recall specifically (doors are safety-critical)

### 6.2 Evaluation Domains

Create held-out test sets for each:

```
Domain 1: Synthetic - Watabou (baseline)
Domain 2: Synthetic - Other generators
Domain 3: Real - Dungeon maps (stone, indoor)
Domain 4: Real - Outdoor/wilderness maps
Domain 5: Real - Urban/building maps
Domain 6: Real - Unusual (ships, caves, planar)
Domain 7: Real - Isometric/non-top-down
```

Track per-domain metrics to identify weaknesses.

### 6.3 Failure Mode Analysis

Document and analyze:
- False walls (shadows, dark floor patterns)
- Missed walls (light-colored, thin lines)
- Door/window confusion (both are openings, differ in passability)
- Terrain vs floor boundaries (soft edges)
- Text misclassified as walls (room labels, map titles, legends)
- Grid line interference

---

## 7. Practical Considerations

### 7.1 Input Resolution

**Problem:** Real battlemaps vary wildly in resolution and aspect ratio.

**Solution:**
```
Training: Multi-scale training
- 512x512 (current)
- 768x768
- 1024x1024
- Random crop from larger maps

Inference: Sliding window with overlap
- Process large maps in tiles
- Blend overlapping predictions
- Preserve fine details
```

### 7.2 Text Handling

**Problem:** Map labels, room names, compass roses, and legends contain dark pixels that may be misclassified as walls.

**Options:**

| Approach | Pros | Cons |
|----------|------|------|
| **Label as floor in training** | Simple, model learns to ignore | Requires diverse text examples in training data |
| **Pre-processing text detection** | Clean input to model | Extra pipeline step, OCR can miss stylized text |
| **Synthetic text augmentation** | Cheap to generate, forces model to generalize | May not match real map text styles |
| **Post-processing filter** | Catches obvious text (thin, isolated segments) | May miss thick/decorative text, may remove valid thin walls |
| **Separate text class** | Explicit handling | More classes = harder training, text is highly variable |

**Recommended approach:** Combination of:
1. Include maps with text in training data, labeled as floor
2. Synthetic text augmentation (random fonts, sizes, positions)
3. Post-processing sanity check (isolated thin segments with no structural connection)

### 7.3 Grid Handling

**Problem:** Most TTRPG battlemaps have visible grids (square or hex) that could be misclassified as walls.

**Why grids are easier than text:**
- Highly regular, repeating pattern (unlike arbitrary wall shapes)
- Uniform across entire map (walls form room boundaries)
- Consistent line thickness and spacing
- CNNs naturally learn to ignore regular textures

**Potential issues:**
| Issue | Risk | Notes |
|-------|------|-------|
| Grid lines detected as thin walls | Low | Regularity makes them distinguishable |
| Grid fragmenting wall detection | Medium | Lines crossing walls could cause breaks |
| Hex grids vs square grids | Low | Both are regular patterns |
| Thick/stylized grids | Medium | Some maps have decorative grids |

**Handling approaches:**

| Approach | Pros | Cons |
|----------|------|------|
| **Train with gridded maps** | Simple, model learns naturally | Need diverse grid styles in training |
| **Synthetic grid augmentation** | Cheap, forces generalization | May not match all real grid styles |
| **Pre-processing removal (FFT/Hough)** | Clean input | Risky - may remove grid-aligned walls |
| **Post-processing filter** | Remove regular thin segments | Could affect legitimate thin walls |

**Recommended approach:**
1. Include gridded maps in training data, label grid lines as floor
2. Synthetic grid augmentation (square, hex, various thicknesses/opacities)
3. No pre-processing removal (too risky for grid-aligned walls)

### 7.4 Grid Detection for Scale (Optional Enhancement)

Separately from ignoring grids, detecting grid spacing provides useful scale information:

```
Optional pre-processing:
1. Detect grid presence and spacing (FFT or Hough transform)
2. Calculate pixels-per-grid-cell
3. Use for post-processing wall snapping (align walls to grid)
4. Normalize output coordinates to grid units
```

This is independent of segmentation - the model ignores the grid visually, but we can still extract scale info.

### 7.5 Post-Processing Pipeline

Extend current mask_to_walls approach:

```
1. Raw model output (probabilistic)
2. CRF refinement (optional, for edge cleanup)
3. Morphological operations (close small gaps)
4. Contour extraction (wall boundaries)
5. Line simplification (Douglas-Peucker)
6. Geometric cleanup:
   - Snap near-horizontal/vertical
   - Merge close endpoints
   - Extend to intersections
7. Door segment extraction
8. Output: Wall/door line segments
```

### 7.6 Confidence and Uncertainty

For practical use, expose uncertainty:

```python
# Instead of argmax, provide:
{
    "prediction": class_id,
    "confidence": softmax_probability,
    "uncertainty_map": entropy_per_pixel,
    "needs_review_regions": low_confidence_contours
}
```

Allow users to manually fix low-confidence regions.

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Implement wall detection pre-training on Watabou
- [ ] Set up data pipeline for multiple synthetic sources
- [ ] Implement enhanced UNet with deeper architecture
- [ ] Establish evaluation framework and metrics dashboard
- [ ] Create annotation tool/workflow for real maps

### Phase 2: Multi-Source Synthetic (Weeks 5-8)
- [ ] Integrate 2-3 additional synthetic generators
- [ ] Implement domain randomization augmentation
- [ ] Train multi-domain synthetic model
- [ ] Evaluate cross-domain generalization
- [ ] Iterate on architecture if needed

### Phase 3: Real Data Integration (Weeks 9-12)
- [ ] Curate 200+ labeled real battlemaps
- [ ] Fine-tune on real data with careful validation
- [ ] Implement multi-scale inference pipeline
- [ ] Create per-domain evaluation reports
- [ ] Identify systematic failure modes

### Phase 4: Production Hardening (Weeks 13-16)
- [ ] Implement semi-supervised expansion
- [ ] Add uncertainty estimation
- [ ] Optimize inference speed (quantization, ONNX)
- [ ] Build API/tool integration
- [ ] User testing and feedback loop

### Phase 5: Advanced Features (Ongoing)
- [ ] Full 6-class model (wall, floor, terrain, door, window, secret door)
- [ ] Isometric/perspective support
- [ ] Style-conditioned inference
- [ ] Active learning pipeline
- [ ] Community contribution workflow

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Insufficient labeled data | High | High | Start synthetic, semi-supervised expansion |
| Domain gap (synthetic → real) | High | Medium | Domain randomization, real fine-tuning |
| Door detection failure | Medium | High | Separate door head, high recall target |
| Compute requirements | Medium | Medium | Start small, scale architecture gradually |
| Art style extremes fail | Medium | Medium | Per-domain evaluation, targeted data collection |
| Ambiguous ground truth | High | Medium | Clear labeling guidelines, multi-annotator |

---

## 10. Success Criteria

**Minimum Viable Product (MVP):**
- 85%+ mIoU on synthetic test set
- 75%+ mIoU on real dungeon maps
- 70%+ Door Dice on real maps
- Works on 512x512+ resolution
- <5 second inference on GPU

**Production Quality:**
- 90%+ mIoU on synthetic
- 85%+ mIoU on real (all domains)
- 85%+ Door Dice on real maps
- Confidence calibration
- Multi-scale inference

**Stretch Goals:**
- Isometric map support
- Instance segmentation (separate rooms)
- Real-time inference (<1 second)
- Style transfer to normalize inputs

---

## 11. Open Questions

1. **Annotation depth:** Full pixel masks vs. polygon boundaries vs. bounding boxes?
2. **Terrain object boundaries:** How to handle objects that partially overlap floors (e.g., table legs)?
3. **Window edge cases:** Arrow slits vs. decorative windows vs. open balconies?
4. **Licensing:** Can we legally use maps from Reddit/Patreon for training?
5. **User correction:** How to efficiently collect feedback to improve model?
6. **Grid handling:** Pre-process to detect/normalize grid, or learn grid-agnostic?
7. **Class hierarchy:** Should terrain objects be subdivided (furniture vs. natural vs. magical)?

---

## 12. Next Steps

1. **Immediate:** Implement wall detection pre-training (your existing plan)
2. **This week:** Survey additional synthetic data sources
3. **Next week:** Design annotation schema and start labeling pilot
4. **2 weeks:** Deeper architecture experiments on Watabou baseline

---

## Appendix A: Synthetic Data Sources

| Source | Type | Ground Truth | Availability |
|--------|------|--------------|--------------|
| Watabou Dungeon | Dungeon | JSON geometry | API available |
| Watabou Village | Outdoor | JSON geometry | API available |
| Dungeondraft | Various | VTT export JSON | Requires tool |
| Wonderdraft | Outdoor | Unknown | Investigate |
| Dungeon Alchemist | 3D Dungeon | Render with masks | Requires tool |
| Inkarnate | Various | Unknown | Investigate |
| Custom procedural | Various | Built-in | Implement |

## Appendix B: Reference Papers

- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
- Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation
- Domain Randomization for Sim-to-Real Transfer
- Self-Training with Noisy Student Improves ImageNet Classification

---

*Document created: 2026-01-29*
*Project: InstaWall Universal Battlemap Segmentation*
