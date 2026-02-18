# User-Guided Texture Refinement

## Problem
Different battlemaps use vastly different art styles and wall textures. The model may miss walls that have unfamiliar textures or low contrast against the background. If the user could highlight a known wall region, the system could key off that texture to improve predictions on the rest of the map.

## Approaches

### Option 1: Post-Hoc Logit Boosting (No Retraining)

**Complexity**: Low
**Retraining Required**: No

After the model runs inference, the user highlights a region they know is wall. The system:

1. Extracts a texture/color profile from the highlighted region (color histograms, LBP descriptors, or simple patch statistics)
2. For each pixel in the output, computes a similarity score between that pixel's local texture and the reference
3. For pixels where the model is uncertain (softmax confidence below a threshold), boosts the "wall" logit proportionally to texture similarity
4. Re-applies argmax to get the refined prediction

**Pros**: Simple to implement, no model changes, works with any checkpoint
**Cons**: Crude texture matching, doesn't leverage the model's learned feature representations

### Option 2: Feature Prototype at Inference (No Retraining) -- Recommended

**Complexity**: Medium
**Retraining Required**: No

Uses the SegFormer encoder's own feature representations, which already cluster similar textures in feature space.

1. Run the image through the encoder as normal
2. User highlights a wall region
3. Extract the mean feature vector from the highlighted region at one or more encoder stages -- this becomes the "wall prototype"
4. Compute cosine similarity between every spatial location's feature vector and the wall prototype
5. Use the similarity map to adjust the decoder's wall logit before the final argmax:
   `adjusted_wall_logit = original_wall_logit + alpha * similarity`
   where `alpha` controls the strength of the user guidance
6. Apply argmax on the adjusted logits

**Implementation sketch:**
- Modify `inference.py` to optionally accept a user hint mask
- Hook into the encoder to extract intermediate feature maps (SegFormer stages 1-4)
- Compute prototype as mean feature vector over highlighted pixels
- Cosine similarity at each spatial location -> blend into logits
- `alpha` could be a slider in the UI (0 = ignore hint, 1+ = strong guidance)

**Pros**: Leverages the model's learned representations, no retraining, principled similarity in feature space
**Cons**: Requires access to intermediate encoder features (minor refactor), alpha needs tuning

### Option 3: Extra Input Channel (Requires Retraining)

**Complexity**: High
**Retraining Required**: Yes

Add a 4th input channel to the model -- a binary "user hint" mask indicating known wall pixels.

1. Modify the model's first layer to accept 4 channels instead of 3
2. During training, randomly generate hint masks:
   - With probability p (e.g., 0.5), provide no hint (all zeros) so the model still works standalone
   - Otherwise, randomly sample a connected patch of ground-truth wall pixels as the hint
   - Vary hint size (small patch to large region) for robustness
3. The model learns to propagate wall predictions from the hinted region to texturally similar areas
4. At inference, user's highlighted region becomes the hint channel

**Pros**: Model learns to use hints natively, strongest approach, works end-to-end
**Cons**: Requires retraining from scratch, more complex data pipeline, SegFormer pretrained weights won't cover the 4th channel (would need to zero-init or learn from scratch for that channel)

## Recommendation

**Start with Option 2** (Feature Prototype). It requires no retraining, leverages what the model already knows, and can be implemented entirely in the inference pipeline. If it works well, it may be all that's needed. If it's not strong enough, Option 3 gives the most room for improvement but at a significant training cost.

## UI Considerations

Regardless of approach, the user interaction is the same:
- User loads a battlemap and runs initial inference
- User sees the prediction overlay
- User draws/highlights a region they know is wall (brush tool, lasso, or rectangle select)
- System refines the prediction using the highlighted texture as guidance
- User can adjust strength (alpha slider) and re-highlight if needed
- Could support multiple highlights for different wall textures on the same map

## Dependencies

- A working inference pipeline (already exists in `inference.py`)
- For Option 2: ability to extract intermediate SegFormer encoder features
- For Option 3: model architecture changes + retraining
- UI for highlight input (could start with a simple script that takes a mask image)
