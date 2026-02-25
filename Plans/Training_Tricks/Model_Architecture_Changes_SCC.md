# Model Architecture Changes — SCC Meetup Suggestions

*Date: 2026-02-25*

Six ideas from a friend at a meetup. Analysis and verdict on each below.

---

## 1. Furniture Removal Before Wall Model

**Idea:** Run a preprocessing model to detect and erase furniture/objects from the image before feeding it to the wall segmentation model.

**Analysis:**

The appeal is clear — furniture can occlude walls and confuse the model. But this is essentially building a second full ML problem:
- Need labelled furniture data (separate annotation effort)
- Need a detection/segmentation model (Mask R-CNN, instance segmentation)
- Need inpainting to fill removed regions convincingly (otherwise the gap looks like a door or void to the wall model)

A realistic shortcut is **SAM (Segment Anything Model)** — it's pretrained and can segment arbitrary objects with point prompts. You could let the user click furniture to remove it, rather than doing it automatically. Still complex but removes the training data problem.

The simpler alternative: include maps with lots of furniture in training data and let the model learn to ignore it. This has been working reasonably well already.

**Verdict:** Not worth building from scratch. Could revisit with SAM as an *interactive* cleanup tool (user-assisted, not automatic) once the wall model itself is more mature. Low priority.

---

## 2. Grayscale / B&W Conversion

**Idea:** Convert map images to grayscale (or train on grayscale versions) so the model works with simpler, less biased data.

**Analysis:**

Wall position is fundamentally a structural/textural signal, not a colour signal. Colour can hurt generalisation — a model trained on maps with red brick walls might learn "red = wall" rather than "edge-of-solid-region = wall".

**Pros:**
- Removes colour bias entirely
- Map art styles vary wildly in palette; grayscale normalises this
- Could improve generalisation to new styles
- Easy to implement: `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` repeated to 3 channels

**Cons:**
- Some maps use colour to distinguish wall types (e.g. blue for water barriers, brown for wooden walls) — grayscale loses this
- Already-trained RGB model would need retraining from scratch
- The pretrained SegFormer was trained on colour images; grayscale may degrade those features initially

**Better approach than full grayscale: Random Grayscale Augmentation.** During training, randomly convert tiles to grayscale (e.g. 30% probability). This teaches the model to work with both colour and grayscale inputs without discarding colour information entirely. This is a standard torchvision augmentation (`transforms.RandomGrayscale(p=0.3)`).

**Verdict:** Don't switch to full grayscale. Add random grayscale as a training augmentation instead — low effort, probably helps generalisation. Worth adding in the next training run.

---

## 3. User Texture Input as Wall Examples

**Idea:** Accept 1–N example images (crops of walls and doors from the current map) as additional input during both training and inference. The model uses these to identify "what counts as a wall on this map."

**Analysis:**

This is the most exciting idea on the list. It's essentially **few-shot or example-based segmentation** — a well-researched area. The core insight: walls in TTRPG maps vary enormously in appearance across artists. A model trained globally averages over all styles; a model shown "here's what a wall looks like on *this* map" can specialise.

**How it could work:**

1. **Template matching (simple baseline):** Compute cross-correlation between example wall patches and the current tile. Add the resulting similarity map as an extra input channel. The model learns to associate "high similarity to example" with wall class.

2. **Prototype embedding (better):**
   - Encode example crops with a CNN to produce a prototype vector (similar to GlobalEncoder)
   - Compute cosine similarity between the prototype and each spatial position in the tile feature map
   - Concatenate the similarity map to the decoder features (same fusion point as the global context vector)

3. **Cross-attention (most powerful, also most complex):** Use the example crops as "keys/values" in a cross-attention layer where tile features are "queries." This is how models like Painter and few-shot transformers work.

**Training:**
- For each training sample, randomly sample 1–3 small crops from the ground-truth wall regions of the same image as "example textures"
- Use these as the texture prompt input
- The model learns: "given these crops as wall examples, find all similar regions"

**Why it's a gamechanger:** It turns a general model into a per-map specialist at inference time, with no retraining. Users paste in a few wall texture samples and the model becomes far more accurate on that specific style.

**Complexity:** Medium-high. Prototype embedding approach is the best starting point — similar complexity to the global context encoder already built. Cross-attention is significantly more work.

**Verdict:** High priority for a future experiment. Start with the prototype embedding approach. This is the single idea most likely to meaningfully improve real-world accuracy.

---

## 4. Focal Loss Curriculum: No Focal → Focal → No Focal

**Idea:** Train in three phases:
1. Standard cross-entropy to learn general representations
2. Switch to focal loss to force learning on hard/rare examples
3. Switch back to cross-entropy to "force accuracy on what it learned"

**Analysis:**

The first two phases make good sense as a curriculum:
- Phase 1 CE gives the model stable early training (focal loss can destabilise training from a random init since it overweights mispredicted pixels)
- Phase 2 focal loss pushes the model to tackle hard maps (unusual textures, sparse walls, different lighting) rather than just refining already-easy examples

The rationale for Phase 3 is less clear to me. The theory might be:
- Focal loss during phase 2 pushes the model to be more aggressive (predict wall on uncertain pixels), which increases false positives
- Returning to CE in phase 3 penalises false positives equally with false negatives, which should prune the aggressive predictions back

This is plausible — we saw focal loss alone increase false positives substantially. The curriculum might get the benefit (learning hard examples) while phase 3 corrects the precision problem.

**How to try it:**
```
Phase 1: Epochs 1–15, CE with class weights [0.1, 1.0, 2.0]
Phase 2: Epochs 16–30, Focal (γ=2) with class weights [0.1, 1.0, 2.0]
Phase 3: Epochs 31–40, CE with class weights [0.1, 1.0, 2.0]
```

Use a learning rate warmup for each phase transition to avoid instability.

**Risk:** Phase transitions can cause training instability. Use a lower LR for phase 3.

**Verdict:** Worth trying. Already have focal loss implemented. Easy to add phase-based loss switching to the training script. Medium priority — try after the grayscale augmentation, which is lower effort.

---

## 5. Freeze Early Layers, Fine-tune Later Ones

**Idea:** After initial training, freeze the early encoder layers (which have learned general visual features) and only update the later layers and decoder (which are more task-specific).

**Analysis:**

This is standard transfer learning practice. For SegFormer the natural split is:
- **Freeze:** SegFormer encoder (Mix Transformer backbone, pretrained on ImageNet/ADE20K) — it already knows edges, textures, and structural patterns
- **Train:** SegFormer decoder + classifier, GlobalEncoder, any new modules

This is actually already partially being done — the SegFormer is initialised with pretrained ADE20K weights. The question is whether the full encoder needs continued updates or whether it's already good enough and just the decoder needs tuning.

**When this matters most:** If paired with Idea 3 (texture input). Workflow:
1. Train full model end-to-end until plateau
2. Freeze encoder
3. Add texture prototype input at the decoder level
4. Train only the decoder + new texture modules

This gives the texture input module a stable feature space to work with, and the frozen encoder won't drift due to the new task signal.

**How to implement:**
```python
# Freeze encoder
for param in model.segformer_model.segformer.parameters():
    param.requires_grad = False
# Only decoder and classifier get gradient updates
```

**Verdict:** Medium priority. Not a standalone improvement, but a good technique to pair with the texture input idea (Idea 3) or to stabilise fine-tuning in later training phases.

---

## 6. Replace Global 256×256 Context with Neighbourhood Tile; Remove Overlap

**Idea:** Instead of the 256×256 downscaled whole-map global context, provide a larger crop of the map surrounding the current tile as context. Also remove tile overlap since the model already "sees" walls adjacent to the masked region.

**Analysis:**

**On removing the 256×256 global context:**

The global context currently answers: *"What kind of map is this?"* — stone dungeon vs wooden inn vs outdoor path. This macro-level signal is genuinely useful and different from local neighbourhood context. A neighbourhood crop answers: *"What's immediately adjacent to this tile?"* — useful but doesn't tell the model about the overall art style.

Replacing one with the other is a tradeoff, not a strict improvement. A neighbourhood crop would help with local continuity but lose style-level context.

**Alternative: use both.** Keep the 256×256 global context vector, and separately also extend the tile to include a border of surrounding pixels (padding the tile with 1–2 grid cells of context). This is less complex than adding a full second context image.

**On removing tile overlap:**

Currently tiles overlap by 50%, and predictions are averaged in the overlap regions. If we increase the tile size to include surrounding context (without needing overlap-averaging for border quality), removing overlap makes sense for efficiency.

But removing overlap *without* giving the model context of surrounding tiles means it loses border awareness. The overlap is serving two purposes:
1. Redundancy/averaging for better predictions at tile edges
2. Providing context about adjacent regions

If we pad each tile with surrounding map pixels (say, 64px of context on each side that get masked out in the loss), we get the context benefit without the averaging overhead.

**Verdict:** Partially agree. The "replace global 256×256 with neighbourhood" part is a trade-down. But adding surrounding-pixel padding to tiles (without needing overlap averaging) is a good idea. The overlap could be reduced from 50% to 25% or 0% if tiles include padding context, which would speed up inference meaningfully. Medium priority; cleaner architecture.

---

## Priority Order

| Priority | Idea | Effort | Expected Impact |
|----------|------|--------|----------------|
| 1 | **Random Grayscale Augmentation** (#2 partial) | Very Low | Medium — better generalisation |
| 2 | **Focal Loss Curriculum** (#4) | Low | Medium — may fix the false positive problem |
| 3 | **Tile Padding + Reduced Overlap** (#6 partial) | Medium | Medium — cleaner inference, less overlap averaging |
| 4 | **Texture Prototype Input** (#3) | High | High — per-map specialisation |
| 5 | **Freeze + Fine-tune for Texture Input** (#5) | Low (paired with #3) | Depends on #3 |
| 6 | **SAM-assisted Furniture Removal** (#1) | High | Unknown — maybe marginal |

---

*Ideas source: SCC meetup conversation, 2026-02-25*
