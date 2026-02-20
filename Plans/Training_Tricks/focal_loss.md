# Focal Loss

## Problem It Solves

Standard cross-entropy with class weights can still be dominated by "easy" examples —
pixels where the model is already confident and correct. With class weights of
[0.1, 1.0, 2.0], wall pixels contribute more to the loss, but a cobblestone wall the
model already recognises at 95% confidence contributes almost nothing useful.

The result: gradients are dominated by the examples the model already handles well
(familiar cobblestone walls), and the hard cases it keeps getting wrong (unfamiliar
textures, unusual lighting, maps with few walls) barely move the weights.

## How It Works

Focal loss multiplies each pixel's cross-entropy loss by a factor `(1 - p)^γ`:

```
FL = -(1 - p_correct)^γ * log(p_correct)
```

Where:
- `p_correct` is the model's predicted probability for the correct class
- `γ` (gamma) controls how aggressively easy examples are down-weighted

### Concrete example (γ = 2):

| Model confidence | CE loss | Focal factor | Focal loss |
|-----------------|---------|--------------|------------|
| p = 0.95 (easy, correct) | 0.051 | (0.05)² = 0.003 | 0.00015 |
| p = 0.70 (moderate) | 0.357 | (0.30)² = 0.090 | 0.032 |
| p = 0.30 (hard, wrong) | 1.204 | (0.70)² = 0.490 | 0.590 |

The cobblestone walls the model already knows get near-zero gradient. Unfamiliar
textures it keeps getting wrong dominate training. This is exactly what we want.

## Hyperparameter: gamma (γ)

| γ | Effect |
|---|--------|
| 0 | Equivalent to standard cross-entropy |
| 1 | Mild — slight down-weighting of easy examples |
| 2 | Standard (original paper, recommended starting point) |
| 5 | Aggressive — focuses almost entirely on hard examples, can destabilise training |

γ = 2 is the standard choice and a safe default.

## Interaction with class weights

Focal loss and class weights are complementary:
- Class weights: up-weights the wall *class* globally (compensates for background dominance)
- Focal loss: down-weights *easy* examples dynamically (compensates for texture shortcuts)

Both should be used together. The class weights may need slight reduction when using
focal loss (since focal loss already reduces background dominance somewhat).

## CLI flag

```
--focal-loss --focal-gamma 2.0
```

## Reference

Lin et al., "Focal Loss for Dense Object Detection" (RetinaNet paper), ICCV 2017.
