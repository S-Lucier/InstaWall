# Mask Dilation

## Problem It Solves

Foundry line masks represent walls as 2–3px thin lines in a 512px tile. That's less than
0.1% of all pixels. The model can miss entire walls and barely lose any IoU — the signal
is so sparse that gradient updates from wall pixels are swamped by the background.

This is one reason the model latches onto easy texture shortcuts (cobblestone): those
maps have many wall pixels that are easy to get right, while sparse-wall maps barely
move the loss whether the model is right or wrong.

## How It Works

Before the tile is fed to the model, each wall pixel is expanded outward by a fixed
radius (e.g. 5px) using morphological dilation. A wall segment that was 2px wide
becomes ~12px wide. The model now sees a meaningful band to optimise against.

Dilation is applied **only during training** — the validation loop and inference use
the original thin masks. This means IoU during training will appear slightly higher
than at inference (because the label is wider), but the model learns *where* walls are
from a much stronger signal.

Class boundaries are preserved: dilation only writes into background pixels.
Door pixels are never overwritten by wall dilation, and vice versa.

## Hyperparameter: dilation radius

| Radius | Effect |
|--------|--------|
| 0 | Off (original thin masks) |
| 3 | Subtle — widens 2px lines to ~8px |
| 5 | Moderate — recommended starting point |
| 8 | Aggressive — starts to blur spatial precision |

Too large a radius causes the model to predict wide bands rather than precise wall lines.
5px is a good default for 140px grid / 512px tile (each grid cell ≈ 29px at tile scale,
so 5px dilation ≈ 17% of a grid cell width).

## CLI flag

```
--mask-dilation 5
```

## Interaction with other tricks

- Works well with focal loss: focal loss handles the easy/hard imbalance across maps,
  dilation handles the sparse-signal problem within each map.
- Slightly reduces the need for high class weights on the wall class, since wall pixels
  are now more numerous.
