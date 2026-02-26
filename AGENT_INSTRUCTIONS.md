# InstaWall — Agent Instructions

*Last updated: 2026-02-25. Branch at time of writing: `segformer-experiment`.*

This file is the ground truth for running and training the model. Read it before touching anything.

> **IMPORTANT FOR AGENTS:** Update this file whenever you make changes to the codebase. Specifically:
> - New CLI flags → add to the relevant flags table
> - New model types or architecture changes → update Architecture Notes
> - Experiment results → add a row to Experiment History
> - New files or directory structure changes → update Directory Structure
> - New gotchas discovered → add to Known Issues
> - Update the `Last updated` date and branch name at the top
>
> Keep it accurate. A future agent reading stale instructions will waste time or break things.

---

## What This Project Does

ML pipeline to detect walls and doors in TTRPG battlemap images and export them as Foundry VTT wall objects. Pipeline:

1. **Foundry JSON → mask** — `Tools/foundry_to_mask_v3.py` converts Foundry scene exports to single-channel segmentation masks
2. **Train** — `primary_model_training/train.py` trains a segmentation model on those image/mask pairs
3. **Inference** — `primary_model_training/inference.py` runs the trained model on a new image and outputs a predicted mask + Foundry wall JSON

---

## Directory Structure

```
InstaWall/
├── primary_model_training/    # Main model code
│   ├── train.py               # Training script
│   ├── inference.py           # Inference script
│   ├── model.py               # WallSegmentationUNet, SegFormerWrapper, GlobalContextSegFormer
│   ├── dataset.py             # WallSegmentationDataset (tile extraction, augmentation)
│   ├── tiling.py              # TileExtractor, TileStitcher, TilePipeline
│   ├── losses.py              # FocalLoss
│   ├── config.py              # Config dataclass (all hyperparameters)
│   └── pretrained/            # Local pretrained SegFormer weights
│       └── segformer-b0/      # (and b2, etc.)
├── Tools/
│   └── foundry_to_mask_v3.py  # Foundry JSON → mask converter
├── data/
│   └── foundry_to_mask/
│       ├── Map_Images/        # Battlemap image files (.jpg/.png/.webp)
│       ├── line_masks/        # Generated masks (*_mask_lines.png)
│       └── *.json             # Foundry scene exports (fvtt-*.json)
├── outputs/
│   └── wall_segmentation/     # Training output dirs (timestamped)
│       └── YYYYMMDD_HHMMSS/
│           ├── checkpoint_best.pt
│           ├── checkpoint_latest.pt
│           ├── checkpoint_epochN.pt
│           ├── config.json
│           ├── history.json
│           └── stop_training.bat   # Drop STOP file to gracefully halt training
├── Plans/                     # Design docs and experiment plans
│   ├── FOUNDRY_TO_MASK_PLAN.md
│   ├── Model_Architecture_Changes_SCC.md   # Ideas from meetup, with analysis
│   └── Training_Tricks/
│       ├── focal_loss.md
│       ├── mask_dilation.md
│       ├── global_context_plan.md
│       └── implementation_notes.md         # Notes on recent code changes
├── gpu_power_training.bat     # Run as admin before training (sets GPU power limit)
└── AGENT_INSTRUCTIONS.md      # This file
```

---

## Step 1 — Generate Training Masks

Convert Foundry VTT scene exports to segmentation masks.

```bash
# Single file
python Tools/foundry_to_mask_v3.py data/foundry_to_mask/fvtt-Scene-MyMap.json --viz

# Batch (all fvtt-*.json in a directory)
python Tools/foundry_to_mask_v3.py data/foundry_to_mask/ --batch --viz

# Remove border walls (walls that run along the image edge)
python Tools/foundry_to_mask_v3.py data/foundry_to_mask/ --batch --remove-edge-walls
```

**Output:** `data/foundry_to_mask/line_masks/{MapName}_mask_lines.png`

**Class IDs in mask files (stored as value × 50):**
- `0` → Background
- `50` → Wall (includes secret doors and windows)
- `100` → Terrain (partial blockers, `sight=10, light=10`)
- `150` → Door

The training script divides by 50 to get class indices 0–3.

---

## Step 2 — Train

Run from the project root. The script is a module so it must be run with `-m`.

```bash
python -m primary_model_training.train [OPTIONS]
```

**Before training on GPU:** run `gpu_power_training.bat` as administrator.

### Common Training Recipes

```bash
# Baseline SegFormer-B0 with all current defaults (context tiling on, grayscale aug on)
python -m primary_model_training.train \
  --model segformer \
  --segformer-variant b0 \
  --merge-terrain \
  --epochs 400 \
  --batch-size 4 \
  --lr 1e-4

# SegFormer with global context (256x256 whole-map awareness)
python -m primary_model_training.train \
  --model segformer_gc \
  --merge-terrain \
  --epochs 400 \
  --batch-size 4

# SegFormer-B0 with focal loss schedule (CE -> Focal at epoch 15 -> CE at epoch 30)
python -m primary_model_training.train \
  --model segformer \
  --merge-terrain \
  --focal-schedule \
  --focal-start-epoch 15 \
  --focal-end-epoch 30 \
  --focal-gamma 2.0 \
  --epochs 200

# Disable new tiling (fall back to old 50% overlap stitching)
python -m primary_model_training.train \
  --model segformer \
  --merge-terrain \
  --no-tile-context \
  --epochs 200

# Resume from checkpoint
python -m primary_model_training.train \
  --model segformer \
  --merge-terrain \
  --resume outputs/wall_segmentation/YYYYMMDD_HHMMSS/checkpoint_latest.pt \
  --epochs 400
```

### All Training Flags

#### Data
| Flag | Default | Description |
|------|---------|-------------|
| `--image-dir` | `data/foundry_to_mask/Map_Images` | Battlemap images |
| `--mask-dir` | `data/foundry_to_mask/line_masks` | Mask files |
| `--metadata-file` | None | JSON mapping map names to grid sizes |
| `--output-dir` | `outputs/wall_segmentation` | Where checkpoints go |
| `--watabou-dir` | None | Watabou data directory (see below) |
| `--watabou-prob` | 0.36 | Per-epoch inclusion probability for Watabou maps |

#### Model
| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `unet` | `unet`, `segformer`, `segformer_gc` |
| `--segformer-variant` | `b0` | `b0`–`b5` (only for segformer/segformer_gc) |
| `--no-aspp` | — | Disable ASPP bottleneck (UNet only) |
| `--no-attention` | — | Disable attention gates (UNet only) |
| `--merge-terrain` | — | Merge terrain into wall class (4→3 classes) |

#### Training
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 1000 | Total epochs |
| `--batch-size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--early-stopping` | 50 | Stop after N epochs without improvement |
| `--ema-decay` | 0.999 | EMA decay (0 = disabled) |
| `--seed` | 42 | Random seed |
| `--num-workers` | 4 | DataLoader workers |
| `--no-augment` | — | Disable all augmentation |
| `--resume` | None | Path to checkpoint to resume |
| `--save-interval` | 50 | Save checkpoint every N epochs |

#### Loss
| Flag | Default | Description |
|------|---------|-------------|
| `--focal-loss` | — | Always use focal loss (replaces cross-entropy) |
| `--focal-gamma` | 2.0 | Focal loss gamma |
| `--focal-schedule` | — | CE->Focal->CE curriculum |
| `--focal-start-epoch` | 15 | Epoch to switch CE -> Focal |
| `--focal-end-epoch` | 30 | Epoch to switch Focal -> CE (0 = never) |

#### Augmentation
| Flag | Default | Description |
|------|---------|-------------|
| `--no-grayscale-aug` | — | Disable random greyscale (default: 30% probability) |
| `--mask-dilation` | 0 | Dilate training mask lines by N pixels (strengthens sparse wall signal) |

#### Tiling
| Flag | Default | Description |
|------|---------|-------------|
| `--tile-grid-cells` | 8 | Grid cells per tile (model sees this many at once) |
| `--tile-size` | 512 | Model input resolution in pixels |
| `--tile-context-cells` | 1 | Extra grid cells of surrounding context included in each tile |
| `--no-tile-context` | — | Disable context border (equivalent to `--tile-context-cells 0`) |
| `--no-global-context` | — | Disable 256x256 whole-map context for `segformer_gc` |

#### To Stop Training Early

Double-click `stop_training.bat` inside the run's output directory. Creates a `STOP` sentinel file; training saves and exits cleanly at the end of the current epoch.

---

## Step 3 — Inference

```bash
python -m primary_model_training.inference \
  --checkpoint outputs/wall_segmentation/YYYYMMDD_HHMMSS/checkpoint_best.pt \
  --image path/to/map.jpg \
  --grid-size 140
```

The script reads `model_type`, `tile_context_cells`, `use_global_context`, etc. directly from the checkpoint's saved config, so most settings are automatic.

### Common Inference Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--image` | required | Input battlemap image |
| `--grid-size` | required | Grid cell size in pixels for this map |
| `--output-dir` | same as image | Where to write outputs |
| `--no-mask` | — | Skip saving predicted mask PNG |
| `--no-viz` | — | Skip saving visualization overlay |
| `--no-walls` | — | Skip saving Foundry wall JSON |
| `--overlap` | 0.5* | Tile overlap for stitching (*auto-set to 0.0 for context-tile checkpoints) |
| `--tta` | 0 | Test-time augmentation passes (1–8; 8 = all flips + rotations, majority vote) |
| `--min-component-size` | 0.0 | Remove isolated predictions smaller than `N × grid_size²` pixels |
| `--no-global-context` | — | Disable 256x256 whole-map context (segformer_gc only) |
| `--cpu` | — | Force CPU inference |

### Outputs

- `{name}_predicted_mask.png` — class mask (pixel values 0/50/100/150)
- `{name}_predicted_viz.jpg` — colour overlay on original image
- `{name}_walls.json` — Foundry VTT compatible wall JSON

---

## Architecture Notes

### Model Types

| Model | Params | Notes |
|-------|--------|-------|
| `unet` | ~31M | Custom UNet with ASPP bottleneck + attention gates |
| `segformer` | ~4M (b0) | HuggingFace SegFormer, pretrained on ADE20K |
| `segformer_gc` | ~4.5M (b0) | SegFormer + GlobalEncoder: 256x256 whole-map context vector injected into decoder |

### Class System (default, 4 classes)

| Index | Name | Foundry Values |
|-------|------|---------------|
| 0 | Background | Unmarked pixels |
| 1 | Wall | `door=0, sight=20, light=20`; also secret doors and windows |
| 2 | Terrain | `sight=10, light=10` (partial blockers) |
| 3 | Door | `door=1` |

With `--merge-terrain`: terrain→wall, door becomes class 2. 3 classes total.

### Tiling (Context Border Mode — current default)

- Training tiles extract `(tile_grid_cells + 2 × tile_context_cells) × grid_size` native pixels, resized to 512px
- Model sees more of the map per tile; walls at tile edges have full neighbourhood context
- Stitching crops the context border out of predictions and places only the inner tile in the output
- No overlap needed at inference (auto-set to 0.0)

### Tiling (Legacy Overlap Mode — `--no-tile-context`)

- Training tiles: `tile_grid_cells × grid_size` pixels, no overlap
- Inference: `tile_overlap` (default 0.5) overlap between tiles
- Stitching uses center-crop: only the central fraction of each tile prediction is used, where model has most context
- Edge tiles use a wider crop to ensure full image coverage

---

## Experiment History

| Branch | Model | Key Changes | Result |
|--------|-------|-------------|--------|
| `main` | UNet | ASPP + attention, various class weights | Baseline |
| `segformer-experiment` | SegFormer B0/B2 | Global context (256x256), EMA, TTA | Improved generalisation |
| `segformer-experiment` | SegFormer B0 | Mask dilation | Stronger wall gradient signal |
| `segformer-experiment` | SegFormer B0 | Focal loss | Fewer false negatives, but too many false positives |
| `segformer-experiment` (current) | SegFormer B0 | Grayscale aug, focal schedule, context tiling | In progress |

**Focal loss alone:** reduced false negatives but caused widespread false positives (model predicted walls across entire maps). Consider `--focal-schedule` to limit it to the middle of training.

**Mask dilation:** helps when wall masks are very thin (2–3px). Use radius 3–5 (`--mask-dilation 5`).

**Global context (`segformer_gc`):** gives tile predictions awareness of the overall map art style. Particularly useful for maps where local texture alone is ambiguous.

---

## Watabou Data

Watabou maps are procedurally generated dungeons with clean, vector-style walls. They are a different distribution from Foundry photorealistic maps and are included at reduced probability to supplement training without dominating it.

```bash
# Include Watabou data
python -m primary_model_training.train \
  --model segformer \
  --merge-terrain \
  --watabou-dir data/watabou \
  --watabou-prob 0.36 \
  --epochs 400
```

Expected structure under `--watabou-dir`:
```
watabou/
├── watabou_images/       # PNG map images
└── watabou_edge_mask/    # Corresponding mask PNGs (same filename)
```

---

## GPU Setup (Windows)

Before each training run:
1. Run `gpu_power_training.bat` as **administrator** (sets GPU power limit to avoid thermal throttle)
2. After training: `gpu_power_default.bat` to restore

Monitor temperature: `gpu_temp_monitor.bat`

---

## Known Issues / Gotchas

- **Checkpoint format:** checkpoints store `config` as a dict (via `vars(config)`). Loading old checkpoints with new code is generally fine since new config fields have defaults. Inference reads model_type, tile_context_cells, etc. directly from the saved config.
- **Context tiles + old checkpoints:** old checkpoints don't have `tile_context_cells` in their config, so inference defaults to 0 (old overlap mode). Use `--overlap 0.5` explicitly if needed.
- **Foundry padding offset:** Foundry wall coordinates are in padded canvas space. The mask converter applies `math.ceil(width × padding / grid_size) × grid_size` offset to align walls with the image. This is correct per Foundry's implementation.
- **Windows path separators:** the codebase uses `pathlib.Path` throughout; forward/back slashes are handled automatically.
- **UnicodeEncodeError in print:** Windows cp1252 console can't print non-ASCII like `γ`, `→`, `×`. Print strings use ASCII alternatives.
