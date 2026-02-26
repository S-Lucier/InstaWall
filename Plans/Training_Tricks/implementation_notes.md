# Implementation Notes — Augmentation & Tiling Changes

*Session: 2026-02-25*

Three features implemented on branch `segformer-experiment`:

1. **Grayscale augmentation** — random greyscale conversion during training (flag, default on)
2. **Focal loss phase schedule** — CE → Focal → CE curriculum (flag, default off)
3. **Context border tiling** — tiles include surrounding pixels as context; overlap-based stitching replaced with context-crop stitching (flag, default on); global context (256×256 whole-map) made toggleable

---

## 1. Grayscale Augmentation

**Goal:** Teach the model to recognise walls by shape/texture rather than colour. Reduces colour bias across different map art styles.

**Flag:** `--no-grayscale-aug` to disable (default: enabled, p=0.3).

### Files changed

#### `config.py`
Added field:
```python
grayscale_aug: bool = True
```

#### `dataset.py`
- `WallSegmentationDataset.__init__`: new `grayscale_aug: bool = True` param
- `_build_transform`: adds `A.ToGray(p=0.3)` to Foundry augmentation pipeline when `grayscale_aug=True`
- Watabou transform already had `A.ToGray(p=0.2)` — left unchanged

#### `train.py`
- New CLI flag: `--no-grayscale-aug` → sets `grayscale_aug=False` in Config
- Passed through to dataset

**Design note:** Probability 0.3 (30% of tiles converted to greyscale). High enough to regularise against colour shortcuts, low enough to keep colour information available most of the time.

---

## 2. Focal Loss Phase Schedule

**Goal:** CE → Focal curriculum: train with standard CE first for stable initialisation, switch to focal loss to force learning on hard maps, optionally switch back to CE for precision cleanup.

**Flag:** `--focal-schedule` (default off). When enabled, controls phase timing with `--focal-start-epoch` and `--focal-end-epoch`.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--focal-schedule` | off | Enable the CE→Focal (→CE) curriculum |
| `--focal-start-epoch N` | 15 | Epoch to switch from CE to Focal |
| `--focal-end-epoch N` | 30 | Epoch to switch back to CE (0 = never) |
| `--focal-gamma F` | 2.0 | Focal loss gamma (existing flag, unchanged) |

`--focal-loss` (existing flag) still works as before: always-on focal loss with no scheduling.

### Files changed

#### `config.py`
Added fields:
```python
focal_loss_schedule: bool = False
focal_schedule_start: int = 15
focal_schedule_end: int = 30
```

#### `train.py`
- `Trainer.__init__`: always creates both `self.ce_criterion` and `self.focal_criterion` (when either focal flag is active); `self.criterion` starts as CE when schedule is enabled
- New `_update_criterion(epoch)` method: called at start of each epoch, switches criterion if epoch crosses the schedule thresholds, logs the switch
- CLI: `--focal-schedule`, `--focal-start-epoch`, `--focal-end-epoch` flags added

**Design note:** `_update_criterion` uses object identity (`is not`) to avoid redundant log messages when the criterion is already at the target. Criterion switch is always logged with the epoch number.

---

## 3. Context Border Tiling

**Goal:** Each tile includes N grid cells of surrounding map as context. The model sees its neighbours when predicting, removing the need for overlap-based stitching. At inference, non-overlapping tiles with context are faster and have no averaging artefacts at tile boundaries.

**Flags:**
- `--tile-context-cells N` (default 1): cells of context on each side. `0` = disabled (old behaviour).
- `--no-tile-context`: shorthand for `--tile-context-cells 0`
- When context > 0, inference overlap defaults to 0.0 (no need to average edges).

### How it works

**Extraction (TileExtractor):**
- Inner tile: `tile_grid_cells * grid_size` native pixels (e.g. 8×140 = 1120 px)
- With context: extract `(tile_grid_cells + 2×context_cells) * grid_size` px (e.g. 10×140 = 1400 px)
- Region is padded with zeros at image edges
- Resized to `tile_size` (512 px) as before — model always receives a 512×512 input
- The effective scale changes: more native pixels → model sees walls at slightly smaller relative size (comparable to zooming out ~12% for context=1 with 8-cell tiles)

**Training:**
- Dataset uses same larger extraction for image tile AND mask tile
- Loss is computed on the full output including context border pixels (they have real labels)
- The context border produces some overlap between adjacent training tiles, but since training tiles are randomly sampled this is benign

**Stitching (TileStitcher):**
- Context-aware path: crop `context_model_px` from each edge of the prediction before placing
- `context_model_px = round(tile_size × context_px / total_native_px)`
- For context=1, tile_cells=8, tile_size=512: `context_model_px = round(512 × 140 / 1400) = 51 px` cropped from each side; centre 410×410 region used
- Result placed at `(info.x, info.y)` in the output (inner tile position)
- No special edge-tile handling needed: context padding is zero at image edges, and we always crop the same margin

**Inference overlap:**
- With context > 0: overlap defaults to 0.0 (context handles edge quality)
- Old checkpoints (no `tile_context_cells` in config): default reads as 0, old overlap of 0.5 still applies unless overridden

### Files changed

#### `config.py`
- `tile_context_cells: int = 1` (default on)
- Fixed assertion: `0 < tile_overlap < 1` → `0 <= tile_overlap < 1` (allows 0.0 for context-tile inference)

#### `tiling.py`
- `TileExtractor.__init__`: `context_cells: int = 0` param
- `TileExtractor.extract_tile`: expands extraction region by `context_cells × grid_size` on all sides, zero-pads at image edges, resizes full (context + tile + context) region to `tile_size`
- `TileStitcher.__init__`: `context_cells: int = 0` param
- `TileStitcher.stitch`: when `context_cells > 0`, uses context-crop path instead of overlap-crop path; always crops `context_model_px` from each edge
- `TilePipeline.__init__`: `context_cells: int = 0`, passed to extractor and stitcher

#### `dataset.py`
- `WallSegmentationDataset.__init__`: `tile_context_cells: int = 1` param
- `TileExtractor` instantiated with `context_cells=tile_context_cells`
- Mask extraction shares the same extractor, so both image and mask tiles include the context border

#### `train.py`
- `--tile-context-cells N` / `--no-tile-context` flags
- `--no-global-context`: disables 256×256 whole-map context for `segformer_gc` (overrides the auto-enable in Config)
- Passed through to dataset and config

#### `inference.py`
- `predict_mask`: reads `tile_context_cells` from model checkpoint config (defaults to 0 for old checkpoints)
- Passes `context_cells` to `TilePipeline`
- When `tile_context_cells > 0`, inference overlap auto-set to 0.0 (overridable with `--overlap`)
- `--no-global-context` flag: disables global image pass even for `segformer_gc` checkpoints

---

## Backward Compatibility

| Scenario | Behaviour |
|----------|-----------|
| Old checkpoint + new inference | `tile_context_cells` reads as 0 from config → old overlap-based stitching used |
| New checkpoint (context=1) + new inference | context=1 read from config, overlap auto=0.0 |
| Old checkpoint + `--overlap 0.5` explicit | Correct — old path, specified overlap |
| `--focal-schedule` without `--focal-loss` | Focal only active during schedule window, CE elsewhere |
| `--no-grayscale-aug` | Foundry transform has no ToGray; Watabou transform unchanged (it always had ToGray) |
