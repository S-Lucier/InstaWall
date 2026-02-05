# Foundry VTT to Mask Converter

## Overview

Convert Foundry VTT scene exports (JSON) into segmentation masks for training the Universal Battlemap model.

---

## 1. Foundry JSON Format

### Scene Structure
```json
{
  "name": "Scene Name",
  "width": 4800,           // Scene dimensions in pixels
  "height": 6000,
  "padding": 0.25,         // Padding fraction (0 to 0.5 typical)
  "grid": {
    "size": 200,           // Pixels per grid cell
    "type": 1,             // 1 = square grid
    "distance": 5,         // Feet per cell
    "units": "ft"
  },
  "background": {
    "src": "path/to/image.jpg",
    "offsetX": 0,
    "offsetY": 0
  },
  "walls": [...]           // Array of wall objects
}
```

### Wall Object Structure
```json
{
  "_id": "unique_id",
  "c": [x1, y1, x2, y2],   // Line segment coordinates
  "move": 20,              // Movement restriction (20=blocks, 0=none)
  "sight": 20,             // Vision restriction (20=blocks, 10=limited, 0=none)
  "light": 20,             // Light restriction (20=blocks, 10=limited, 0=none)
  "door": 0,               // Door type (0=none, 1=door, 2=secret door)
  "ds": 0,                 // Door state (0=closed, 1=open, 2=locked)
  "dir": 0,                // Direction (for one-way walls)
  "sound": 20              // Sound restriction
}
```

---

## 2. Coordinate System & Offset Calculation

### The Problem
Foundry uses a padded canvas coordinate system. Wall coordinates are in canvas space, not image space.

### Solution: Grid-Aligned Offset
```python
def calculate_offset(scene_width, scene_height, padding, grid_size):
    if padding == 0:
        return (0, 0)

    offset_x = round(scene_width * padding / grid_size) * grid_size
    offset_y = round(scene_height * padding / grid_size) * grid_size
    return (offset_x, offset_y)
```

### Why Rounding?
- Padding may result in non-integer grid cells (e.g., 7.5 cells)
- Foundry rounds to nearest grid boundary
- Using `round()` (not floor/ceil) matches Foundry's behavior

### Verified Test Cases
| Map | Padding | Grid | Raw Offset | Grid-Aligned |
|-----|---------|------|------------|--------------|
| Crypt (padded) | 0.25 | 200 | (1200, 1500) | (1200, 1600) |
| Brazenthrone | 0.25 | 140 | (1610, 1540) | (1680, 1540) |
| Crypt (no pad) | 0 | 200 | (0, 0) | (0, 0) |

---

## 3. Wall Type Classification

### From Foundry Values to Classes

| Foundry Values | Original Type | Converted To | Reason |
|----------------|---------------|--------------|--------|
| `door=0, sight=20, light=20` | **Wall** | Wall | Standard wall (blocks all) |
| `door=0, sight=10, light=10` | **Terrain** | Terrain | Partial blocker (rubble, pillars) |
| `door=1` | **Door** | Door | Normal door |
| `door=2` | **Secret Door** | **Wall** | Visually indistinguishable from walls |
| `light=0, sight=0, move=20` | **Window (normal)** | **Wall** | Model can't distinguish from wall gaps |
| `light=30, sight=30, move=20` | **Window (proximity)** | **Wall** | Model can't distinguish from wall gaps |

### Special Handling

**Secret Doors:**
- Drawn as walls (class = wall) because they're visually indistinguishable
- BUT treated as doors for room connectivity check in hybrid mode
- Rooms behind secret doors are still marked as playable floor, not enclosed wall space

**Windows:**
- Two types in Foundry:
  - Normal window: `light=0, sight=0, move=20`
  - Proximity window: `light=30, sight=30, move=20` (has threshold data)
- Both converted to walls since model can't reliably distinguish from wall gaps
- Could potentially add as separate class in future if enough training data

### Final Class System (4 classes for training)
1. **Background/Void** - Unmarked (lines mode) or exterior (hybrid mode)
2. **Wall** - Includes standard walls, secret doors, and windows
3. **Terrain** - `sight=10, light=10` (partial blockers)
4. **Door** - `door=1` only (normal doors)

---

## 4. Mask Generation Modes

### CHOSEN: Mode A - Lines-Only (Same as Option D)
Draw wall segments as thick lines on the mask. No floor detection.

**Pros:**
- Simple implementation
- Directly represents Foundry data
- No flood-fill edge cases
- Model learns floor vs void from visual appearance

**Cons:**
- No explicit floor class
- Model must infer playable area from image

**Implementation:**
```python
for wall in walls:
    x1, y1, x2, y2 = wall['c']
    # Apply offset
    x1, y1 = x1 - offset_x, y1 - offset_y
    x2, y2 = x2 - offset_x, y2 - offset_y
    # Draw with thickness
    draw.line([(x1, y1), (x2, y2)], fill=class_id, width=wall_thickness)
```

**Wall Thickness:**
- Default: `grid_size * 0.05` (5% of grid)
- Configurable via `--wall-thickness`

---

### CHOSEN: Mode C - Hybrid (Lines + Floor Detection)
Combine line-based walls with flood-fill floor detection.

**Steps:**
1. Create blank mask (all zeros = void)
2. Draw walls/terrain/doors as thick lines with class IDs
3. Flood fill from corners → mark as void (class 0)
4. Remaining unmarked pixels → mark as floor (class 1)
5. Post-process landlocked regions:
   - Small enclosed areas (< 1 grid cell²) → mark as terrain
   - Large enclosed areas → keep as floor

**Pros:**
- Explicit floor class for training
- Distinguishes interior from exterior

**Cons:**
- More complex
- Landlocked regions need special handling

---

### NOT USING: Option B (Pure Flood-Fill)
Skipped - hybrid approach (C) incorporates the useful parts.

---

## 5. Output Format

### Single-Channel Class Mask (Chosen)
Classes are mutually exclusive, so single-channel is optimal:
- More memory efficient
- CrossEntropyLoss naturally enforces "pick one class"
- Softmax output gives comparable confidences across classes
- Standard for semantic segmentation (UNet, SegFormer, etc.)

### Class IDs
**Mode A (Lines-Only):**
```
0 = Background (unmarked - model learns floor vs void from image)
1 = Wall (includes secret doors and windows)
2 = Terrain
3 = Door
```

**Mode B (Wall Mask - inverted floor):**
```
0 = Void/Exterior
1 = Wall (interior + drawn walls + secret doors + windows)
2 = Terrain
3 = Door
```

**Mode C (Hybrid with Floor):**
```
0 = Void/Exterior
1 = Floor
2 = Wall (includes secret doors and windows)
3 = Terrain
4 = Door
```

### Output Files
- `{name}_mask.png` - Single-channel class mask (PNG, lossless)
- `{name}_viz.jpg` - Colored overlay for review (optional)

---

## 6. Script Interface

### Two Modes
Single script with mode flag, or two separate scripts:

**Mode A: Lines-Only** (`--mode lines`)
- Draws walls/terrain/doors as thick lines
- Background remains class 0 (unmarked)
- Model learns floor vs void from visual appearance
- Simplest, avoids flood-fill edge cases

**Mode C: Hybrid with Floor** (`--mode hybrid`)
- Draws walls/terrain/doors as thick lines
- Flood fills from corners to detect exterior void
- Remaining unmarked pixels become floor class
- Post-processes small landlocked regions

### Input
- Foundry JSON export file
- Corresponding background image (auto-detected from JSON if in same dir)

### Output
- `{name}_mask.png` - Single-channel class mask
- `{name}_viz.jpg` - Colored overlay (optional, with `--viz`)

### CLI Options
```bash
# Single file - lines only
python foundry_to_mask.py scene.json --mode lines

# Single file - hybrid with floor detection
python foundry_to_mask.py scene.json --mode hybrid --viz

# Batch process directory
python foundry_to_mask.py ./foundry_exports/ --mode lines --batch

# Custom wall thickness
python foundry_to_mask.py scene.json --wall-thickness 12
```

### Batch Mode
Process all JSON files in directory, auto-matching images by name.

---

## 7. Implementation Phases

### Phase 1: Core Converter - Mode A (Lines-Only)
- [x] Parse Foundry JSON
- [x] Calculate grid-aligned offset
- [x] Classify wall types (wall/terrain/door)
- [ ] Draw walls as thick lines on mask
- [ ] Output single-channel PNG mask
- [ ] CLI interface with `--mode lines`
- [ ] Visualization output (`--viz`)

### Phase 2: Mode C (Hybrid with Floor)
- [ ] Exterior flood fill from corners
- [ ] Mark remaining pixels as floor
- [ ] Landlocked region handling (size threshold)
- [ ] CLI flag `--mode hybrid`

### Phase 3: Batch & Polish
- [ ] Batch directory processing
- [ ] Auto-match JSON to images by name
- [ ] Progress bar for batch
- [ ] Summary statistics output

---

## 8. Decisions & Open Questions

### Decided
- **Output format**: Single-channel class mask (classes are mutually exclusive)
- **Mask modes**: Both lines-only (A) and hybrid (C) - compare training results
- **Floor detection**: Test both approaches to see which trains better

### Still Open
1. **Wall thickness**: Fixed pixels or proportional to grid?
   - Suggestion: Default to `grid_size * 0.05` (5% of grid), configurable

2. **Landlocked handling** (for hybrid mode): Size threshold seems simplest
   - Suggestion: Regions < 1 grid cell² = mark as wall/terrain

3. **Secret doors**: Include as separate class or merge with walls?
   - They're designed to look like walls, may be hard to detect
   - Suggestion: Keep separate class, can always merge later if performance is poor

4. **Multi-floor maps**: Some Foundry scenes have elevation data
   - Suggestion: Ignore for now, handle in future version if needed

---

## 9. File Naming Convention

**Input:**
- `fvtt-Scene-{name}-{id}.json`
- `{MapName}.jpg`

**Output:**
- `{MapName}_mask.png` - Class mask
- `{MapName}_viz.jpg` - Visualization (optional)
- `{MapName}_meta.json` - Metadata (optional)

---

*Document created: 2026-02-04*
*Project: InstaWall - Foundry to Mask Converter*
