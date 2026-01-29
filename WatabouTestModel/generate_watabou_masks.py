"""
Generate masks using the EXACT transformation from the Foundry VTT module.

Based on: https://github.com/TarkanAl-Kazily/one-page-parser

Key parameters:
- Grid size: 70 pixels per grid unit (Watabou's default export)
- Wall offset: 0.25 grid units toward interior corners
- Edge offset: -0.25 or 0.75 based on door presence
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


GRID_SIZE = 70  # Watabou's default export grid size
WALL_OFFSET = 0.25  # From Foundry module


class MatrixMap:
    """Recreate the MatrixMap logic from Foundry module."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = {}

    def put(self, x, y):
        """Mark a grid cell as filled."""
        self.grid[(x, y)] = True

    def get(self, x, y):
        """Check if a grid cell is filled."""
        return self.grid.get((x, y), False)

    def addRect(self, rect):
        """Add a rectangular room to the grid."""
        for x in range(rect['x'], rect['x'] + rect['w']):
            for y in range(rect['y'], rect['y'] + rect['h']):
                self.put(x, y)


def generate_mask_foundry_style(json_path, png_path, circular=True):
    """
    Generate mask using Foundry VTT transformation logic.

    Args:
        json_path: Path to Watabou JSON
        png_path: Path to Watabou PNG
        circular: Whether to render circular rooms

    Returns:
        mask: 3-class mask matching PNG dimensions
              - 0 (black) = walls/exterior
              - 127 (gray) = doors
              - 255 (white) = floors/rooms
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    doors = data.get('doors', [])

    # Load PNG for dimensions
    img = Image.open(png_path)

    # Create matrix map (for Foundry's wall generation logic)
    # We don't need walls, but we'll use the grid for room placement
    min_x = min(r['x'] for r in rects)
    max_x = max(r['x'] + r['w'] for r in rects)
    min_y = min(r['y'] for r in rects)
    max_y = max(r['y'] + r['h'] for r in rects)

    # Check for doors on edges (Foundry logic)
    # Find leftmost and topmost tiles
    min_tile_x = min(r['x'] for r in rects)
    min_tile_y = min(r['y'] for r in rects)

    # Check if edges have doors (ALL door types count for offset calculation)
    x_edge_has_tile = any(
        d['x'] == min_tile_x
        for d in doors
    )
    y_edge_has_tile = any(
        d['y'] == min_tile_y
        for d in doors
    )

    # Door types we actually render as gray (127)
    DOOR_TYPES_TO_MARK = {1, 5, 6, 7}

    # Calculate offsets (Foundry logic)
    x_offset = -0.25 * GRID_SIZE if x_edge_has_tile else 0.75 * GRID_SIZE
    y_offset = -0.25 * GRID_SIZE if y_edge_has_tile else 0.75 * GRID_SIZE

    # Apply correction for observed 18px shift (right and down)
    x_offset += 18
    y_offset += 18

    # Create mask
    mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # Detect circular rooms based on 'rotunda' property
    room_shapes = {}
    if circular:
        for i, rect in enumerate(rects):
            # Check if this room is marked as a rotunda (circular)
            if rect.get('rotunda', False):
                room_shapes[i] = 'circular'
            else:
                room_shapes[i] = 'rectangular'
    else:
        room_shapes = {i: 'rectangular' for i in range(len(rects))}

    # Render rooms (including 1x1 hallway segments)
    for i, rect in enumerate(rects):
        # Transform grid coordinates to pixel coordinates (Foundry style)
        # Step 1: Multiply by grid size
        px1 = rect['x'] * GRID_SIZE
        py1 = rect['y'] * GRID_SIZE
        px2 = (rect['x'] + rect['w']) * GRID_SIZE
        py2 = (rect['y'] + rect['h']) * GRID_SIZE

        # Step 2: Normalize to origin (subtract minimum)
        px1 -= min_x * GRID_SIZE
        px2 -= min_x * GRID_SIZE
        py1 -= min_y * GRID_SIZE
        py2 -= min_y * GRID_SIZE

        # Step 3: Add offsets
        px1 += x_offset
        px2 += x_offset
        py1 += y_offset
        py2 += y_offset

        # Convert to integers
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

        # Clamp to image bounds
        px1 = max(0, min(img.width, px1))
        px2 = max(0, min(img.width, px2))
        py1 = max(0, min(img.height, py1))
        py2 = max(0, min(img.height, py2))

        # Render room
        if room_shapes.get(i) == 'circular' and px2 > px1 and py2 > py1:
            center_x = (px1 + px2) // 2
            center_y = (py1 + py2) // 2
            radius_x = (px2 - px1) // 2
            radius_y = (py2 - py1) // 2

            if radius_x > 0 and radius_y > 0:
                cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        elif px2 > px1 and py2 > py1:
            mask[py1:py2, px1:px2] = 255

    # Render actual doors as separate class (1x1 grid cells, value=127)
    # Only mark types 1, 5, 6, 7 as doors (see watabou_door_types.txt)
    # DOOR_TYPES_TO_MARK already defined above for edge detection

    for door in doors:
        door_type = door.get('type', 0)

        # Skip non-door types (open passages, stairs, barred, etc.)
        if door_type not in DOOR_TYPES_TO_MARK:
            continue

        # Transform door coordinates
        px1 = door['x'] * GRID_SIZE
        py1 = door['y'] * GRID_SIZE
        px2 = (door['x'] + 1) * GRID_SIZE
        py2 = (door['y'] + 1) * GRID_SIZE

        # Normalize to origin
        px1 -= min_x * GRID_SIZE
        px2 -= min_x * GRID_SIZE
        py1 -= min_y * GRID_SIZE
        py2 -= min_y * GRID_SIZE

        # Add offsets
        px1 += x_offset
        px2 += x_offset
        py1 += y_offset
        py2 += y_offset

        # Convert to integers
        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

        # Clamp to image bounds
        px1 = max(0, min(img.width, px1))
        px2 = max(0, min(img.width, px2))
        py1 = max(0, min(img.height, py1))
        py2 = max(0, min(img.height, py2))

        # Render door as gray (127) for 3-class segmentation
        if px2 > px1 and py2 > py1:
            mask[py1:py2, px1:px2] = 127

    return mask


def batch_process(input_dir, output_dir, circular=True, debug=False):
    """Process all Watabou exports."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Processing {len(json_files)} dungeons with Foundry transformation...")
    print(f"Grid size: {GRID_SIZE}px")
    print(f"Circular rooms: {circular}")
    print()

    # Create output directories
    img_dir = Path(output_dir) / "images"
    mask_dir = Path(output_dir) / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(exist_ok=True)

    processed = 0
    for json_file in json_files:
        png_file = json_file.with_suffix('.png')

        if not png_file.exists():
            print(f"Warning: No PNG for {json_file.name}, skipping")
            continue

        try:
            base_name = json_file.stem

            # Generate mask
            mask = generate_mask_foundry_style(json_file, png_file, circular)

            # Save mask
            mask_output = mask_dir / f"{base_name}.png"
            Image.fromarray(mask).save(mask_output)

            # Copy PNG
            img_output = img_dir / f"{base_name}.png"
            Image.open(png_file).save(img_output)

            # Debug overlay
            if debug:
                img_cv = cv2.imread(str(png_file))
                overlay = img_cv.copy()
                overlay[mask == 255] = [0, 255, 0]
                blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
                cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

            print(f"Processed: {base_name}")
            processed += 1

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"Successfully processed {processed}/{len(json_files)} dungeons")
    print(f"Output: {output_dir}")
    print()
    print("Masks use Foundry VTT transformation:")
    print(f"  - Grid size: {GRID_SIZE}px per unit")
    print("  - Edge offset: 0.25 or 0.75 units based on doors")
    print("  - Circular rooms detected by aspect ratio")
    print()
    print("3-class mask values:")
    print("  - 0 (black) = walls/exterior")
    print("  - 127 (gray) = doors (types 1, 5, 6, 7 only)")
    print("  - 255 (white) = floors/rooms")
    print()
    print("Door filtering (see Text_Files/watabou_door_types.txt):")
    print("  - Marked as doors: Types 1, 5, 6, 7")
    print("  - Skipped: Types 0, 2, 3, 4, 8, 9 (passages/stairs/bars)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate masks using Foundry VTT transformation")
    parser.add_argument("input_dir", help="Directory with Watabou PNG/JSON exports")
    parser.add_argument("--output", default="data/watabou_foundry",
                       help="Output directory")
    parser.add_argument("--circular", action='store_true', default=True,
                       help="Render circular rooms (default: True)")
    parser.add_argument("--no-circular", dest='circular', action='store_false',
                       help="Disable circular rooms")
    parser.add_argument("--debug", action="store_true", help="Save debug overlays")

    args = parser.parse_args()

    batch_process(args.input_dir, args.output, args.circular, args.debug)
