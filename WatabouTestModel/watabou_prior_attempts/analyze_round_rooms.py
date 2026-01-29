"""
Analyze which rooms Watabou renders as circular vs rectangular.
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def analyze_room_rendering(json_path, png_path):
    """
    Analyze which rooms appear circular in the PNG vs rectangular in JSON.
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    columns = data.get('columns', [])

    # Create a set of rooms with columns
    column_rooms = set()
    for col in columns:
        # Find which room contains this column
        for i, rect in enumerate(rects):
            if (rect['x'] <= col['x'] < rect['x'] + rect['w'] and
                rect['y'] <= col['y'] < rect['y'] + rect['h']):
                column_rooms.add(i)
                break

    print(f"Analysis for {Path(json_path).stem}:")
    print(f"Total rooms: {len(rects)}")
    print(f"Rooms with columns: {len(column_rooms)}")
    print()

    # Analyze room characteristics
    for i, rect in enumerate(rects):
        w, h = rect['w'], rect['h']
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 999
        has_columns = i in column_rooms
        is_square = aspect_ratio < 1.3  # Nearly square

        shape_guess = "circular?" if (is_square and w >= 3) or has_columns else "rectangular"

        print(f"  Room {i}: {w}x{h}, aspect={aspect_ratio:.2f}, "
              f"columns={has_columns}, likely={shape_guess}")

    print()


def create_circular_mask_variant(json_path, pixels_per_grid=150, circular_threshold=1.3):
    """
    Create mask with circular rooms where appropriate.

    Heuristic: Rooms that are nearly square (aspect ratio < threshold)
    and large enough are rendered as circles.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    columns = data.get('columns', [])

    # Find bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    padding = 2
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    px_width = int(grid_width * pixels_per_grid)
    px_height = int(grid_height * pixels_per_grid)

    # Create mask
    mask = np.zeros((px_height, px_width), dtype=np.uint8)

    # Identify rooms with columns
    column_rooms = set()
    for col in columns:
        for i, rect in enumerate(rects):
            if (rect['x'] <= col['x'] < rect['x'] + rect['w'] and
                rect['y'] <= col['y'] < rect['y'] + rect['h']):
                column_rooms.add(i)
                break

    # Fill rooms
    for i, rect in enumerate(rects):
        x1 = int((rect['x'] - min_x) * pixels_per_grid)
        y1 = int((rect['y'] - min_y) * pixels_per_grid)
        x2 = int((rect['x'] + rect['w'] - min_x) * pixels_per_grid)
        y2 = int((rect['y'] + rect['h'] - min_y) * pixels_per_grid)

        w_px = x2 - x1
        h_px = y2 - y1
        aspect_ratio = max(w_px, h_px) / min(w_px, h_px) if min(w_px, h_px) > 0 else 999

        # Decide if circular
        min_size = 200  # Minimum pixels for circular rendering
        is_circular = (aspect_ratio < circular_threshold and
                      min(w_px, h_px) >= min_size) or i in column_rooms

        if is_circular:
            # Draw circle/ellipse
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius_x = (x2 - x1) // 2
            radius_y = (y2 - y1) // 2
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                       0, 0, 360, 255, -1)
        else:
            # Draw rectangle
            mask[y1:y2, x1:x2] = 255

    return mask


def batch_analyze(input_dir):
    """Analyze all dungeons to understand circular room patterns."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    for json_file in json_files:
        png_file = json_file.with_suffix('.png')
        if png_file.exists():
            analyze_room_rendering(json_file, png_file)
            print("="*60)


def batch_generate_circular(input_dir, output_dir="data/watabou_circular", pixels_per_grid=150):
    """Generate masks with circular rooms."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating masks with circular rooms...")
    print(f"Using {pixels_per_grid} pixels per grid unit\n")

    for json_file in json_files:
        try:
            mask = create_circular_mask_variant(json_file, pixels_per_grid)

            base_name = json_file.stem
            mask_output = mask_dir / f"{base_name}.png"

            Image.fromarray(mask).save(mask_output)
            print(f"Processed: {base_name}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and generate circular room masks")
    parser.add_argument("input_dir", help="Directory with PNG/JSON pairs")
    parser.add_argument("--analyze", action="store_true", help="Analyze room patterns")
    parser.add_argument("--generate", action="store_true", help="Generate masks with circular rooms")
    parser.add_argument("--output", default="data/watabou_circular", help="Output directory")
    parser.add_argument("--pixels-per-grid", type=int, default=150, help="Pixels per grid unit")

    args = parser.parse_args()

    if args.analyze:
        batch_analyze(args.input_dir)

    if args.generate:
        batch_generate_circular(args.input_dir, args.output, args.pixels_per_grid)
