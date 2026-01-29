"""
Unified script to generate training masks from Watabou dungeons.

Usage:
    python generate_training_masks.py <input_dir> [options]

Examples:
    # Generate playable area masks (recommended)
    python generate_training_masks.py "C:\\path\\to\\watabou_exports" --type playable --circular

    # Generate wall masks
    python generate_training_masks.py "C:\\path\\to\\watabou_exports" --type walls

    # Generate wall centerlines
    python generate_training_masks.py "C:\\path\\to\\watabou_exports" --type centerlines
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# Optional import for centerlines
try:
    from skimage.morphology import skeletonize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def create_playable_mask(json_path, pixels_per_grid=150, circular=True):
    """
    Create mask of playable areas.
    White = playable, Black = walls/exterior
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

    mask = np.zeros((px_height, px_width), dtype=np.uint8)

    # Identify circular rooms
    column_rooms = set()
    if circular:
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

        if circular:
            aspect_ratio = max(w_px, h_px) / min(w_px, h_px) if min(w_px, h_px) > 0 else 999
            min_size = 200
            is_circular = (aspect_ratio < 1.3 and min(w_px, h_px) >= min_size) or i in column_rooms

            if is_circular:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius_x = (x2 - x1) // 2
                radius_y = (y2 - y1) // 2
                cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y),
                           0, 0, 360, 255, -1)
                continue

        # Rectangular
        mask[y1:y2, x1:x2] = 255

    return mask


def create_wall_mask(playable_mask, wall_thickness=2):
    """
    Create mask of walls only (boundaries between playable and non-playable).
    White = walls, Black = everything else
    """
    # Find boundaries
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(playable_mask, kernel, iterations=wall_thickness)
    eroded = cv2.erode(playable_mask, kernel, iterations=wall_thickness)

    # Walls are the difference
    walls = dilated - eroded

    return walls


def create_centerline_mask(playable_mask):
    """
    Create mask of wall centerlines (skeleton of wall regions).
    White = centerlines, Black = everything else

    Requires: scikit-image (pip install scikit-image)
    """
    if not HAS_SKIMAGE:
        raise ImportError("centerlines require scikit-image. Install with: pip install scikit-image")

    # Get wall regions (inverse of playable)
    walls = 255 - playable_mask

    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Skeletonize
    skeleton = skeletonize(walls > 0)
    centerlines = (skeleton * 255).astype(np.uint8)

    return centerlines


def process_dungeon(json_path, output_dir, mask_type='playable', pixels_per_grid=150, circular=True):
    """Process a single dungeon and generate the specified mask type."""
    base_name = Path(json_path).stem

    # Generate base playable mask
    playable_mask = create_playable_mask(json_path, pixels_per_grid, circular)

    # Generate requested mask type
    if mask_type == 'playable':
        final_mask = playable_mask
    elif mask_type == 'walls':
        final_mask = create_wall_mask(playable_mask)
    elif mask_type == 'centerlines':
        final_mask = create_centerline_mask(playable_mask)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")

    # Save
    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_output = mask_dir / f"{base_name}.png"
    Image.fromarray(final_mask).save(mask_output)

    return mask_output


def batch_process(input_dir, output_dir, mask_type='playable', pixels_per_grid=150, circular=True):
    """Process all dungeons in a directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Generating {mask_type} masks for {len(json_files)} dungeons...")
    print(f"Pixels per grid: {pixels_per_grid}")
    print(f"Circular rooms: {circular}")
    print()

    processed = 0
    for json_file in json_files:
        try:
            mask_path = process_dungeon(json_file, output_dir, mask_type, pixels_per_grid, circular)
            print(f"Processed: {json_file.stem}")
            processed += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Successfully processed {processed}/{len(json_files)} dungeons")
    print(f"Output directory: {output_dir}")
    print(f"\nMask type: {mask_type}")
    print("  - White pixels = target feature")
    print("  - Black pixels = background")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training masks from Watabou dungeons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Playable area masks with circular rooms (recommended)
  python generate_training_masks.py "C:\\path\\to\\exports" --type playable --circular

  # Wall boundary masks
  python generate_training_masks.py "C:\\path\\to\\exports" --type walls --output data/walls

  # Wall centerline masks
  python generate_training_masks.py "C:\\path\\to\\exports" --type centerlines
        """
    )

    parser.add_argument("input_dir", help="Directory containing JSON files from Watabou")
    parser.add_argument("--type", choices=['playable', 'walls', 'centerlines'],
                       default='playable',
                       help="Type of mask to generate (default: playable)")
    parser.add_argument("--output", help="Output directory (default: auto-generated)")
    parser.add_argument("--pixels-per-grid", type=int, default=150,
                       help="Pixels per grid unit (default: 150)")
    parser.add_argument("--circular", action='store_true', default=True,
                       help="Render circular rooms (default: True)")
    parser.add_argument("--no-circular", dest='circular', action='store_false',
                       help="Disable circular room rendering")

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output is None:
        suffix = "_circular" if args.circular else "_rect"
        args.output = f"data/watabou_{args.type}{suffix}"

    batch_process(args.input_dir, args.output, args.type, args.pixels_per_grid, args.circular)

    print("\nNext steps:")
    print("1. In GIMP: Overlay masks on PNGs and scale uniformly to align")
    print("2. Generate more dungeons from https://watabou.itch.io/one-page-dungeon")
    print("3. Prepare dataset: python prepare_dataset.py")
    print("4. Train model: python train.py")
