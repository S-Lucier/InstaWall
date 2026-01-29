"""
Scale the tight crop masks to fit the PNG dimensions properly.

Strategy:
- Start with tight crop masks (correct geometry from JSON)
- Determine scale factor by comparing dungeon extent in PNG to mask size
- Place scaled mask at correct position within full PNG canvas
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def find_dungeon_bounds_in_png(png_path):
    """
    Find the actual dungeon region in the PNG by looking for content.

    Returns: (x, y, width, height) of dungeon region
    """
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to find dark content (walls, rooms)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find bounding box of all content
    coords = cv2.findNonZero(binary)
    if coords is None:
        return (0, 0, img.shape[1], img.shape[0])

    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, w, h)


def create_scaled_mask(json_path, png_path, tight_mask_path, debug=False):
    """
    Create a full-size mask by scaling the tight crop mask appropriately.
    """
    # Load tight mask (has correct geometry)
    tight_mask = np.array(Image.open(tight_mask_path))

    # Load PNG to get dimensions
    png = Image.open(png_path)

    # Find where dungeon actually is in the PNG
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = find_dungeon_bounds_in_png(png_path)

    if debug:
        print(f"{Path(png_path).stem}:")
        print(f"  PNG size: {png.size}")
        print(f"  Dungeon region: ({dungeon_x}, {dungeon_y}) {dungeon_w}×{dungeon_h}")
        print(f"  Tight mask size: {tight_mask.shape[1]}×{tight_mask.shape[0]}")

    # Calculate scale to fit tight mask into dungeon region
    # Use the minimum scale to ensure everything fits
    scale_x = dungeon_w / tight_mask.shape[1]
    scale_y = dungeon_h / tight_mask.shape[0]
    scale = min(scale_x, scale_y)

    # Scale the tight mask
    new_w = int(tight_mask.shape[1] * scale)
    new_h = int(tight_mask.shape[0] * scale)

    scaled_mask = cv2.resize(tight_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center it in the dungeon region
    offset_x = dungeon_x + (dungeon_w - new_w) // 2
    offset_y = dungeon_y + (dungeon_h - new_h) // 2

    # Create full-size mask canvas
    full_mask = np.zeros((png.height, png.width), dtype=np.uint8)

    # Place scaled mask at correct position
    y1, y2 = offset_y, offset_y + new_h
    x1, x2 = offset_x, offset_x + new_w

    # Clamp to bounds
    y1, y2 = max(0, y1), min(png.height, y2)
    x1, x2 = max(0, x1), min(png.width, x2)

    # Copy scaled mask into position
    mask_y1 = 0 if offset_y >= 0 else -offset_y
    mask_x1 = 0 if offset_x >= 0 else -offset_x
    mask_y2 = mask_y1 + (y2 - y1)
    mask_x2 = mask_x1 + (x2 - x1)

    full_mask[y1:y2, x1:x2] = scaled_mask[mask_y1:mask_y2, mask_x1:mask_x2]

    if debug:
        print(f"  Scale factor: {scale:.3f}")
        print(f"  Scaled mask: {new_w}×{new_h}")
        print(f"  Position: ({offset_x}, {offset_y})")

    return full_mask


def process_all(png_dir, tight_mask_dir, output_dir, debug=False):
    """
    Process all dungeons.
    """
    png_dir = Path(png_dir)
    tight_mask_dir = Path(tight_mask_dir)
    output_dir = Path(output_dir)

    # Find all PNGs
    png_files = list(png_dir.glob("*.png"))

    print(f"Processing {len(png_files)} dungeons...")
    print()

    # Create output directories
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    processed = 0
    for png_file in png_files:
        base_name = png_file.stem

        # Find corresponding tight mask
        tight_mask_file = tight_mask_dir / f"{base_name}.png"

        if not tight_mask_file.exists():
            print(f"Warning: No tight mask for {base_name}, skipping")
            continue

        try:
            # Create scaled mask
            full_mask = create_scaled_mask(None, png_file, tight_mask_file, debug)

            # Save mask
            mask_output = mask_dir / f"{base_name}.png"
            Image.fromarray(full_mask).save(mask_output)

            # Copy original PNG
            img_output = img_dir / f"{base_name}.png"
            Image.open(png_file).save(img_output)

            # Create debug overlay
            if debug:
                img_cv = cv2.imread(str(png_file))
                overlay = img_cv.copy()
                overlay[full_mask == 255] = [0, 255, 0]
                blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
                cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

            if not debug:
                print(f"Processed: {base_name}")

            processed += 1

        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"Successfully processed {processed}/{len(png_files)} dungeons")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale tight crop masks to fit PNGs")
    parser.add_argument("--png-dir", default=r"C:\Users\shini\Downloads\watabou_exports",
                       help="Directory with original PNG files")
    parser.add_argument("--tight-masks", default="data/watabou_circular/masks",
                       help="Directory with tight crop masks")
    parser.add_argument("--output", default="data/watabou_final",
                       help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Save debug overlays")

    args = parser.parse_args()

    process_all(args.png_dir, args.tight_masks, args.output, args.debug)
