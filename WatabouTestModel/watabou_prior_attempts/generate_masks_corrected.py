"""
Generate accurate training masks from Watabou JSON with proper scaling.

Key insight: Grid cells are NOT square - they're stretched independently in X and Y
to fit the PNG layout. We need separate scaling factors for each axis.
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def find_dungeon_region(png_path):
    """Find the actual dungeon map region within the PNG."""
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use edge detection to find textured region
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = img.shape[:2]
        margin = 50
        return (margin, margin, w - 2*margin, h - 2*margin)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h)


def json_to_mask(json_path, png_path, debug=False):
    """
    Convert Watabou JSON to properly scaled training mask.

    Creates mask where:
    - Playable areas (rooms) = WHITE (255)
    - Walls/exterior = BLACK (0)

    Returns:
        mask: Binary mask matching PNG dimensions
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    if not rects:
        raise ValueError("No rectangles found in JSON")

    # Find grid coordinate bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # Load PNG and find dungeon region
    img = Image.open(png_path)
    region_x, region_y, region_w, region_h = find_dungeon_region(png_path)

    # Calculate scaling factors (different for X and Y!)
    scale_x = region_w / grid_width
    scale_y = region_h / grid_height

    # Create full-size mask (all black = walls/exterior)
    full_mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # Fill rooms with white (playable space)
    for rect in rects:
        # Convert grid coordinates to pixel coordinates
        px1 = (rect['x'] - min_x) * scale_x + region_x
        py1 = (rect['y'] - min_y) * scale_y + region_y
        px2 = (rect['x'] + rect['w'] - min_x) * scale_x + region_x
        py2 = (rect['y'] + rect['h'] - min_y) * scale_y + region_y

        # Convert to integers and clamp
        x1, y1 = int(round(px1)), int(round(py1))
        x2, y2 = int(round(px2)), int(round(py2))
        x1, x2 = max(0, x1), min(img.width, x2)
        y1, y2 = max(0, y1), min(img.height, y2)

        # Fill room
        full_mask[y1:y2, x1:x2] = 255

    if debug:
        debug_dir = Path(png_path).parent / "debug2"
        debug_dir.mkdir(exist_ok=True)
        base_name = Path(png_path).stem

        # Save mask
        Image.fromarray(full_mask).save(debug_dir / f"{base_name}_mask.png")

        # Create overlay
        img_cv = cv2.imread(str(png_path))
        overlay = img_cv.copy()

        # Color playable areas green
        overlay[full_mask == 255] = [0, 255, 0]

        # Blend
        blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

        # Draw dungeon region box
        region_img = img_cv.copy()
        cv2.rectangle(region_img, (region_x, region_y),
                     (region_x + region_w, region_y + region_h),
                     (0, 255, 0), 3)
        cv2.imwrite(str(debug_dir / f"{base_name}_region.png"), region_img)

        print(f"{base_name}:")
        print(f"  Grid: {grid_width}×{grid_height} units")
        print(f"  Region: {region_w}×{region_h} pixels at ({region_x}, {region_y})")
        print(f"  Scale: X={scale_x:.1f}, Y={scale_y:.1f} px/unit")

    return full_mask


def process_dungeon_pair(png_path, json_path, output_dir, debug=False):
    """Process a PNG/JSON pair and generate training data."""
    # Generate mask
    mask = json_to_mask(json_path, png_path, debug=debug)

    # Load original image
    img = Image.open(png_path).convert('RGB')

    # Create output paths
    base_name = Path(png_path).stem
    img_dir = Path(output_dir) / "images"
    mask_dir = Path(output_dir) / "masks"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_output = img_dir / f"{base_name}.png"
    mask_output = mask_dir / f"{base_name}.png"

    # Save
    img.save(img_output)
    Image.fromarray(mask).save(mask_output)

    return img_output, mask_output


def batch_process(input_dir, output_dir="data/watabou_corrected", debug=False):
    """Process all PNG/JSON pairs in a directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Processing {len(json_files)} dungeon pairs...\n")

    processed = 0
    for json_file in json_files:
        png_file = json_file.with_suffix('.png')

        if not png_file.exists():
            print(f"Warning: No PNG found for {json_file.name}, skipping")
            continue

        try:
            img_out, mask_out = process_dungeon_pair(png_file, json_file, output_dir, debug)
            if not debug:
                print(f"Processed: {json_file.stem}")
            processed += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Successfully processed {processed}/{len(json_files)} dungeons")
    print(f"Output directory: {output_dir}")

    if debug:
        print(f"Debug visualizations saved to: {Path(input_dir) / 'debug2'}")

    return processed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate training masks from Watabou dungeons with correct scaling"
    )
    parser.add_argument("input_dir", help="Directory containing PNG/JSON pairs")
    parser.add_argument("--output", default="data/watabou_corrected",
                       help="Output directory for training data")
    parser.add_argument("--debug", action="store_true",
                       help="Save debug visualizations")

    args = parser.parse_args()

    batch_process(args.input_dir, args.output, args.debug)

    print("\nNext steps:")
    print("1. Check debug2/ folder to verify mask alignment")
    print("2. If masks look good, proceed with training data preparation")
