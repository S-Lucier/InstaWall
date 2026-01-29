"""
Generate masks using ONLY JSON data to avoid region detection errors.

Creates a tight bounding box around the dungeon based purely on JSON coordinates,
then you can overlay and scale this to match your PNG exactly.
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def json_to_mask_tight(json_path, pixels_per_grid=100, padding=2):
    """
    Convert JSON to mask using ONLY JSON data (no PNG region detection).

    Args:
        json_path: Path to JSON file
        pixels_per_grid: Uniform scaling factor (will create square grid cells)
        padding: Grid units of padding around the dungeon

    Returns:
        mask: Binary mask (tight crop, square grid cells)
        info: Dict with dimensions and bounds for debugging
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    if not rects:
        raise ValueError("No rectangles found in JSON")

    # Find grid bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # Calculate pixel dimensions
    px_width = int(grid_width * pixels_per_grid)
    px_height = int(grid_height * pixels_per_grid)

    # Create mask (all black = walls)
    mask = np.zeros((px_height, px_width), dtype=np.uint8)

    # Fill rooms (white = playable)
    for rect in rects:
        x1 = int((rect['x'] - min_x) * pixels_per_grid)
        y1 = int((rect['y'] - min_y) * pixels_per_grid)
        x2 = int((rect['x'] + rect['w'] - min_x) * pixels_per_grid)
        y2 = int((rect['y'] + rect['h'] - min_y) * pixels_per_grid)

        # Clamp
        x1, x2 = max(0, x1), min(px_width, x2)
        y1, y2 = max(0, y1), min(px_height, y2)

        mask[y1:y2, x1:x2] = 255

    info = {
        'grid_bounds': (min_x, min_y, max_x, max_y),
        'grid_size': (grid_width, grid_height),
        'pixel_size': (px_width, px_height),
        'pixels_per_grid': pixels_per_grid
    }

    return mask, info


def json_to_mask_matched(json_path, png_path, debug=False):
    """
    Create mask that exactly matches PNG dimensions by analyzing
    the image content to find precise dungeon boundaries.
    """
    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    if not rects:
        raise ValueError("No rectangles found in JSON")

    # Find grid bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # Load PNG
    img = Image.open(png_path)
    img_cv = cv2.imread(str(png_path))

    # More precise region detection using the grid pattern
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect grid lines (common in Watabou maps)
    edges = cv2.Canny(gray, 30, 100)

    # Find horizontal and vertical projections
    h_projection = np.sum(edges, axis=1)  # Sum across width
    v_projection = np.sum(edges, axis=0)  # Sum across height

    # Find the range where there's significant edge content
    h_threshold = np.max(h_projection) * 0.05
    v_threshold = np.max(v_projection) * 0.05

    y_start = np.argmax(h_projection > h_threshold)
    y_end = len(h_projection) - np.argmax(h_projection[::-1] > h_threshold)

    x_start = np.argmax(v_projection > v_threshold)
    x_end = len(v_projection) - np.argmax(v_projection[::-1] > v_threshold)

    region_x, region_y = x_start, y_start
    region_w, region_h = x_end - x_start, y_end - y_start

    # Calculate scaling
    scale_x = region_w / grid_width
    scale_y = region_h / grid_height

    # Create full-size mask
    full_mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # Fill rooms
    for rect in rects:
        px1 = (rect['x'] - min_x) * scale_x + region_x
        py1 = (rect['y'] - min_y) * scale_y + region_y
        px2 = (rect['x'] + rect['w'] - min_x) * scale_x + region_x
        py2 = (rect['y'] + rect['h'] - min_y) * scale_y + region_y

        x1, y1 = int(round(px1)), int(round(py1))
        x2, y2 = int(round(px2)), int(round(py2))
        x1, x2 = max(0, x1), min(img.width, x2)
        y1, y2 = max(0, y1), min(img.height, y2)

        full_mask[y1:y2, x1:x2] = 255

    if debug:
        debug_dir = Path(png_path).parent / "debug3"
        debug_dir.mkdir(exist_ok=True)
        base_name = Path(png_path).stem

        # Save mask
        Image.fromarray(full_mask).save(debug_dir / f"{base_name}_mask.png")

        # Create overlay
        overlay = img_cv.copy()
        overlay[full_mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

        # Draw detected region
        region_img = img_cv.copy()
        cv2.rectangle(region_img, (region_x, region_y),
                     (region_x + region_w, region_y + region_h),
                     (0, 255, 0), 3)
        cv2.imwrite(str(debug_dir / f"{base_name}_region.png"), region_img)

        # Save projections for debugging
        proj_viz = np.zeros((img.height, img.width, 3), dtype=np.uint8)
        # Normalize projections
        h_norm = (h_projection / np.max(h_projection) * 50).astype(int)
        v_norm = (v_projection / np.max(v_projection) * 50).astype(int)

        # Draw them
        for i, val in enumerate(h_norm):
            cv2.line(proj_viz, (0, i), (val, i), (255, 0, 0), 1)
        for i, val in enumerate(v_norm):
            cv2.line(proj_viz, (i, 0), (i, val), (0, 255, 0), 1)

        cv2.imwrite(str(debug_dir / f"{base_name}_projections.png"), proj_viz)

        print(f"{base_name}:")
        print(f"  Region: ({region_x}, {region_y}) {region_w}x{region_h}")
        print(f"  Scale: X={scale_x:.1f}, Y={scale_y:.1f}")

    return full_mask


def batch_process_tight(input_dir, output_dir="data/watabou_tight", pixels_per_grid=100):
    """Generate tight (cropped) masks from JSON only."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Generating tight masks for {len(json_files)} dungeons...")
    print(f"Using {pixels_per_grid} pixels per grid unit\n")

    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for json_file in json_files:
        try:
            mask, info = json_to_mask_tight(json_file, pixels_per_grid)

            base_name = json_file.stem
            mask_output = mask_dir / f"{base_name}.png"

            Image.fromarray(mask).save(mask_output)

            print(f"{base_name}:")
            print(f"  Grid: {info['grid_size'][0]:.1f} x {info['grid_size'][1]:.1f}")
            print(f"  Mask: {info['pixel_size'][0]} x {info['pixel_size'][1]} px")

            processed += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print(f"\nProcessed {processed}/{len(json_files)} dungeons")
    print(f"Output: {output_dir}")
    print("\nThese are tight crops - overlay and scale to fit your PNG")


def batch_process_matched(input_dir, output_dir="data/watabou_matched", debug=False):
    """Generate full-size masks with improved region detection."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Generating matched masks for {len(json_files)} dungeons...\n")

    img_dir = Path(output_dir) / "images"
    mask_dir = Path(output_dir) / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for json_file in json_files:
        png_file = json_file.with_suffix('.png')
        if not png_file.exists():
            continue

        try:
            mask = json_to_mask_matched(json_file, png_file, debug)

            base_name = json_file.stem

            # Copy image
            img = Image.open(png_file)
            img.save(img_dir / f"{base_name}.png")

            # Save mask
            Image.fromarray(mask).save(mask_dir / f"{base_name}.png")

            if not debug:
                print(f"Processed: {base_name}")

            processed += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nProcessed {processed}/{len(json_files)} dungeons")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate exact masks from Watabou JSON")
    parser.add_argument("input_dir", help="Directory with PNG/JSON pairs")
    parser.add_argument("--mode", choices=["tight", "matched"], default="matched",
                       help="tight=cropped masks (JSON only), matched=full-size (improved detection)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--pixels-per-grid", type=int, default=100,
                       help="Pixels per grid unit (tight mode only)")
    parser.add_argument("--debug", action="store_true", help="Debug visualizations")

    args = parser.parse_args()

    if args.mode == "tight":
        output = args.output or "data/watabou_tight"
        batch_process_tight(args.input_dir, output, args.pixels_per_grid)
    else:
        output = args.output or "data/watabou_matched"
        batch_process_matched(args.input_dir, output, args.debug)
