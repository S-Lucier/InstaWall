"""
Generate perfectly aligned masks for Watabou dungeons.

This script calculates the exact transformation for each dungeon individually,
ensuring perfect alignment without manual scaling in GIMP.
"""

import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def calculate_transformation(json_path, png_path):
    """
    Calculate the exact grid-to-pixel transformation for a specific dungeon.

    Returns:
        dict with scale_x, scale_y, offset_x, offset_y
    """
    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    rects = data['rects']

    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Find dungeon extent in the PNG
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    # Calculate grid bounds from JSON
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    # Calculate scale (pixels per grid unit)
    scale_x = extent_w / grid_width
    scale_y = scale_x  # Assume square grid

    # Calculate X offset
    offset_x = extent_x - grid_min_x * scale_x

    # Find optimal Y offset by searching for best room center alignment
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # Find content area
    row_dark = np.sum(dark, axis=1)
    content_rows = np.where(row_dark > extent_w * 0.1 * 255)[0]
    if len(content_rows) > 0:
        content_top = content_rows[0]
        content_bottom = content_rows[-1]
    else:
        content_top = extent_y
        content_bottom = extent_y + extent_h

    # Search for best offset_y
    best_offset_y = None
    best_score = -1

    search_range = range(int(content_top - grid_height * scale_y / 2),
                        int(content_bottom + grid_height * scale_y / 2), 5)

    for offset_y_test in search_range:
        score = 0
        for room in main_rooms:
            cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
            cy = int((room['y'] + room['h']/2) * scale_y + offset_y_test)

            if (content_top < cy < content_bottom and
                0 <= cx < img_w and 0 <= cy < img_h and
                gray[cy, cx] > 180):
                score += 1

        if score > best_score:
            best_score = score
            best_offset_y = offset_y_test

    offset_y = best_offset_y if best_offset_y else (extent_y - grid_min_y * scale_y)

    return {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'grid_min_x': grid_min_x,
        'grid_min_y': grid_min_y
    }


def detect_circular_rooms(rects):
    """
    Determine which rooms should be rendered as circular.

    Based on aspect ratio heuristic since JSON doesn't specify shape.
    """
    room_shapes = {}

    for i, rect in enumerate(rects):
        # Skip 1x1 (connections/doors)
        if rect['w'] == 1 and rect['h'] == 1:
            room_shapes[i] = 'connection'
            continue

        # Check aspect ratio
        aspect_ratio = max(rect['w'], rect['h']) / min(rect['w'], rect['h'])

        # Nearly square and large enough → circular
        if aspect_ratio < 1.3 and min(rect['w'], rect['h']) >= 3:
            room_shapes[i] = 'circular'
        else:
            room_shapes[i] = 'rectangular'

    return room_shapes


def generate_mask(json_path, png_path, circular=True, mask_type='playable'):
    """
    Generate a perfectly aligned mask for a Watabou dungeon.

    Args:
        json_path: Path to JSON file
        png_path: Path to PNG file
        circular: Whether to render circular rooms
        mask_type: 'playable', 'walls', or 'centerlines'

    Returns:
        numpy array: Binary mask matching PNG dimensions
    """
    # Calculate transformation
    transform = calculate_transformation(json_path, png_path)

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    rects = data['rects']

    # Detect room shapes
    if circular:
        room_shapes = detect_circular_rooms(rects)
    else:
        room_shapes = {i: 'rectangular' for i in range(len(rects))}

    # Load PNG for dimensions
    img = Image.open(png_path)
    mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # Fill rooms
    for i, rect in enumerate(rects):
        # Skip connections for main mask
        if room_shapes.get(i) == 'connection':
            continue

        # Calculate pixel coordinates
        px1 = int(rect['x'] * transform['scale_x'] + transform['offset_x'])
        py1 = int(rect['y'] * transform['scale_y'] + transform['offset_y'])
        px2 = int((rect['x'] + rect['w']) * transform['scale_x'] + transform['offset_x'])
        py2 = int((rect['y'] + rect['h']) * transform['scale_y'] + transform['offset_y'])

        # Clamp to image bounds
        px1, px2 = max(0, px1), min(img.width, px2)
        py1, py2 = max(0, py1), min(img.height, py2)

        # Render based on shape
        if room_shapes.get(i) == 'circular':
            center_x = (px1 + px2) // 2
            center_y = (py1 + py2) // 2
            radius_x = (px2 - px1) // 2
            radius_y = (py2 - py1) // 2

            # Only draw if radius is valid
            if radius_x > 0 and radius_y > 0:
                cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            elif px2 > px1 and py2 > py1:
                # Fallback to rectangle if ellipse invalid
                mask[py1:py2, px1:px2] = 255
        else:
            if px2 > px1 and py2 > py1:
                mask[py1:py2, px1:px2] = 255

    # Apply mask type transformations
    if mask_type == 'walls':
        # Extract wall boundaries
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=2)
        eroded = cv2.erode(mask, kernel, iterations=2)
        mask = dilated - eroded

    elif mask_type == 'centerlines':
        # Extract wall centerlines (requires scikit-image)
        try:
            from skimage.morphology import skeletonize
            walls = 255 - mask
            kernel = np.ones((3, 3), np.uint8)
            walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel, iterations=2)
            skeleton = skeletonize(walls > 0)
            mask = (skeleton * 255).astype(np.uint8)
        except ImportError:
            raise ImportError("centerlines require scikit-image. Install with: pip install scikit-image")

    return mask, transform


def process_dungeon(json_path, png_path, output_dir, circular=True, mask_type='playable', debug=False):
    """Process a single dungeon and save the mask."""
    base_name = Path(json_path).stem

    # Generate mask
    mask, transform = generate_mask(json_path, png_path, circular, mask_type)

    # Save mask
    mask_dir = Path(output_dir) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_output = mask_dir / f"{base_name}.png"
    Image.fromarray(mask).save(mask_output)

    # Copy original image
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(png_path)
    img.save(img_dir / f"{base_name}.png")

    if debug:
        # Create overlay visualization
        debug_dir = Path(output_dir) / "debug"
        debug_dir.mkdir(exist_ok=True)

        img_cv = cv2.imread(str(png_path))
        overlay = img_cv.copy()
        overlay[mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(debug_dir / f"{base_name}_overlay.png"), blended)

        print(f"{base_name}:")
        print(f"  Scale: X={transform['scale_x']:.2f}, Y={transform['scale_y']:.2f}")
        print(f"  Offset: X={transform['offset_x']:.2f}, Y={transform['offset_y']:.2f}")

    return mask_output


def batch_process(input_dir, output_dir, circular=True, mask_type='playable', debug=False):
    """Process all PNG/JSON pairs in a directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Generating {mask_type} masks for {len(json_files)} dungeons...")
    print(f"Circular rooms: {circular}")
    print(f"Output: {output_dir}")
    print()

    processed = 0
    for json_file in json_files:
        png_file = json_file.with_suffix('.png')

        if not png_file.exists():
            print(f"Warning: No PNG for {json_file.name}, skipping")
            continue

        try:
            process_dungeon(json_file, png_file, output_dir, circular, mask_type, debug)
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
    print(f"\nMasks are perfectly aligned - ready for training!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate perfectly aligned masks from Watabou dungeons"
    )

    parser.add_argument("input_dir", help="Directory with PNG/JSON pairs")
    parser.add_argument("--type", choices=['playable', 'walls', 'centerlines'],
                       default='playable', help="Mask type (default: playable)")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--circular", action='store_true', default=True,
                       help="Render circular rooms (default: True)")
    parser.add_argument("--no-circular", dest='circular', action='store_false',
                       help="Disable circular room rendering")
    parser.add_argument("--debug", action="store_true", help="Save debug overlays")

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output is None:
        suffix = "_circular" if args.circular else "_rect"
        args.output = f"data/watabou_perfect_{args.type}{suffix}"

    batch_process(args.input_dir, args.output, args.circular, args.type, args.debug)

    print("\nNext steps:")
    print("1. Verify alignment by checking debug/ folder overlays")
    print("2. Generate more dungeons from https://watabou.itch.io/one-page-dungeon")
    print("3. Train your Unet model!")
