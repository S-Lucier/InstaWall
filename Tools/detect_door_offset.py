"""
Detect door graphic position offset from Watabou PNG exports.

Analyzes where the dark door line actually appears in the PNG image
compared to the tile center position computed from JSON coordinates.

For each door:
1. Use grid_to_pixel transform to locate the door tile in the PNG
2. Crop the 1x1 cell region
3. For vertical door lines (ddx != 0): find darkest column cluster
4. For horizontal door lines (ddy != 0): find darkest row cluster
5. Compare against tile center to compute offset

Reports offset in pixels and grid units, grouped by dir and type.
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import argparse


GRID_SIZE = 70  # Watabou's default export grid size
DOOR_TYPES_TO_ANALYZE = {1, 4, 5, 7, 8}


def analyze_map(json_path, png_path):
    """Analyze door offsets for a single map.

    Returns list of dicts with door offset measurements.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    doors = data.get('doors', [])

    if not rects or not doors:
        return []

    img = Image.open(png_path).convert('L')  # grayscale
    img_arr = np.array(img, dtype=np.float64)
    img_h, img_w = img_arr.shape

    # --- Coordinate transform (same as generate_watabou_wall_masks.py) ---
    min_x = min(r['x'] for r in rects)
    min_y = min(r['y'] for r in rects)

    x_edge_has_tile = any(d['x'] == min_x for d in doors)
    y_edge_has_tile = any(d['y'] == min_y for d in doors)

    x_px_offset = -0.25 * GRID_SIZE if x_edge_has_tile else 0.75 * GRID_SIZE
    y_px_offset = -0.25 * GRID_SIZE if y_edge_has_tile else 0.75 * GRID_SIZE

    x_px_offset += 18
    y_px_offset += 18

    def grid_to_pixel(gx, gy):
        px = gx * GRID_SIZE - min_x * GRID_SIZE + x_px_offset
        py = gy * GRID_SIZE - min_y * GRID_SIZE + y_px_offset
        return px, py  # return float for sub-pixel precision

    results = []

    for door in doors:
        door_type = door.get('type', 0)
        if door_type not in DOOR_TYPES_TO_ANALYZE:
            continue

        cx, cy = door['x'], door['y']
        ddx, ddy = door['dir']['x'], door['dir']['y']

        # Get pixel coords of the tile corners
        tile_left, tile_top = grid_to_pixel(cx, cy)
        tile_right, tile_bottom = grid_to_pixel(cx + 1, cy + 1)

        # Tile center in pixel space
        center_px_x = (tile_left + tile_right) / 2.0
        center_px_y = (tile_top + tile_bottom) / 2.0

        # Crop region (clamp to image bounds)
        x0 = max(0, int(round(tile_left)))
        y0 = max(0, int(round(tile_top)))
        x1 = min(img_w, int(round(tile_right)))
        y1 = min(img_h, int(round(tile_bottom)))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img_arr[y0:y1, x0:x1]

        if ddx != 0:
            # Vertical door line - find darkest column
            # Restrict to middle rows (passage area) to avoid wall textures
            h = crop.shape[0]
            margin = max(1, h // 4)
            passage_crop = crop[margin:h - margin, :]

            if passage_crop.size == 0:
                continue

            # Column-wise mean brightness (lower = darker)
            col_brightness = passage_crop.mean(axis=0)

            # Find the darkest column
            darkest_col = np.argmin(col_brightness)
            darkest_px_x = x0 + darkest_col + 0.5  # center of pixel

            offset_px = darkest_px_x - center_px_x
            offset_grid = offset_px / GRID_SIZE

            results.append({
                'map': Path(json_path).stem,
                'cx': cx, 'cy': cy,
                'ddx': ddx, 'ddy': ddy,
                'type': door_type,
                'orientation': 'vertical',
                'offset_px': offset_px,
                'offset_grid': offset_grid,
                'tile_width_px': x1 - x0,
                'darkest_brightness': col_brightness[darkest_col],
                'median_brightness': np.median(col_brightness),
            })

        else:
            # Horizontal door line - find darkest row
            # Restrict to middle columns (passage area) to avoid wall textures
            w = crop.shape[1]
            margin = max(1, w // 4)
            passage_crop = crop[:, margin:w - margin]

            if passage_crop.size == 0:
                continue

            # Row-wise mean brightness
            row_brightness = passage_crop.mean(axis=1)

            # Find the darkest row
            darkest_row = np.argmin(row_brightness)
            darkest_px_y = y0 + darkest_row + 0.5

            offset_px = darkest_px_y - center_px_y
            offset_grid = offset_px / GRID_SIZE

            results.append({
                'map': Path(json_path).stem,
                'cx': cx, 'cy': cy,
                'ddx': ddx, 'ddy': ddy,
                'type': door_type,
                'orientation': 'horizontal',
                'offset_px': offset_px,
                'offset_grid': offset_grid,
                'tile_height_px': y1 - y0,
                'darkest_brightness': row_brightness[darkest_row],
                'median_brightness': np.median(row_brightness),
            })

    return results


def print_stats(label, offsets_px):
    """Print statistics for a list of pixel offsets."""
    if not offsets_px:
        print(f"  {label}: no data")
        return
    arr = np.array(offsets_px)
    print(f"  {label}: n={len(arr)}, "
          f"avg={arr.mean():.2f}px ({arr.mean()/GRID_SIZE:.4f} grid), "
          f"std={arr.std():.2f}px, "
          f"min={arr.min():.2f}px, max={arr.max():.2f}px, "
          f"median={np.median(arr):.2f}px")


def main():
    parser = argparse.ArgumentParser(
        description="Detect door graphic offset from Watabou PNGs"
    )
    parser.add_argument("--json-dir",
                        default="C:/Users/shini/InstaWall/data/watabou_to_mask/watabou_jsons",
                        help="Directory with Watabou JSON files")
    parser.add_argument("--image-dir",
                        default="C:/Users/shini/InstaWall/data/watabou_to_mask/watabou_images",
                        help="Directory with Watabou PNG files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-door details")
    parser.add_argument("--filter-contrast", type=float, default=20.0,
                        help="Min brightness difference between darkest and median "
                             "to count as a valid detection (default: 20)")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    image_dir = Path(args.image_dir)

    json_files = sorted(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} maps")

    all_results = []
    maps_processed = 0

    for json_file in json_files:
        png_file = image_dir / f"{json_file.stem}.png"
        if not png_file.exists():
            continue

        try:
            results = analyze_map(json_file, png_file)
            all_results.extend(results)
            maps_processed += 1
        except Exception as e:
            print(f"  Error on {json_file.name}: {e}")

    print(f"Processed {maps_processed} maps, {len(all_results)} doors analyzed")
    print()

    # Filter out low-contrast detections (likely no visible door line)
    valid = [r for r in all_results
             if (r['median_brightness'] - r['darkest_brightness']) >= args.filter_contrast]
    rejected = len(all_results) - len(valid)
    print(f"Valid detections (contrast >= {args.filter_contrast}): {len(valid)} "
          f"(rejected {rejected} low-contrast)")
    print()

    if not valid:
        print("No valid detections found!")
        return

    # --- Overall stats ---
    print("=== OVERALL ===")
    all_offsets = [r['offset_px'] for r in valid]
    print_stats("All doors", all_offsets)
    print()

    # --- By orientation ---
    print("=== BY ORIENTATION ===")
    for orient in ['vertical', 'horizontal']:
        offsets = [r['offset_px'] for r in valid if r['orientation'] == orient]
        print_stats(orient, offsets)
    print()

    # --- By dir ---
    print("=== BY DIR (ddx, ddy) ===")
    by_dir = defaultdict(list)
    for r in valid:
        by_dir[(r['ddx'], r['ddy'])].append(r['offset_px'])
    for key in sorted(by_dir.keys()):
        print_stats(f"dir=({key[0]:+d},{key[1]:+d})", by_dir[key])
    print()

    # --- By type ---
    print("=== BY DOOR TYPE ===")
    by_type = defaultdict(list)
    for r in valid:
        by_type[r['type']].append(r['offset_px'])
    for key in sorted(by_type.keys()):
        print_stats(f"type={key}", by_type[key])
    print()

    # --- By dir AND type ---
    print("=== BY DIR x TYPE ===")
    by_dir_type = defaultdict(list)
    for r in valid:
        by_dir_type[(r['ddx'], r['ddy'], r['type'])].append(r['offset_px'])
    for key in sorted(by_dir_type.keys()):
        print_stats(f"dir=({key[0]:+d},{key[1]:+d}) type={key[2]}", by_dir_type[key])
    print()

    # --- Per-map consistency check ---
    print("=== PER-MAP AVERAGES (checking consistency) ===")
    by_map = defaultdict(list)
    for r in valid:
        by_map[r['map']].append(r['offset_px'])
    map_avgs = []
    for name in sorted(by_map.keys()):
        avg = np.mean(by_map[name])
        map_avgs.append(avg)
        if args.verbose:
            print(f"  {name}: n={len(by_map[name])}, avg={avg:.2f}px")
    map_avgs = np.array(map_avgs)
    print(f"  Map-level avg of avgs: {map_avgs.mean():.2f}px, "
          f"std: {map_avgs.std():.2f}px, "
          f"min: {map_avgs.min():.2f}px, max: {map_avgs.max():.2f}px")
    print()

    # --- Verbose per-door output ---
    if args.verbose:
        print("=== PER-DOOR DETAILS ===")
        for r in valid:
            print(f"  {r['map']}: ({r['cx']},{r['cy']}) "
                  f"dir=({r['ddx']:+d},{r['ddy']:+d}) type={r['type']} "
                  f"-> offset={r['offset_px']:.1f}px ({r['offset_grid']:.4f} grid) "
                  f"[dark={r['darkest_brightness']:.0f}, med={r['median_brightness']:.0f}]")


if __name__ == "__main__":
    main()
