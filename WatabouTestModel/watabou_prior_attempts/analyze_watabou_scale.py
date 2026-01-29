"""
Analyze Watabou dungeon exports to determine the correct scaling between JSON and PNG.
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def find_dungeon_region(png_path, debug=False):
    """
    Find the actual dungeon map region within the PNG (excluding title, borders, etc).

    Returns:
        (x, y, width, height) of the dungeon region
    """
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The dungeon region has texture/detail, while title/border areas are more uniform
    # Use edge detection to find the textured region
    edges = cv2.Canny(gray, 50, 150)

    # Find contours of high-detail regions
    kernel = np.ones((20, 20), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)

    # Find the largest contiguous region (the dungeon)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: use the whole image minus some margin
        h, w = img.shape[:2]
        margin = 50
        return (margin, margin, w - 2*margin, h - 2*margin)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if debug:
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 5)
        debug_dir = Path(png_path).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{Path(png_path).stem}_region.png"), debug_img)

        print(f"Detected dungeon region: x={x}, y={y}, w={w}, h={h}")

    return (x, y, w, h)


def analyze_scaling(png_path, json_path, debug=False):
    """
    Analyze a PNG/JSON pair to determine the grid scaling factor.

    Returns:
        dict with scaling information
    """
    # Load JSON to get grid bounds
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    if not rects:
        return None

    # Find grid coordinate bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    grid_width = max_x - min_x
    grid_height = max_y - min_y

    # Load PNG and find dungeon region
    img = Image.open(png_path)
    region_x, region_y, region_w, region_h = find_dungeon_region(png_path, debug)

    # Calculate grid size (pixels per grid unit)
    grid_size_x = region_w / grid_width if grid_width > 0 else 0
    grid_size_y = region_h / grid_height if grid_height > 0 else 0

    result = {
        'png_path': str(png_path),
        'png_size': (img.width, img.height),
        'dungeon_region': (region_x, region_y, region_w, region_h),
        'grid_bounds': (min_x, min_y, max_x, max_y),
        'grid_size': (grid_width, grid_height),
        'pixels_per_grid_x': grid_size_x,
        'pixels_per_grid_y': grid_size_y,
        'aspect_ratio_match': abs(grid_size_x - grid_size_y) / max(grid_size_x, grid_size_y) < 0.1
    }

    return result


def batch_analyze(input_dir, debug=False):
    """Analyze all PNG/JSON pairs in a directory."""
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*.json"))

    print(f"Analyzing {len(json_files)} dungeon pairs...\n")

    results = []
    for json_file in json_files:
        png_file = json_file.with_suffix('.png')
        if not png_file.exists():
            continue

        try:
            result = analyze_scaling(png_file, json_file, debug)
            if result:
                results.append(result)
                print(f"{json_file.stem}:")
                print(f"  PNG: {result['png_size']}")
                print(f"  Dungeon region: {result['dungeon_region']}")
                print(f"  Grid bounds: {result['grid_bounds']}")
                print(f"  Grid size: {result['grid_size']}")
                print(f"  Pixels per grid: X={result['pixels_per_grid_x']:.1f}, Y={result['pixels_per_grid_y']:.1f}")
                print(f"  Aspect ratio match: {result['aspect_ratio_match']}")
                print()
        except Exception as e:
            print(f"Error analyzing {json_file.stem}: {e}\n")

    # Summary statistics
    if results:
        avg_grid_x = np.mean([r['pixels_per_grid_x'] for r in results])
        avg_grid_y = np.mean([r['pixels_per_grid_y'] for r in results])
        print(f"\nAverage pixels per grid: X={avg_grid_x:.1f}, Y={avg_grid_y:.1f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Watabou PNG/JSON scaling")
    parser.add_argument("input_dir", help="Directory containing PNG/JSON pairs")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")

    args = parser.parse_args()

    results = batch_analyze(args.input_dir, args.debug)
