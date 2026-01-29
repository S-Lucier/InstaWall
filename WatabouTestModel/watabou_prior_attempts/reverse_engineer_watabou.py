"""
Reverse-engineer Watabou's rendering by analyzing the actual PNG.

Strategy:
1. Detect which rooms are circular vs rectangular in the PNG
2. Find the exact grid size and offset by analyzing room positions
3. Generate perfectly aligned masks
"""

import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


def detect_room_shapes_from_png(png_path, json_path, debug=False):
    """
    Analyze the PNG to determine which rooms are rendered as circular.

    Returns:
        dict mapping room index to shape ('circular' or 'rectangular')
    """
    # Load PNG and JSON
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])

    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Filter to significant contours (rooms)
    min_contour_area = 5000  # Adjust based on image size
    room_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    print(f"Found {len(room_contours)} room contours in PNG")

    room_shapes = {}
    for i, rect in enumerate(rects):
        # Skip 1x1 rects (doors/connections)
        if rect['w'] == 1 and rect['h'] == 1:
            room_shapes[i] = 'connection'
            continue

        # For actual rooms, check if nearly square
        aspect_ratio = max(rect['w'], rect['h']) / min(rect['w'], rect['h'])

        # Heuristic: If aspect ratio close to 1, likely rendered as circular
        if aspect_ratio < 1.3 and min(rect['w'], rect['h']) >= 3:
            room_shapes[i] = 'circular'
        else:
            room_shapes[i] = 'rectangular'

    if debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, room_contours, -1, (0, 255, 0), 2)
        debug_dir = Path(png_path).parent / "debug_reverse"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{Path(png_path).stem}_contours.png"), debug_img)

    return room_shapes


def find_exact_grid_mapping(png_path, json_path, debug=False):
    """
    Find the exact grid size and offset by analyzing room positions in the PNG.

    This reverse-engineers Watabou's coordinate transformation.
    """
    # Load data
    img = cv2.imread(str(png_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])

    # Find grid lines in the PNG (Watabou dungeons have visible grid)
    edges = cv2.Canny(gray, 30, 100)

    # Detect horizontal and vertical lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is None:
        print("Warning: Could not detect grid lines")
        return None

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Check if horizontal (small y difference)
        if abs(y2 - y1) < 10:
            h_lines.append((y1 + y2) // 2)  # Average y position
        # Check if vertical (small x difference)
        elif abs(x2 - x1) < 10:
            v_lines.append((x1 + x2) // 2)  # Average x position

    # Find most common spacing between grid lines
    if h_lines and v_lines:
        h_lines.sort()
        v_lines.sort()

        # Calculate spacing between consecutive lines
        h_spacings = [h_lines[i+1] - h_lines[i] for i in range(len(h_lines)-1)]
        v_spacings = [v_lines[i+1] - v_lines[i] for i in range(len(v_lines)-1)]

        if h_spacings and v_spacings:
            # Most common spacing is likely the grid size
            grid_size_y = np.median(h_spacings)
            grid_size_x = np.median(v_spacings)

            print(f"Detected grid size: X={grid_size_x:.1f}, Y={grid_size_y:.1f}")

            # Find the offset (where grid (0,0) would be)
            # Use the first room to calibrate
            first_room = None
            for rect in rects:
                if rect['w'] > 1 and rect['h'] > 1:  # Skip 1x1 connections
                    first_room = rect
                    break

            if first_room:
                # Find the bounding box of this room in the PNG
                # This is tricky - we'd need to match contours to rooms
                # For now, use a simpler approach based on min coordinates

                min_x = min(r['x'] for r in rects)
                min_y = min(r['y'] for r in rects)

                # The offset should place min_x, min_y at some position in the PNG
                # We need to find where the actual dungeon starts

                # Use edge density to find dungeon boundaries
                h_density = np.sum(edges, axis=1)
                v_density = np.sum(edges, axis=0)

                # Find first significant edge content
                threshold = np.max(h_density) * 0.1
                y_start = np.argmax(h_density > threshold)
                y_end = len(h_density) - np.argmax(h_density[::-1] > threshold)

                threshold = np.max(v_density) * 0.1
                x_start = np.argmax(v_density > threshold)
                x_end = len(v_density) - np.argmax(v_density[::-1] > threshold)

                return {
                    'grid_size_x': grid_size_x,
                    'grid_size_y': grid_size_y,
                    'offset_x': x_start,
                    'offset_y': y_start,
                    'dungeon_width': x_end - x_start,
                    'dungeon_height': y_end - y_start,
                    'min_grid_x': min_x,
                    'min_grid_y': min_y
                }

    return None


def generate_exact_mask(json_path, png_path, debug=False):
    """
    Generate a mask that exactly matches the PNG dimensions and alignment.

    Uses reverse-engineered grid mapping.
    """
    # Get grid mapping
    mapping = find_exact_grid_mapping(png_path, json_path, debug)

    if not mapping:
        print("Could not determine exact grid mapping, using fallback")
        return None

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])

    # Detect room shapes
    room_shapes = detect_room_shapes_from_png(png_path, json_path, debug)

    # Load PNG for dimensions
    img = Image.open(png_path)

    # Create mask
    mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # Fill rooms using exact mapping
    for i, rect in enumerate(rects):
        # Skip connections
        if room_shapes.get(i) == 'connection':
            continue

        # Calculate pixel coordinates
        px1 = int((rect['x'] - mapping['min_grid_x']) * mapping['grid_size_x'] + mapping['offset_x'])
        py1 = int((rect['y'] - mapping['min_grid_y']) * mapping['grid_size_y'] + mapping['offset_y'])
        px2 = int((rect['x'] + rect['w'] - mapping['min_grid_x']) * mapping['grid_size_x'] + mapping['offset_x'])
        py2 = int((rect['y'] + rect['h'] - mapping['min_grid_y']) * mapping['grid_size_y'] + mapping['offset_y'])

        # Clamp to image bounds
        px1, px2 = max(0, px1), min(img.width, px2)
        py1, py2 = max(0, py1), min(img.height, py2)

        # Render based on detected shape
        if room_shapes.get(i) == 'circular':
            center_x = (px1 + px2) // 2
            center_y = (py1 + py2) // 2
            radius_x = (px2 - px1) // 2
            radius_y = (py2 - py1) // 2
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
        else:
            mask[py1:py2, px1:px2] = 255

    if debug:
        debug_dir = Path(png_path).parent / "debug_reverse"
        debug_dir.mkdir(exist_ok=True)
        base_name = Path(png_path).stem

        # Save mask
        Image.fromarray(mask).save(debug_dir / f"{base_name}_exact_mask.png")

        # Create overlay
        img_cv = cv2.imread(str(png_path))
        overlay = img_cv.copy()
        overlay[mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(debug_dir / f"{base_name}_exact_overlay.png"), blended)

        print(f"Debug output saved for {base_name}")
        print(f"Mapping: {mapping}")

    return mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Reverse-engineer Watabou rendering")
    parser.add_argument("png_path", help="Path to PNG file")
    parser.add_argument("json_path", help="Path to JSON file")
    parser.add_argument("--debug", action="store_true", help="Save debug visualizations")

    args = parser.parse_args()

    # Analyze
    print("Analyzing room shapes...")
    shapes = detect_room_shapes_from_png(args.png_path, args.json_path, args.debug)

    print("\nRoom shapes detected:")
    for i, shape in shapes.items():
        print(f"  Room {i}: {shape}")

    print("\nFinding exact grid mapping...")
    mapping = find_exact_grid_mapping(args.png_path, args.json_path, args.debug)

    if mapping:
        print("\nGrid mapping:")
        for k, v in mapping.items():
            print(f"  {k}: {v}")

    print("\nGenerating exact mask...")
    mask = generate_exact_mask(args.json_path, args.png_path, args.debug)

    if mask is not None:
        print("Success! Check debug_reverse/ folder for results")
