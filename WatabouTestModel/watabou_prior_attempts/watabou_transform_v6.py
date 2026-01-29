"""
Watabou Coordinate Transformation - Version 6
Use boundary detection to find room edges precisely
"""

import json
import cv2
import numpy as np

def find_room_bounds_precise(gray, approx_cx, approx_cy, expected_w, expected_h, scale_est):
    """
    Find precise room boundaries by scanning for wall edges
    """
    h, w = gray.shape

    # Expected room size in pixels
    exp_w_px = expected_w * scale_est
    exp_h_px = expected_h * scale_est

    # Search window
    search_w = int(exp_w_px * 1.5)
    search_h = int(exp_h_px * 1.5)

    x1 = max(0, int(approx_cx - search_w))
    x2 = min(w, int(approx_cx + search_w))
    y1 = max(0, int(approx_cy - search_h))
    y2 = min(h, int(approx_cy + search_h))

    roi = gray[y1:y2, x1:x2]

    # Find the room by looking for a rectangular light area
    _, floor = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

    # Find the largest connected white region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(floor)

    if num_labels <= 1:
        return None

    # Find the largest component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1

    # Get the bounding box of this component
    bx = stats[largest_idx, cv2.CC_STAT_LEFT] + x1
    by = stats[largest_idx, cv2.CC_STAT_TOP] + y1
    bw = stats[largest_idx, cv2.CC_STAT_WIDTH]
    bh = stats[largest_idx, cv2.CC_STAT_HEIGHT]

    return bx, by, bw, bh


def main():
    json_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.json"
    img_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png"

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data['rects']

    # Load image
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Image size: {img_w} x {img_h}")

    # Get dungeon extent
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = cv2.boundingRect(coords)

    # Grid bounds
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Dungeon pixel extent: ({dungeon_x}, {dungeon_y}) {dungeon_w}x{dungeon_h}")

    # Initial scale estimate
    scale_est = dungeon_w / grid_width  # ~145

    # Main rooms
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    print(f"\n--- Analyzing {len(main_rooms)} main rooms ---")

    # For each room, find its boundaries
    room_data = []

    # Sort rooms by area (start with largest for better detection)
    rooms_by_area = sorted(main_rooms, key=lambda r: r['w'] * r['h'], reverse=True)

    for room in rooms_by_area:
        # Approximate center using initial estimate
        grid_cx = room['x'] + room['w'] / 2
        grid_cy = room['y'] + room['h'] / 2

        # Use extent-based initial estimate
        approx_cx = grid_cx * (dungeon_w / grid_width) + (dungeon_x - grid_min_x * (dungeon_w / grid_width))
        approx_cy = grid_cy * (dungeon_h / grid_height) + (dungeon_y - grid_min_y * (dungeon_h / grid_height))

        bounds = find_room_bounds_precise(gray, approx_cx, approx_cy, room['w'], room['h'], scale_est)

        if bounds:
            bx, by, bw, bh = bounds
            room_data.append({
                'grid': room,
                'grid_x': room['x'],
                'grid_y': room['y'],
                'grid_w': room['w'],
                'grid_h': room['h'],
                'pixel_x': bx,
                'pixel_y': by,
                'pixel_w': bw,
                'pixel_h': bh
            })
            print(f"Room ({room['x']},{room['y']}) {room['w']}x{room['h']} -> pixel ({bx},{by}) {bw}x{bh}")

    print(f"\nFound {len(room_data)} room boundaries")

    # ========== Solve for transformation using corner correspondences ==========
    # For each room, we have 4 corner correspondences:
    # grid (x, y) -> pixel (px, py)
    # grid (x+w, y) -> pixel (px+pw, py)
    # grid (x, y+h) -> pixel (px, py+ph)
    # grid (x+w, y+h) -> pixel (px+pw, py+ph)

    grid_points_x = []
    grid_points_y = []
    pixel_points_x = []
    pixel_points_y = []

    for rd in room_data:
        # Add all 4 corners
        gx, gy = rd['grid_x'], rd['grid_y']
        gw, gh = rd['grid_w'], rd['grid_h']
        px, py = rd['pixel_x'], rd['pixel_y']
        pw, ph = rd['pixel_w'], rd['pixel_h']

        # Top-left
        grid_points_x.append(gx)
        grid_points_y.append(gy)
        pixel_points_x.append(px)
        pixel_points_y.append(py)

        # Top-right
        grid_points_x.append(gx + gw)
        grid_points_y.append(gy)
        pixel_points_x.append(px + pw)
        pixel_points_y.append(py)

        # Bottom-left
        grid_points_x.append(gx)
        grid_points_y.append(gy + gh)
        pixel_points_x.append(px)
        pixel_points_y.append(py + ph)

        # Bottom-right
        grid_points_x.append(gx + gw)
        grid_points_y.append(gy + gh)
        pixel_points_x.append(px + pw)
        pixel_points_y.append(py + ph)

    grid_points_x = np.array(grid_points_x)
    grid_points_y = np.array(grid_points_y)
    pixel_points_x = np.array(pixel_points_x)
    pixel_points_y = np.array(pixel_points_y)

    print(f"\nUsing {len(grid_points_x)} corner correspondences")

    # Solve: pixel = grid * scale + offset
    A_x = np.column_stack([grid_points_x, np.ones(len(grid_points_x))])
    result_x = np.linalg.lstsq(A_x, pixel_points_x, rcond=None)
    scale_x, offset_x = result_x[0]

    A_y = np.column_stack([grid_points_y, np.ones(len(grid_points_y))])
    result_y = np.linalg.lstsq(A_y, pixel_points_y, rcond=None)
    scale_y, offset_y = result_y[0]

    print(f"\nTransformation from corners:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # Calculate residuals
    pred_x = grid_points_x * scale_x + offset_x
    pred_y = grid_points_y * scale_y + offset_y

    residuals_x = pixel_points_x - pred_x
    residuals_y = pixel_points_y - pred_y

    print(f"\nResiduals (pixels):")
    print(f"  X: mean={np.mean(np.abs(residuals_x)):.1f}, std={np.std(residuals_x):.1f}, max={np.max(np.abs(residuals_x)):.1f}")
    print(f"  Y: mean={np.mean(np.abs(residuals_y)):.1f}, std={np.std(residuals_y):.1f}, max={np.max(np.abs(residuals_y)):.1f}")

    # ========== Validation ==========
    print("\n--- Validation ---")

    debug_img = img.copy()

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 10, py1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_v6_validation.png", debug_img)
    print("Saved debug_v6_validation.png")

    # ========== Final Results ==========
    print("\n" + "=" * 60)
    print("FINAL TRANSFORMATION PARAMETERS")
    print("=" * 60)
    print(f"scale_x = {scale_x:.6f}")
    print(f"scale_y = {scale_y:.6f}")
    print(f"offset_x = {offset_x:.6f}")
    print(f"offset_y = {offset_y:.6f}")
    print("\nTransformation formula:")
    print(f"  pixel_x = grid_x * {scale_x:.4f} + {offset_x:.4f}")
    print(f"  pixel_y = grid_y * {scale_y:.4f} + {offset_y:.4f}")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
