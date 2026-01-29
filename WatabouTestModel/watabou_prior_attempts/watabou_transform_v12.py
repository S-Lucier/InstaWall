"""
Watabou Coordinate Transformation - Version 12
Use multiple room measurements to determine scale independently for X and Y
"""

import json
import cv2
import numpy as np

def measure_room_by_wall_scan(gray, center_x, center_y, max_search=500):
    """
    Measure room boundaries by scanning outward from center until hitting walls.
    Uses multiple scan lines for robustness.
    """
    h, w = gray.shape
    cx = int(center_x)
    cy = int(center_y)

    # Scan multiple horizontal lines to find left/right walls
    offsets = [-50, -25, 0, 25, 50]
    left_walls = []
    right_walls = []
    top_walls = []
    bottom_walls = []

    for offset in offsets:
        # Horizontal scan at cy + offset
        scan_y = cy + offset
        if 0 <= scan_y < h:
            # Left scan
            for x in range(cx, max(0, cx - max_search), -1):
                if gray[scan_y, x] < 80:  # Wall found
                    left_walls.append(x)
                    break
            # Right scan
            for x in range(cx, min(w, cx + max_search)):
                if gray[scan_y, x] < 80:
                    right_walls.append(x)
                    break

        # Vertical scan at cx + offset
        scan_x = cx + offset
        if 0 <= scan_x < w:
            # Top scan
            for y in range(cy, max(0, cy - max_search), -1):
                if gray[y, scan_x] < 80:
                    top_walls.append(y)
                    break
            # Bottom scan
            for y in range(cy, min(h, cy + max_search)):
                if gray[y, scan_x] < 80:
                    bottom_walls.append(y)
                    break

    # Use median values
    left = np.median(left_walls) if left_walls else cx - 100
    right = np.median(right_walls) if right_walls else cx + 100
    top = np.median(top_walls) if top_walls else cy - 100
    bottom = np.median(bottom_walls) if bottom_walls else cy + 100

    return int(left), int(top), int(right), int(bottom)


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

    # ========== Get dungeon extent ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    print(f"Dungeon extent: ({extent_x}, {extent_y}) size {extent_w}x{extent_h}")

    # ========== Grid bounds ==========
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # ========== Initial estimate using extent ==========
    scale_x_init = extent_w / grid_width
    scale_y_init = extent_h / grid_height
    offset_x_init = extent_x - grid_min_x * scale_x_init
    offset_y_init = extent_y - grid_min_y * scale_y_init

    print(f"\nInitial estimate from extent:")
    print(f"  scale: ({scale_x_init:.2f}, {scale_y_init:.2f})")
    print(f"  offset: ({offset_x_init:.2f}, {offset_y_init:.2f})")

    # ========== Measure specific rooms ==========
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    print("\n--- Measuring rooms ---")

    measurements = []

    for room in main_rooms:
        # Predict room center using initial estimate
        grid_cx = room['x'] + room['w'] / 2
        grid_cy = room['y'] + room['h'] / 2

        pred_cx = grid_cx * scale_x_init + offset_x_init
        pred_cy = grid_cy * scale_y_init + offset_y_init

        # Measure actual room boundaries
        left, top, right, bottom = measure_room_by_wall_scan(gray, pred_cx, pred_cy)

        actual_w = right - left
        actual_h = bottom - top
        actual_cx = (left + right) / 2
        actual_cy = (top + bottom) / 2

        # Calculate scale from this room
        if room['w'] > 0 and room['h'] > 0:
            room_scale_x = actual_w / room['w']
            room_scale_y = actual_h / room['h']

            # Only trust measurements that are reasonable
            if 80 < room_scale_x < 200 and 80 < room_scale_y < 200:
                measurements.append({
                    'room': room,
                    'grid_cx': grid_cx, 'grid_cy': grid_cy,
                    'pixel_cx': actual_cx, 'pixel_cy': actual_cy,
                    'pixel_w': actual_w, 'pixel_h': actual_h,
                    'scale_x': room_scale_x, 'scale_y': room_scale_y
                })

                print(f"Room ({room['x']},{room['y']}) {room['w']}x{room['h']}:")
                print(f"  Pixel bounds: ({left}, {top}) to ({right}, {bottom})")
                print(f"  Pixel size: {actual_w} x {actual_h}")
                print(f"  Implied scale: ({room_scale_x:.1f}, {room_scale_y:.1f})")

    if len(measurements) < 2:
        print("Not enough valid measurements, using initial estimate")
        scale_x = scale_x_init
        scale_y = scale_y_init
        offset_x = offset_x_init
        offset_y = offset_y_init
    else:
        # Calculate average scale weighted by room area
        total_weight = sum(m['room']['w'] * m['room']['h'] for m in measurements)
        scale_x = sum(m['scale_x'] * m['room']['w'] * m['room']['h'] for m in measurements) / total_weight
        scale_y = sum(m['scale_y'] * m['room']['w'] * m['room']['h'] for m in measurements) / total_weight

        print(f"\nWeighted average scale: ({scale_x:.2f}, {scale_y:.2f})")

        # Calculate offset using all measurements (least squares)
        grid_cx_arr = np.array([m['grid_cx'] for m in measurements])
        grid_cy_arr = np.array([m['grid_cy'] for m in measurements])
        pixel_cx_arr = np.array([m['pixel_cx'] for m in measurements])
        pixel_cy_arr = np.array([m['pixel_cy'] for m in measurements])

        # pixel = grid * scale + offset
        # offset = mean(pixel - grid * scale)
        offset_x = np.mean(pixel_cx_arr - grid_cx_arr * scale_x)
        offset_y = np.mean(pixel_cy_arr - grid_cy_arr * scale_y)

        print(f"Calculated offset: ({offset_x:.2f}, {offset_y:.2f})")

        # Verify
        pred_x = grid_cx_arr * scale_x + offset_x
        pred_y = grid_cy_arr * scale_y + offset_y
        errors = np.sqrt((pixel_cx_arr - pred_x)**2 + (pixel_cy_arr - pred_y)**2)
        print(f"Prediction errors: mean={np.mean(errors):.1f}, max={np.max(errors):.1f}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        label = f"{room['x']},{room['y']}"
        cv2.putText(debug_img, label, (px1 + 5, py1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Also draw the measured room boundaries for comparison
    for m in measurements:
        room = m['room']
        left, top, right, bottom = measure_room_by_wall_scan(gray,
            m['grid_cx'] * scale_x + offset_x,
            m['grid_cy'] * scale_y + offset_y)
        cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imwrite("//debug_v12_validation.png", debug_img)
    print("\nSaved debug_v12_validation.png")

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

    # Test points
    print("\n--- Test points ---")
    for gx, gy, desc in [(-30, -2, "Ending room TL"), (-22, -3, "Big room TL"), (-3, -2, "Entry room TL")]:
        px = gx * scale_x + offset_x
        py = gy * scale_y + offset_y
        print(f"  Grid ({gx}, {gy}) '{desc}' -> Pixel ({px:.0f}, {py:.0f})")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
