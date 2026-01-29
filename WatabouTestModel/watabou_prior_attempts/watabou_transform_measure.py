"""
Watabou Coordinate Transformation - Direct Measurement Version
Measure actual room dimensions in the image to determine scale
"""

import json
import cv2
import numpy as np

def find_room_walls(gray, center_x, center_y, direction='horizontal', max_search=600):
    """
    From a center point, scan outward to find the inner edges of walls.
    Returns the distance to wall on each side.
    """
    h, w = gray.shape
    cx, cy = int(center_x), int(center_y)

    if direction == 'horizontal':
        # Scan left
        left_dist = 0
        for x in range(cx, max(0, cx - max_search), -1):
            if gray[cy, x] < 100:  # Hit wall
                left_dist = cx - x
                break

        # Scan right
        right_dist = 0
        for x in range(cx, min(w, cx + max_search)):
            if gray[cy, x] < 100:
                right_dist = x - cx
                break

        return left_dist, right_dist

    else:  # vertical
        # Scan up
        up_dist = 0
        for y in range(cy, max(0, cy - max_search), -1):
            if gray[y, cx] < 100:
                up_dist = cy - y
                break

        # Scan down
        down_dist = 0
        for y in range(cy, min(h, cy + max_search)):
            if gray[y, cx] < 100:
                down_dist = y - cy
                break

        return up_dist, down_dist


def main():
    json_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.json"
    img_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png"

    # Load JSON and image
    with open(json_path, 'r') as f:
        data = json.load(f)
    rects = data['rects']

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    print(f"Image size: {img_w} x {img_h}")

    # Get dungeon extent
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    # Grid bounds
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Dungeon extent: ({extent_x}, {extent_y}) {extent_w}x{extent_h}")
    print(f"Grid: x=[{grid_min_x}, {grid_max_x}]={grid_width}, y=[{grid_min_y}, {grid_max_y}]={grid_height}")

    # ========== Calculate scale_x reliably from extent ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x

    print(f"\nScale X from extent: {scale_x:.4f}")
    print(f"Offset X: {offset_x:.4f}")

    # ========== Measure actual room sizes to determine scale_y ==========
    print("\n--- Measuring actual room dimensions ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # For each room, predict X center (reliable) and search for Y center
    measured_scale_y = []

    for room in main_rooms:
        grid_cx = room['x'] + room['w'] / 2
        grid_cy = room['y'] + room['h'] / 2

        # Predict X position
        pred_cx = grid_cx * scale_x + offset_x

        # For Y, we need to find where the room actually is
        # Scan vertically at pred_cx to find the floor region
        col = gray[:, int(pred_cx)]
        floor_mask = col > 200

        # Find the longest continuous floor region
        changes = np.diff(floor_mask.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0]

        if len(starts) > 0 and len(ends) > 0:
            # Match starts with ends
            best_length = 0
            best_center = 0
            for s in starts:
                for e in ends:
                    if e > s:
                        length = e - s
                        if length > best_length and length < 1500:  # Max room height
                            best_length = length
                            best_center = (s + e) / 2
                        break  # Only consider first end after this start

            if best_length > 100:  # Reasonable room height
                # This floor region corresponds to the room height
                # Room grid height is room['h']
                implied_scale_y = best_length / room['h']

                print(f"Room ({room['x']},{room['y']}) {room['w']}x{room['h']}: floor at y={best_center:.0f}, height={best_length}, implied scale_y={implied_scale_y:.2f}")

                measured_scale_y.append({
                    'room': room,
                    'floor_center': best_center,
                    'floor_height': best_length,
                    'implied_scale_y': implied_scale_y
                })

    if not measured_scale_y:
        print("Could not measure any rooms")
        return

    # ========== Calculate scale_y from measurements ==========
    # Weight by room area (larger rooms give more reliable measurements)
    weights = [m['room']['w'] * m['room']['h'] for m in measured_scale_y]
    scale_y_values = [m['implied_scale_y'] for m in measured_scale_y]

    scale_y = np.average(scale_y_values, weights=weights)
    print(f"\nWeighted average scale_y: {scale_y:.4f}")

    # ========== Calculate offset_y ==========
    # Use the room centers to determine offset
    offset_y_values = []
    for m in measured_scale_y:
        room = m['room']
        grid_cy = room['y'] + room['h'] / 2
        pixel_cy = m['floor_center']
        # pixel_cy = grid_cy * scale_y + offset_y
        implied_offset_y = pixel_cy - grid_cy * scale_y
        offset_y_values.append(implied_offset_y)

    offset_y = np.median(offset_y_values)
    print(f"Median offset_y: {offset_y:.4f}")

    # ========== Verification ==========
    print("\n--- Verification ---")

    all_ok = True
    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

        if 0 <= cx < img_w and 0 <= cy < img_h:
            pixel = gray[cy, cx]
            ok = pixel > 180
            if not ok:
                all_ok = False
            print(f"Room ({room['x']},{room['y']}): center=({cx},{cy}) pixel={pixel} {'OK' if ok else 'FAIL'}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    for room in main_rooms:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"({room['x']},{room['y']})", (px1 + 5, py1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_measure.png", debug_img)
    print("\nSaved debug_measure.png")

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
    print(f"\nScale ratio Y/X: {scale_y/scale_x:.4f}")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
