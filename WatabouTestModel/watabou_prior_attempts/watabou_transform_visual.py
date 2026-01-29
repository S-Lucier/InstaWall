"""
Watabou Coordinate Transformation - Visual Measurement
Measure actual room extent by analyzing specific known rooms
"""

import json
import cv2
import numpy as np

def measure_room_extent(gray, pred_cx, search_range=800):
    """
    Find the actual vertical extent of a room by scanning for the top and bottom walls.
    """
    h, w = gray.shape
    cx = int(pred_cx)

    if cx < 0 or cx >= w:
        return None, None

    col = gray[:, cx]

    # Find the main floor region (continuous white area)
    floor_mask = col > 200

    # Find transitions (walls)
    transitions = np.diff(floor_mask.astype(int))
    floor_starts = np.where(transitions == 1)[0] + 1  # Where floor begins
    floor_ends = np.where(transitions == -1)[0]  # Where floor ends

    # Find the largest floor region (main room)
    best_start = 0
    best_end = h
    best_length = 0

    for s in floor_starts:
        for e in floor_ends:
            if e > s and (e - s) > best_length:
                best_length = e - s
                best_start = s
                best_end = e

    if best_length > 100:
        return best_start, best_end

    return None, None


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
    grid_width = grid_max_x - grid_min_x

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # ========== SCALE_X: From dungeon width ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x
    print(f"\nScale X: {scale_x:.4f}")
    print(f"Offset X: {offset_x:.4f}")

    # ========== Measure actual room heights ==========
    print("\n--- Measuring actual room heights ---")

    room_measurements = []

    for room in main_rooms:
        # Predict X center
        pred_cx = (room['x'] + room['w']/2) * scale_x + offset_x

        # Find actual floor extent at this X
        top, bottom = measure_room_extent(gray, pred_cx)

        if top is not None:
            actual_height = bottom - top
            expected_scale_y = actual_height / room['h']

            print(f"Room ({room['x']},{room['y']}) {room['w']}x{room['h']}: floor y={top}-{bottom}, height={actual_height}, implied scale_y={expected_scale_y:.2f}")

            room_measurements.append({
                'room': room,
                'floor_top': top,
                'floor_bottom': bottom,
                'actual_height': actual_height,
                'implied_scale_y': expected_scale_y
            })

    # ========== Calculate scale_y from measurements ==========
    if room_measurements:
        # Use weighted average based on room area
        weights = [m['room']['w'] * m['room']['h'] for m in room_measurements]
        scale_y_values = [m['implied_scale_y'] for m in room_measurements]
        scale_y = np.average(scale_y_values, weights=weights)
        print(f"\nWeighted average scale_y: {scale_y:.4f}")
        print(f"Scale ratio Y/X: {scale_y/scale_x:.4f}")

        # Calculate offset_y
        # Use the room centers
        offset_y_values = []
        for m in room_measurements:
            room = m['room']
            grid_cy = room['y'] + room['h']/2
            floor_center = (m['floor_top'] + m['floor_bottom']) / 2
            implied_offset_y = floor_center - grid_cy * scale_y
            offset_y_values.append(implied_offset_y)

        offset_y = np.median(offset_y_values)
        print(f"Median offset_y: {offset_y:.4f}")
    else:
        scale_y = scale_x
        offset_y = 1000

    # ========== Verify ==========
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
            print(f"Room ({room['x']:3d},{room['y']:2d}): center=({cx:4d},{cy:4d}) pixel={pixel:3d} {'OK' if ok else 'FAIL'}")

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

    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Also draw measured floor extents
    for m in room_measurements:
        room = m['room']
        pred_cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cv2.line(debug_img, (pred_cx-20, m['floor_top']), (pred_cx+20, m['floor_top']), (0, 255, 0), 2)
        cv2.line(debug_img, (pred_cx-20, m['floor_bottom']), (pred_cx+20, m['floor_bottom']), (0, 255, 0), 2)

    output_path = "//debug_visual.png"
    cv2.imwrite(output_path, debug_img)
    print(f"\nSaved {output_path}")

    # ========== Final Results ==========
    print("\n" + "=" * 60)
    print("FINAL TRANSFORMATION PARAMETERS")
    print("=" * 60)
    print(f"scale_x = {scale_x:.6f}")
    print(f"scale_y = {scale_y:.6f}")
    print(f"offset_x = {offset_x:.6f}")
    print(f"offset_y = {offset_y:.6f}")
    print(f"\nScale ratio Y/X: {scale_y/scale_x:.4f}")
    print("\nTransformation formula:")
    print(f"  pixel_x = grid_x * {scale_x:.4f} + {offset_x:.4f}")
    print(f"  pixel_y = grid_y * {scale_y:.4f} + {offset_y:.4f}")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
