"""
Watabou Coordinate Transformation - Version 9
Direct algebraic solution using manually identified reference points
"""

import json
import cv2
import numpy as np

def find_feature_positions(img, gray):
    """Find specific identifiable features in the image"""
    features = {}

    # 1. Find the large circle (most reliable feature)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=80, maxRadius=300)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest = circles[0][np.argmax(circles[0][:, 2])]
        features['circle'] = {'x': int(largest[0]), 'y': int(largest[1]), 'r': int(largest[2])}

    # 2. Find dungeon extent (leftmost and rightmost walls)
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    x, y, w, h = cv2.boundingRect(coords)
    features['dungeon_extent'] = {'left': x, 'top': y, 'right': x + w, 'bottom': y + h}

    # 3. Find the thick outer walls by looking for continuous dark regions
    # Use morphological operations to find thick walls only
    kernel = np.ones((15, 15), np.uint8)
    walls_thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    # Find vertical walls (tall dark regions)
    # Look for columns with lots of dark pixels
    col_dark = np.sum(walls_thick, axis=0)

    # Find significant vertical walls (threshold at 30% of image height)
    h_img = gray.shape[0]
    wall_threshold = h_img * 0.15 * 255

    significant_cols = np.where(col_dark > wall_threshold)[0]

    # Group into wall segments
    if len(significant_cols) > 0:
        walls = []
        start = significant_cols[0]
        for i in range(1, len(significant_cols)):
            if significant_cols[i] - significant_cols[i-1] > 50:  # Gap
                walls.append((start, significant_cols[i-1]))
                start = significant_cols[i]
        walls.append((start, significant_cols[-1]))

        features['vertical_walls'] = walls
        print(f"Found {len(walls)} major vertical wall segments")
        for i, (s, e) in enumerate(walls[:10]):
            print(f"  Wall {i}: x={s} to {e} (width={e-s})")

    return features


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

    # Find features
    features = find_feature_positions(img, gray)

    circle = features.get('circle')
    extent = features.get('dungeon_extent')
    walls = features.get('vertical_walls', [])

    print(f"\nCircle: {circle}")
    print(f"Dungeon extent: {extent}")

    # ========== Grid information ==========
    grid_min_x = min(r['x'] for r in rects)  # -31
    grid_max_x = max(r['x'] + r['w'] for r in rects)  # 1
    grid_min_y = min(r['y'] for r in rects)  # -3
    grid_max_y = max(r['y'] + r['h'] for r in rects)  # 4
    grid_width = grid_max_x - grid_min_x  # 32
    grid_height = grid_max_y - grid_min_y  # 7

    print(f"\nGrid bounds: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # ========== Calculate transformation ==========
    # Using two constraints:
    # 1. Dungeon extent maps to grid extent
    # 2. Circle position refines the offset

    # From dungeon extent:
    dungeon_w = extent['right'] - extent['left']
    dungeon_h = extent['bottom'] - extent['top']

    scale_x = dungeon_w / grid_width
    scale_y = dungeon_h / grid_height

    print(f"\nScale from extent: x={scale_x:.2f}, y={scale_y:.2f}")

    # Initial offset from extent
    offset_x = extent['left'] - grid_min_x * scale_x
    offset_y = extent['top'] - grid_min_y * scale_y

    print(f"Initial offset: x={offset_x:.2f}, y={offset_y:.2f}")

    # ========== Refine using circle ==========
    if circle:
        # The circle is in room at grid (-22, -3) with size 5x7
        # Room center: (-19.5, 0.5)
        # But the circle might not be exactly at room center

        # Let's assume circle center is at room center for now
        room_grid_cx = -22 + 5/2  # -19.5
        room_grid_cy = -3 + 7/2   # 0.5

        # Predicted position of room center
        pred_cx = room_grid_cx * scale_x + offset_x
        pred_cy = room_grid_cy * scale_y + offset_y

        # Actual circle position
        actual_cx = circle['x']
        actual_cy = circle['y']

        print(f"\nCircle refinement:")
        print(f"  Room center grid: ({room_grid_cx}, {room_grid_cy})")
        print(f"  Predicted pixel: ({pred_cx:.0f}, {pred_cy:.0f})")
        print(f"  Actual circle: ({actual_cx}, {actual_cy})")

        # Error
        error_x = actual_cx - pred_cx
        error_y = actual_cy - pred_cy
        print(f"  Error: ({error_x:.0f}, {error_y:.0f})")

        # Adjust offset (split the error - half from offset, half accepted as noise)
        offset_x += error_x * 0.5
        offset_y += error_y * 0.5

        print(f"Adjusted offset: x={offset_x:.2f}, y={offset_y:.2f}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    print("\n--- Room positions ---")
    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        print(f"Room {i}: grid({room['x']},{room['y']}) {room['w']}x{room['h']} -> pixel({px1},{py1})-({px2},{py2})")

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 10, py1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Mark circle
    if circle:
        cv2.circle(debug_img, (circle['x'], circle['y']), 10, (0, 255, 0), -1)

    cv2.imwrite("//debug_v9_validation.png", debug_img)
    print("\nSaved debug_v9_validation.png")

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
