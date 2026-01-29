"""
Watabou Coordinate Transformation - Version 8
Manual measurement of specific room dimensions
"""

import json
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

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

    # ========== Find the prominent circle (anchor) ==========
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=80, maxRadius=300)

    circles = np.uint16(np.around(circles))
    largest = circles[0][np.argmax(circles[0][:, 2])]
    circle_cx, circle_cy, circle_r = largest
    print(f"Circle at pixel ({circle_cx}, {circle_cy}) radius={circle_r}")

    # ========== Measure the room containing the circle ==========
    # This room is at grid (-22, -3) with size 5x7
    # Find its walls by scanning outward from the circle

    print("\n--- Measuring room with circle ---")

    # Scan left from circle center to find left wall
    scan_y = circle_cy
    left_scan = gray[scan_y, :circle_cx]
    left_dark = np.where(left_scan < 80)[0]
    if len(left_dark) > 0:
        room_left = left_dark[-1]  # Rightmost dark pixel to the left
    else:
        room_left = 0

    # Scan right
    right_scan = gray[scan_y, circle_cx:]
    right_dark = np.where(right_scan < 80)[0]
    if len(right_dark) > 0:
        room_right = circle_cx + right_dark[0]  # Leftmost dark pixel to the right
    else:
        room_right = img_w

    # Scan up
    up_scan = gray[:circle_cy, circle_cx]
    up_dark = np.where(up_scan < 80)[0]
    if len(up_dark) > 0:
        room_top = up_dark[-1]
    else:
        room_top = 0

    # Scan down
    down_scan = gray[circle_cy:, circle_cx]
    down_dark = np.where(down_scan < 80)[0]
    if len(down_dark) > 0:
        room_bottom = circle_cy + down_dark[0]
    else:
        room_bottom = img_h

    room_pixel_w = room_right - room_left
    room_pixel_h = room_bottom - room_top

    print(f"Room bounds: ({room_left}, {room_top}) to ({room_right}, {room_bottom})")
    print(f"Room pixel size: {room_pixel_w} x {room_pixel_h}")

    # This room has grid size 5x7
    grid_room_w = 5
    grid_room_h = 7

    scale_x_from_room = room_pixel_w / grid_room_w
    scale_y_from_room = room_pixel_h / grid_room_h

    print(f"Scale from room: x={scale_x_from_room:.2f}, y={scale_y_from_room:.2f}")

    # ========== Measure the ending room (has columns) ==========
    # This room is at grid (-30, -2) with size 7x5
    # It's the leftmost room

    print("\n--- Measuring ending room (leftmost) ---")

    # Find the leftmost significant dark structure
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find leftmost column with significant dark pixels
    col_sums = np.sum(dark, axis=0)
    threshold = img_h * 50  # Significant if 50+ dark pixels

    left_wall_cols = np.where(col_sums > threshold)[0]
    if len(left_wall_cols) > 0:
        dungeon_left_x = left_wall_cols[0]
        print(f"Dungeon starts at x={dungeon_left_x}")

    # Find rightmost
    right_wall_cols = np.where(col_sums > threshold)[0]
    if len(right_wall_cols) > 0:
        dungeon_right_x = right_wall_cols[-1]
        print(f"Dungeon ends at x={dungeon_right_x}")

    dungeon_pixel_width = dungeon_right_x - dungeon_left_x

    # Grid width (including corridors)
    grid_min_x = min(r['x'] for r in rects)  # -31
    grid_max_x = max(r['x'] + r['w'] for r in rects)  # 1
    grid_width = grid_max_x - grid_min_x  # 32

    scale_x_from_extent = dungeon_pixel_width / grid_width
    print(f"Dungeon width: {dungeon_pixel_width} pixels, grid width: {grid_width}")
    print(f"Scale X from extent: {scale_x_from_extent:.2f}")

    # Do the same for height
    row_sums = np.sum(dark, axis=1)
    top_rows = np.where(row_sums > threshold)[0]
    if len(top_rows) > 0:
        dungeon_top_y = top_rows[0]
        dungeon_bottom_y = top_rows[-1]
        print(f"Dungeon Y range: {dungeon_top_y} to {dungeon_bottom_y}")

    dungeon_pixel_height = dungeon_bottom_y - dungeon_top_y

    grid_min_y = min(r['y'] for r in rects)  # -3
    grid_max_y = max(r['y'] + r['h'] for r in rects)  # 4
    grid_height = grid_max_y - grid_min_y  # 7

    scale_y_from_extent = dungeon_pixel_height / grid_height
    print(f"Dungeon height: {dungeon_pixel_height} pixels, grid height: {grid_height}")
    print(f"Scale Y from extent: {scale_y_from_extent:.2f}")

    # ========== Use average of measurements ==========
    print("\n--- Computing final scale ---")

    # Weight the extent measurement more (it spans the full dungeon)
    scale_x = (scale_x_from_room * 0.3 + scale_x_from_extent * 0.7)
    scale_y = (scale_y_from_room * 0.3 + scale_y_from_extent * 0.7)

    print(f"Weighted scale: x={scale_x:.2f}, y={scale_y:.2f}")

    # The scale should be roughly the same for x and y (square grid)
    # Let's check if they're close
    scale_ratio = scale_x / scale_y
    print(f"Scale ratio X/Y: {scale_ratio:.3f}")

    # If scales are very different, something is wrong
    # Let's use the extent-based measurement which is more reliable
    if abs(scale_ratio - 1.0) > 0.2:
        print("Scales differ significantly, using extent-based values")
        scale_x = scale_x_from_extent
        scale_y = scale_y_from_extent

    # ========== Calculate offset using the circle as anchor ==========
    # Circle is at grid (-19.5, 0.5) approximately (center of room at -22,-3 size 5x7)
    circle_grid_x = -22 + 5/2  # -19.5
    circle_grid_y = -3 + 7/2   # 0.5

    offset_x = circle_cx - circle_grid_x * scale_x
    offset_y = circle_cy - circle_grid_y * scale_y

    print(f"\nUsing circle anchor:")
    print(f"  Circle at pixel ({circle_cx}, {circle_cy})")
    print(f"  Circle at grid ({circle_grid_x}, {circle_grid_y})")
    print(f"  offset_x = {offset_x:.2f}")
    print(f"  offset_y = {offset_y:.2f}")

    # ========== Verify with dungeon extent ==========
    pred_left = grid_min_x * scale_x + offset_x
    pred_right = grid_max_x * scale_x + offset_x
    pred_top = grid_min_y * scale_y + offset_y
    pred_bottom = grid_max_y * scale_y + offset_y

    print(f"\nPredicted extent: x=[{pred_left:.0f}, {pred_right:.0f}], y=[{pred_top:.0f}, {pred_bottom:.0f}]")
    print(f"Actual extent: x=[{dungeon_left_x}, {dungeon_right_x}], y=[{dungeon_top_y}, {dungeon_bottom_y}]")

    # Adjust offset if there's systematic error
    x_error = (pred_left - dungeon_left_x + pred_right - dungeon_right_x) / 2
    y_error = (pred_top - dungeon_top_y + pred_bottom - dungeon_bottom_y) / 2

    print(f"Systematic error: x={x_error:.1f}, y={y_error:.1f}")

    if abs(x_error) > 20:
        offset_x -= x_error
        print(f"Adjusted offset_x to {offset_x:.2f}")
    if abs(y_error) > 20:
        offset_y -= y_error
        print(f"Adjusted offset_y to {offset_y:.2f}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

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
    cv2.circle(debug_img, (circle_cx, circle_cy), 10, (0, 255, 0), -1)

    # Mark measured room bounds
    cv2.rectangle(debug_img, (room_left, room_top), (room_right, room_bottom), (0, 255, 255), 2)

    cv2.imwrite("//debug_v8_validation.png", debug_img)
    print("\nSaved debug_v8_validation.png")

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

    # Test with specific points
    print("\n--- Test transformations ---")
    test_points = [
        (0, 0, "Grid origin"),
        (-22, -3, "Large room top-left"),
        (-22 + 5, -3 + 7, "Large room bottom-right"),
        (-30, -2, "Ending room top-left"),
    ]
    for gx, gy, desc in test_points:
        px = gx * scale_x + offset_x
        py = gy * scale_y + offset_y
        print(f"  {desc} ({gx}, {gy}) -> ({px:.0f}, {py:.0f})")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
