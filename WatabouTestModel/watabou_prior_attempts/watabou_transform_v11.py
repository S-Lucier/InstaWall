"""
Watabou Coordinate Transformation - Version 11
Use square grid assumption and manual room boundary detection
"""

import json
import cv2
import numpy as np

def find_actual_room_extent(gray, approx_cx, approx_cy, search_radius=400):
    """Find actual room boundaries by scanning for walls"""
    h, w = gray.shape

    # Ensure we're in bounds
    cx = int(np.clip(approx_cx, search_radius, w - search_radius))
    cy = int(np.clip(approx_cy, search_radius, h - search_radius))

    # Scan from center outward in each direction to find walls
    # Left
    left_scan = gray[cy, max(0, cx - search_radius):cx]
    left_walls = np.where(left_scan < 80)[0]
    left = left_walls[-1] + (cx - search_radius) if len(left_walls) > 0 else cx - search_radius

    # Right
    right_scan = gray[cy, cx:min(w, cx + search_radius)]
    right_walls = np.where(right_scan < 80)[0]
    right = right_walls[0] + cx if len(right_walls) > 0 else cx + search_radius

    # Top
    top_scan = gray[max(0, cy - search_radius):cy, cx]
    top_walls = np.where(top_scan < 80)[0]
    top = top_walls[-1] + (cy - search_radius) if len(top_walls) > 0 else cy - search_radius

    # Bottom
    bottom_scan = gray[cy:min(h, cy + search_radius), cx]
    bottom_walls = np.where(bottom_scan < 80)[0]
    bottom = bottom_walls[0] + cy if len(bottom_walls) > 0 else cy + search_radius

    return left, top, right, bottom


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

    # ========== Get dungeon extent (for X scale) ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    print(f"Dungeon extent: ({extent_x}, {extent_y}) size {extent_w}x{extent_h}")

    # ========== Grid bounds ==========
    grid_min_x = min(r['x'] for r in rects)  # -31
    grid_max_x = max(r['x'] + r['w'] for r in rects)  # 1
    grid_min_y = min(r['y'] for r in rects)  # -3
    grid_max_y = max(r['y'] + r['h'] for r in rects)  # 4
    grid_width = grid_max_x - grid_min_x  # 32
    grid_height = grid_max_y - grid_min_y  # 7

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # ========== Calculate scale_x from dungeon width ==========
    scale_x = extent_w / grid_width
    print(f"\nScale X from extent: {scale_x:.2f}")

    # ========== Assume square grid (scale_y = scale_x) ==========
    # This is a reasonable assumption for Watabou's grid-based system
    scale_y = scale_x
    print(f"Assuming square grid: scale_y = {scale_y:.2f}")

    # ========== Calculate offset ==========
    # Use X offset from extent
    offset_x = extent_x - grid_min_x * scale_x

    # For Y offset, we need to figure out where the grid center is vertically
    # The grid goes from y=-3 to y=4, center at y=0.5
    # Let's find where the rooms actually are in the image

    # First, use the X offset and scale to locate a specific room
    # Room at (-22, -3) is 5x7 - let's find its center X
    room_22_cx = (-22 + 5/2) * scale_x + offset_x  # Grid center X in pixels

    # Search vertically around this X to find the room
    col_profile = gray[:, int(room_22_cx)]
    light_regions = col_profile > 200

    # Find the main light region (the room floor)
    changes = np.diff(light_regions.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0]

    print(f"\nLight regions in column at x={int(room_22_cx)}:")
    if len(starts) > 0 and len(ends) > 0:
        # Find the largest light region
        best_start = 0
        best_end = 0
        best_size = 0
        for s in starts:
            for e in ends:
                if e > s and e - s > best_size:
                    best_size = e - s
                    best_start = s
                    best_end = e

        if best_size > 0:
            room_top_y = best_start
            room_bottom_y = best_end
            room_center_y = (room_top_y + room_bottom_y) / 2
            room_height_px = room_bottom_y - room_top_y

            print(f"  Room spans y={room_top_y} to y={room_bottom_y} (height={room_height_px})")

            # This room is 7 grid units tall
            # room_height_px = 7 * scale_y
            scale_y_from_room = room_height_px / 7
            print(f"  Scale Y from room height: {scale_y_from_room:.2f}")

            # The room center should be at grid y = -3 + 3.5 = 0.5
            # room_center_y = 0.5 * scale_y + offset_y
            # offset_y = room_center_y - 0.5 * scale_y
            offset_y = room_center_y - 0.5 * scale_y
            print(f"  Offset Y from room center: {offset_y:.2f}")

            # Use the measured scale_y if it's close to scale_x
            if abs(scale_y_from_room - scale_x) < scale_x * 0.3:
                print(f"  Using measured scale_y: {scale_y_from_room:.2f}")
                scale_y = scale_y_from_room
                # Recalculate offset_y with new scale
                offset_y = room_center_y - 0.5 * scale_y
    else:
        # Fallback: use center of dungeon extent
        dungeon_center_y = extent_y + extent_h / 2
        grid_center_y = 0.5  # Center of grid y range
        offset_y = dungeon_center_y - grid_center_y * scale_y
        print(f"  Using fallback offset_y: {offset_y:.2f}")

    print(f"\nFinal parameters:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # ========== Measure actual room to verify ==========
    print("\n--- Verifying with room boundaries ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for room in main_rooms[:3]:  # Check first 3 rooms
        grid_cx = room['x'] + room['w'] / 2
        grid_cy = room['y'] + room['h'] / 2

        pred_cx = grid_cx * scale_x + offset_x
        pred_cy = grid_cy * scale_y + offset_y

        left, top, right, bottom = find_actual_room_extent(gray, pred_cx, pred_cy)
        actual_cx = (left + right) / 2
        actual_cy = (top + bottom) / 2
        actual_w = right - left
        actual_h = bottom - top

        pred_w = room['w'] * scale_x
        pred_h = room['h'] * scale_y

        print(f"Room ({room['x']},{room['y']}) {room['w']}x{room['h']}:")
        print(f"  Predicted center: ({pred_cx:.0f}, {pred_cy:.0f})")
        print(f"  Actual center: ({actual_cx:.0f}, {actual_cy:.0f})")
        print(f"  Predicted size: {pred_w:.0f}x{pred_h:.0f}")
        print(f"  Actual size: {actual_w}x{actual_h}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"{room['x']},{room['y']}", (px1 + 5, py1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_v11_validation.png", debug_img)
    print("\nSaved debug_v11_validation.png")

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
