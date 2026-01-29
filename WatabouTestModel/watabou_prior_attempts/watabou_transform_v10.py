"""
Watabou Coordinate Transformation - Version 10
More careful circle detection and validation
"""

import json
import cv2
import numpy as np

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

    # ========== Find ALL circles ==========
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    # Look for circles of different sizes
    circles_large = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 200,
                                     param1=50, param2=30, minRadius=100, maxRadius=300)
    circles_medium = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                                      param1=50, param2=30, minRadius=50, maxRadius=150)

    print("\nLarge circles:")
    if circles_large is not None:
        for c in circles_large[0]:
            print(f"  ({c[0]:.0f}, {c[1]:.0f}) r={c[2]:.0f}")

    print("\nMedium circles:")
    if circles_medium is not None:
        for c in circles_medium[0][:10]:
            print(f"  ({c[0]:.0f}, {c[1]:.0f}) r={c[2]:.0f}")

    # ========== Get dungeon extent ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    print(f"\nDungeon extent: ({extent_x}, {extent_y}) to ({extent_x + extent_w}, {extent_y + extent_h})")

    # ========== Grid bounds ==========
    grid_min_x = min(r['x'] for r in rects)  # -31
    grid_max_x = max(r['x'] + r['w'] for r in rects)  # 1
    grid_min_y = min(r['y'] for r in rects)  # -3
    grid_max_y = max(r['y'] + r['h'] for r in rects)  # 4
    grid_width = grid_max_x - grid_min_x  # 32
    grid_height = grid_max_y - grid_min_y  # 7

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")

    # ========== Simple extent-based transformation ==========
    scale_x = extent_w / grid_width
    scale_y = extent_h / grid_height

    # Offset: grid_min maps to extent_min
    offset_x = extent_x - grid_min_x * scale_x
    offset_y = extent_y - grid_min_y * scale_y

    print(f"\nExtent-based transformation:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # ========== Verify by checking if rooms fit ==========
    # The large room at (-22, -3) should contain a circular feature
    # Let's check where that room is predicted to be

    room_22 = {'x': -22, 'y': -3, 'w': 5, 'h': 7}
    room_22_px1 = room_22['x'] * scale_x + offset_x
    room_22_py1 = room_22['y'] * scale_y + offset_y
    room_22_px2 = (room_22['x'] + room_22['w']) * scale_x + offset_x
    room_22_py2 = (room_22['y'] + room_22['h']) * scale_y + offset_y

    room_22_center_x = (room_22_px1 + room_22_px2) / 2
    room_22_center_y = (room_22_py1 + room_22_py2) / 2

    print(f"\nRoom (-22,-3) predicted bounds: ({room_22_px1:.0f}, {room_22_py1:.0f}) to ({room_22_px2:.0f}, {room_22_py2:.0f})")
    print(f"Room (-22,-3) predicted center: ({room_22_center_x:.0f}, {room_22_center_y:.0f})")

    # Find circles near this predicted center
    if circles_medium is not None:
        best_circle = None
        best_dist = float('inf')
        for c in circles_medium[0]:
            dist = np.sqrt((c[0] - room_22_center_x)**2 + (c[1] - room_22_center_y)**2)
            if dist < best_dist and dist < 500:  # Within 500 pixels
                best_dist = dist
                best_circle = c

        if best_circle is not None:
            print(f"Best matching circle: ({best_circle[0]:.0f}, {best_circle[1]:.0f}) r={best_circle[2]:.0f}, dist={best_dist:.0f}")

            # This circle should be at the room center
            # Adjust offset based on this
            error_x = best_circle[0] - room_22_center_x
            error_y = best_circle[1] - room_22_center_y

            print(f"Position error: ({error_x:.0f}, {error_y:.0f})")

            # Adjust offset
            offset_x += error_x
            offset_y += error_y

            print(f"\nAdjusted transformation:")
            print(f"  scale_x = {scale_x:.4f}")
            print(f"  scale_y = {scale_y:.4f}")
            print(f"  offset_x = {offset_x:.4f}")
            print(f"  offset_y = {offset_y:.4f}")

    # ========== Create detailed validation image ==========
    debug_img = img.copy()

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"{room['x']},{room['y']}", (px1 + 5, py1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Mark all detected circles
    if circles_medium is not None:
        for c in circles_medium[0][:10]:
            cv2.circle(debug_img, (int(c[0]), int(c[1])), int(c[2]), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(c[0]), int(c[1])), 5, (0, 255, 0), -1)

    cv2.imwrite("//debug_v10_validation.png", debug_img)
    print("\nSaved debug_v10_validation.png")

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
