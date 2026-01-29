"""
Watabou Coordinate Transformation - Constrained Optimization
Better constraints to find the actual room positions
"""

import json
import cv2
import numpy as np

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

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # ========== SCALE_X: From dungeon width ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x
    print(f"\nScale X: {scale_x:.4f}")
    print(f"Offset X: {offset_x:.4f}")

    # ========== Find actual dungeon content vertical range ==========
    # Look for the area with actual wall structures, not just any dark pixels

    # Find where walls actually start (not title/description)
    # Walls are thick and span significant horizontal distance

    print("\n--- Finding dungeon content bounds ---")

    # For each row, count how many pixels are part of walls (dark)
    row_wall_count = np.sum(dark, axis=1) / 255

    # Find rows with significant wall content (more than 20% of width)
    threshold = extent_w * 0.15
    wall_rows = np.where(row_wall_count > threshold)[0]

    if len(wall_rows) > 0:
        content_top = wall_rows[0]
        content_bottom = wall_rows[-1]
        print(f"Dungeon content Y range: {content_top} to {content_bottom} (height: {content_bottom - content_top})")
    else:
        content_top = extent_y
        content_bottom = extent_y + extent_h

    # ========== SCALE_Y: Assume square grid ==========
    scale_y = scale_x
    print(f"\nScale Y (square grid): {scale_y:.4f}")

    # ========== OFFSET_Y: Center grid in content area ==========
    # Grid height = 7, so total pixel height = 7 * scale_y
    grid_pixel_height = grid_height * scale_y

    # Center this in the content area
    content_center = (content_top + content_bottom) / 2
    grid_center_y = (grid_min_y + grid_max_y) / 2  # = 0.5

    offset_y = content_center - grid_center_y * scale_y
    print(f"Initial offset_y: {offset_y:.4f}")

    # ========== Verify and adjust ==========
    print("\n--- Verification and adjustment ---")

    # Check room centers
    for iteration in range(3):
        match_count = 0
        adjustments = []

        for room in main_rooms:
            cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
            cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

            if 0 <= cx < img_w and 0 <= cy < img_h:
                pixel = gray[cy, cx]
                is_floor = pixel > 200

                if is_floor:
                    match_count += 1
                else:
                    # Try to find floor nearby vertically
                    col = gray[content_top:content_bottom, cx]
                    floor_pixels = np.where(col > 200)[0]
                    if len(floor_pixels) > 0:
                        floor_center = np.median(floor_pixels) + content_top
                        adjustments.append(floor_center - cy)

        print(f"Iteration {iteration}: {match_count}/{len(main_rooms)} rooms OK")

        if match_count == len(main_rooms) or not adjustments:
            break

        # Apply median adjustment
        median_adj = np.median(adjustments)
        offset_y += median_adj
        print(f"  Adjusted offset_y by {median_adj:.1f} to {offset_y:.4f}")

    # ========== Final Verification ==========
    print("\n--- Final Verification ---")
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

    # Draw content bounds for reference
    cv2.line(debug_img, (0, content_top), (img_w, content_top), (0, 255, 0), 1)
    cv2.line(debug_img, (0, content_bottom), (img_w, content_bottom), (0, 255, 0), 1)

    output_path = "//debug_constrained.png"
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
    print("\nTransformation formula:")
    print(f"  pixel_x = grid_x * {scale_x:.4f} + {offset_x:.4f}")
    print(f"  pixel_y = grid_y * {scale_y:.4f} + {offset_y:.4f}")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
