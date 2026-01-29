"""
Watabou Coordinate Transformation - FINAL VERSION

Based on analysis of chapel_of_supremus.png and chapel_of_supremus.json:

The transformation maps grid coordinates to pixel coordinates:
  pixel_x = grid_x * scale_x + offset_x
  pixel_y = grid_y * scale_y + offset_y

Key findings:
1. Scale X is determined by dungeon width / grid width = 4650 / 32 = 145.3125
2. Scale Y equals Scale X (square grid cells)
3. Offset X aligns left edge of grid with left edge of dungeon
4. Offset Y positions the grid vertically so room centers are on floor pixels
"""

import json
import cv2
import numpy as np

def main():
    json_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.json"
    img_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png"

    # Load data
    with open(json_path, 'r') as f:
        data = json.load(f)
    rects = data['rects']

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    print("=" * 60)
    print("WATABOU COORDINATE TRANSFORMATION ANALYSIS")
    print("=" * 60)
    print(f"\nImage: {img_path}")
    print(f"Image size: {img_w} x {img_h} pixels")

    # ========== Dungeon Extent ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    print(f"\nDungeon pixel extent:")
    print(f"  Left: {extent_x}, Top: {extent_y}")
    print(f"  Width: {extent_w}, Height: {extent_h}")

    # ========== Grid Bounds ==========
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"\nGrid bounds (from JSON):")
    print(f"  X: [{grid_min_x}, {grid_max_x}] = {grid_width} units")
    print(f"  Y: [{grid_min_y}, {grid_max_y}] = {grid_height} units")

    # ========== Calculate Transformation ==========
    print("\n" + "-" * 60)
    print("TRANSFORMATION CALCULATION")
    print("-" * 60)

    # Scale X: pixels per grid unit (horizontal)
    scale_x = extent_w / grid_width
    print(f"\nScale X = dungeon_width / grid_width")
    print(f"        = {extent_w} / {grid_width}")
    print(f"        = {scale_x:.6f} pixels per grid unit")

    # Scale Y: Use square grid assumption
    scale_y = scale_x
    print(f"\nScale Y = Scale X (square grid assumption)")
    print(f"        = {scale_y:.6f} pixels per grid unit")

    # Offset X
    offset_x = extent_x - grid_min_x * scale_x
    print(f"\nOffset X = extent_x - grid_min_x * scale_x")
    print(f"         = {extent_x} - ({grid_min_x}) * {scale_x:.4f}")
    print(f"         = {offset_x:.6f}")

    # Offset Y: Find optimal vertical positioning
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    print("\nOffset Y: Optimizing to place room centers on floor...")

    # Find the vertical range where actual dungeon content exists
    # by looking for rows with significant wall (dark) content
    row_dark = np.sum(dark, axis=1)
    content_rows = np.where(row_dark > extent_w * 0.1 * 255)[0]
    if len(content_rows) > 0:
        content_top = content_rows[0]
        content_bottom = content_rows[-1]
        print(f"  Content area: y={content_top} to {content_bottom}")
    else:
        content_top = extent_y
        content_bottom = extent_y + extent_h

    # Search within the content area for best offset_y
    # The grid center y = 0.5 should map to somewhere in the content area
    best_offset_y = None
    best_score = -1

    for offset_y_test in range(900, 1100, 5):
        # For each offset, check:
        # 1. Room centers are on floor (white) pixels
        # 2. Room centers are within content area
        score = 0
        for room in main_rooms:
            cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
            cy = int((room['y'] + room['h']/2) * scale_y + offset_y_test)

            if content_top < cy < content_bottom:  # Within content area
                if 0 <= cx < img_w and 0 <= cy < img_h and gray[cy, cx] > 200:
                    score += 1

        if score > best_score:
            best_score = score
            best_offset_y = offset_y_test

    offset_y = best_offset_y if best_offset_y else 970  # fallback
    print(f"         = {offset_y:.6f} ({best_score}/{len(main_rooms)} rooms matched)")

    # ========== Final Parameters ==========
    print("\n" + "=" * 60)
    print("FINAL TRANSFORMATION PARAMETERS")
    print("=" * 60)
    print(f"\nscale_x  = {scale_x:.6f}")
    print(f"scale_y  = {scale_y:.6f}")
    print(f"offset_x = {offset_x:.6f}")
    print(f"offset_y = {offset_y:.6f}")

    print("\n" + "-" * 60)
    print("TRANSFORMATION FORMULA")
    print("-" * 60)
    print(f"\npixel_x = grid_x * {scale_x:.4f} + {offset_x:.4f}")
    print(f"pixel_y = grid_y * {scale_y:.4f} + {offset_y:.4f}")

    # ========== Verification ==========
    print("\n" + "-" * 60)
    print("VERIFICATION")
    print("-" * 60)

    print("\nRoom center validation:")
    all_ok = True
    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy = int((room['y'] + room['h']/2) * scale_y + offset_y)
        pixel = gray[cy, cx] if 0 <= cy < img_h and 0 <= cx < img_w else 0
        ok = pixel > 180
        if not ok:
            all_ok = False
        print(f"  Room ({room['x']:3d},{room['y']:2d}) {room['w']}x{room['h']}: "
              f"center ({cx:4d},{cy:4d}) -> pixel {pixel:3d} {'OK' if ok else 'FAIL'}")

    print(f"\nAll room centers on floor: {all_ok}")

    # ========== Create Validation Image ==========
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

    output_path = "//watabou_transformation_result.png"
    cv2.imwrite(output_path, debug_img)
    print(f"\nValidation image saved: {output_path}")

    return {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'offset_x': offset_x,
        'offset_y': offset_y
    }


if __name__ == "__main__":
    result = main()
