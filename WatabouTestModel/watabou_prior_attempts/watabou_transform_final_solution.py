"""
Watabou Coordinate Transformation - Final Solution
Combines extent-based X calculation with optimized Y positioning
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

    # ========== SCALE_Y and OFFSET_Y: Optimize to match room centers ==========
    print("\n--- Optimizing Y transformation ---")

    best_scale_y = None
    best_offset_y = None
    best_match_count = 0

    # Try different scale_y ratios (Watabou might use non-square aspect)
    for scale_ratio in np.arange(1.0, 2.5, 0.02):
        scale_y_test = scale_x * scale_ratio

        # Try different offset_y values
        for offset_y_test in np.arange(500, 1500, 10):
            # Count how many room centers land on floor pixels
            match_count = 0
            for room in main_rooms:
                cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
                cy = int((room['y'] + room['h']/2) * scale_y_test + offset_y_test)

                if 0 <= cx < img_w and 0 <= cy < img_h:
                    pixel = gray[cy, cx]
                    if pixel > 200:  # On floor
                        match_count += 1

            if match_count > best_match_count:
                best_match_count = match_count
                best_scale_y = scale_y_test
                best_offset_y = offset_y_test

    scale_y = best_scale_y if best_scale_y else scale_x
    offset_y = best_offset_y if best_offset_y else 1000

    print(f"Best scale_y: {scale_y:.4f} (ratio: {scale_y/scale_x:.2f})")
    print(f"Best offset_y: {offset_y:.4f}")
    print(f"Rooms matched: {best_match_count}/{len(main_rooms)}")

    # ========== Fine-tune offset_y ==========
    # Adjust so room centers are as centered on floor as possible
    adjustments = []
    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

        if 0 <= cx < img_w:
            # Find the center of floor region in this column
            col = gray[:, cx]
            floor_pixels = np.where(col > 200)[0]
            if len(floor_pixels) > 50:
                floor_center = np.mean(floor_pixels)
                adjustment = floor_center - cy
                adjustments.append(adjustment)

    if adjustments:
        median_adj = np.median(adjustments)
        if abs(median_adj) < 200:  # Reasonable adjustment
            offset_y += median_adj
            print(f"Fine-tuned offset_y by {median_adj:.1f} to {offset_y:.4f}")

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
            print(f"Room ({room['x']:3d},{room['y']:2d}) {room['w']}x{room['h']}: center=({cx:4d},{cy:4d}) pixel={pixel:3d} {'OK' if ok else 'FAIL'}")

    print(f"\nAll rooms OK: {all_ok}")

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

    output_path = "//debug_final_solution.png"
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
