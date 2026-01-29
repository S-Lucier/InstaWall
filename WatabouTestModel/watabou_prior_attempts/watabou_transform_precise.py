"""
Watabou Coordinate Transformation - Precise Version
Use direct measurement of visible room features
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

    # ========== Get dungeon extent ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    extent_x, extent_y, extent_w, extent_h = cv2.boundingRect(coords)

    # ========== Grid bounds ==========
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Dungeon extent: ({extent_x}, {extent_y}) size {extent_w}x{extent_h}")
    print(f"Grid: x=[{grid_min_x}, {grid_max_x}]={grid_width}, y=[{grid_min_y}, {grid_max_y}]={grid_height}")

    # ========== Calculate scale_x from extent ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x

    print(f"\nScale X: {scale_x:.4f}")
    print(f"Offset X: {offset_x:.4f}")

    # ========== Find actual Y bounds of rooms ==========
    # The problem is the dungeon extent includes extra vertical margin

    # Find the vertical extent of actual room content by scanning horizontally
    # Look for rows that have significant "room pattern" (alternating light/dark)

    # Sum white pixels per row in the dungeon area
    dungeon_region = gray[extent_y:extent_y+extent_h, extent_x:extent_x+extent_w]
    _, floor_mask = cv2.threshold(dungeon_region, 200, 255, cv2.THRESH_BINARY)

    row_white = np.sum(floor_mask, axis=1) / 255  # Count of white pixels per row

    # Find rows with significant floor content
    threshold = extent_w * 0.15  # At least 15% of width is floor
    floor_rows = np.where(row_white > threshold)[0]

    if len(floor_rows) > 0:
        content_top = floor_rows[0] + extent_y
        content_bottom = floor_rows[-1] + extent_y
        content_height = content_bottom - content_top

        print(f"\nFloor content: y={content_top} to {content_bottom} (height={content_height})")

        # The floor content should span the grid height
        # But it might have some margin
        scale_y_from_content = content_height / grid_height
        print(f"Scale Y from content: {scale_y_from_content:.4f}")
    else:
        scale_y_from_content = scale_x

    # ========== Use square grid assumption ==========
    # Since grid cells are typically square, scale_x = scale_y
    scale_y = scale_x

    print(f"\nUsing square grid: scale_y = {scale_y:.4f}")

    # ========== Calculate offset_y ==========
    # The content should be roughly centered, but we need the exact position

    # Method: Find where the grid center (y=0.5) should be

    # If the grid is 7 units tall (-3 to 4), and scale_y = 145.3
    # Total height = 7 * 145.3 = 1017 pixels

    # The dungeon content height is ~2318 pixels, but actual rooms are ~1017 pixels
    # The extra space is for title, margin, etc.

    # Let's find the vertical center of the floor content
    if len(floor_rows) > 0:
        # Weight by amount of floor in each row to find center
        weights = row_white[floor_rows]
        weighted_center = np.average(floor_rows, weights=weights) + extent_y
        print(f"Weighted center of floor content: {weighted_center:.0f}")

        # This should correspond to grid center y = 0.5
        grid_center_y = (grid_min_y + grid_max_y) / 2  # = 0.5

        offset_y = weighted_center - grid_center_y * scale_y
        print(f"Offset Y from weighted center: {offset_y:.4f}")
    else:
        # Fallback
        offset_y = extent_y + extent_h/2 - (grid_min_y + grid_max_y)/2 * scale_y

    # ========== Fine-tune by checking specific rooms ==========
    print("\n--- Fine-tuning ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # For each room, check if center is on floor and adjust
    adjustments = []
    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy_pred = int((room['y'] + room['h']/2) * scale_y + offset_y)

        if 0 <= cx < img_w:
            # Scan vertically to find the actual floor region
            col = gray[:, cx]
            floor_pixels = np.where(col > 200)[0]

            if len(floor_pixels) > 50:  # At least some floor
                actual_center = np.median(floor_pixels)
                adjustment = actual_center - cy_pred
                adjustments.append(adjustment)
                print(f"Room ({room['x']},{room['y']}): predicted cy={cy_pred}, actual={actual_center:.0f}, adj={adjustment:.0f}")

    if adjustments:
        median_adjustment = np.median(adjustments)
        print(f"\nMedian adjustment: {median_adjustment:.1f}")
        offset_y += median_adjustment
        print(f"Adjusted offset_y: {offset_y:.4f}")

    # ========== Verification ==========
    print("\n--- Verification ---")

    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

        if 0 <= cx < img_w and 0 <= cy < img_h:
            pixel = gray[cy, cx]
            status = "OK" if pixel > 180 else "FAIL"
            print(f"Room ({room['x']},{room['y']}): center=({cx},{cy}) pixel={pixel} {status}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        label = f"({room['x']},{room['y']})"
        cv2.putText(debug_img, label, (px1 + 5, py1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_precise.png", debug_img)
    print("\nSaved debug_precise.png")

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
