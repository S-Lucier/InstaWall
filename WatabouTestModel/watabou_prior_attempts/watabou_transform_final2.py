"""
Watabou Coordinate Transformation - Final Version 2
Use the dungeon extent for X and manual Y calibration
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

    print(f"Dungeon extent: ({extent_x}, {extent_y}) size {extent_w}x{extent_h}")

    # ========== Grid bounds ==========
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)
    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # ========== SCALE_X: Use dungeon extent width ==========
    # This is reliable - the dungeon width in pixels divided by grid width
    scale_x = extent_w / grid_width
    print(f"\nScale X from extent: {scale_x:.4f}")

    # ========== SCALE_Y: Try to match actual room heights ==========
    # The extent height includes the title and extra margin
    # Look at the actual dungeon content height

    # Find the actual top and bottom of dungeon content by looking at where
    # significant dark pixels start/end vertically

    # Sum dark pixels per row
    row_dark = np.sum(dark, axis=1)

    # Find rows with significant darkness (walls)
    threshold = img_w * 0.05 * 255  # At least 5% of row is dark
    significant_rows = np.where(row_dark > threshold)[0]

    if len(significant_rows) > 0:
        content_top = significant_rows[0]
        content_bottom = significant_rows[-1]
        content_height = content_bottom - content_top

        print(f"Content vertical range: {content_top} to {content_bottom} (height={content_height})")

        # Try using this height
        scale_y_from_content = content_height / grid_height
        print(f"Scale Y from content height: {scale_y_from_content:.4f}")

    # The grid should be roughly square in Watabou dungeons
    # Let's use scale_x for scale_y as a baseline
    scale_y = scale_x
    print(f"Using square grid assumption: scale_y = {scale_y:.4f}")

    # ========== OFFSET_X: Straightforward from extent ==========
    offset_x = extent_x - grid_min_x * scale_x
    print(f"Offset X: {offset_x:.4f}")

    # ========== OFFSET_Y: Find the vertical center ==========
    # With square scale, we need to find where the grid is vertically centered

    # The dungeon content should be centered in the image
    # Calculate where grid y=0 should be

    # If the dungeon is centered: image center should be at grid center
    # Grid center y = (grid_min_y + grid_max_y) / 2 = 0.5
    # But looking at the image, the dungeon is offset from center

    # Use the content range instead
    if len(significant_rows) > 0:
        content_center_y = (content_top + content_bottom) / 2
        grid_center_y = (grid_min_y + grid_max_y) / 2

        # content_center_y = grid_center_y * scale_y + offset_y
        offset_y = content_center_y - grid_center_y * scale_y
        print(f"Offset Y (from content center): {offset_y:.4f}")
    else:
        offset_y = extent_y - grid_min_y * scale_y
        print(f"Offset Y (from extent): {offset_y:.4f}")

    # ========== Validate and potentially adjust ==========
    print("\n--- Validation ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # Check if the predicted rooms are within the image
    all_within = True
    for room in main_rooms:
        px1 = room['x'] * scale_x + offset_x
        py1 = room['y'] * scale_y + offset_y
        px2 = (room['x'] + room['w']) * scale_x + offset_x
        py2 = (room['y'] + room['h']) * scale_y + offset_y

        if px1 < 0 or px2 > img_w or py1 < 0 or py2 > img_h:
            print(f"Room ({room['x']},{room['y']}) outside image: ({px1:.0f},{py1:.0f})-({px2:.0f},{py2:.0f})")
            all_within = False

    if all_within:
        print("All rooms within image bounds")

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

    cv2.imwrite("//debug_final2_validation.png", debug_img)
    print("\nSaved debug_final2_validation.png")

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
