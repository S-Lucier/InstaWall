"""
Watabou Coordinate Transformation - Calibrated Version
Fine-tune using visual inspection results
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

    # ========== Calculate transformation ==========
    # Scale X from extent
    scale_x = extent_w / grid_width

    # Scale Y - the dungeon extent includes extra margin
    # Looking at the validation image, the actual content area is smaller
    # Let's try to find the actual content bounds more precisely

    # Find the main room at (-22, -3) which is 5x7 - it's the largest room
    # And the room at (-30, -2) which is 7x5

    # Method: Look at where the actual walls are in the image

    # First, find the topmost and bottommost room content
    # The room at (-22, -3) spans from y=-3 to y=4 (most vertical extent)

    # Using the X scale, we can locate the horizontal position of this room
    room_22_center_x = (-22 + 5/2) * scale_x + (extent_x - grid_min_x * scale_x)

    print(f"\nRoom (-22,-3) predicted center X: {room_22_center_x:.0f}")

    # Scan vertically at this X to find the top and bottom walls
    col_profile = gray[:, int(room_22_center_x)]

    # Find dark pixels (walls)
    wall_threshold = 80
    dark_pixels = col_profile < wall_threshold

    # Find groups of dark pixels (walls)
    changes = np.diff(dark_pixels.astype(int))
    wall_starts = np.where(changes == 1)[0] + 1
    wall_ends = np.where(changes == -1)[0]

    print(f"Found {len(wall_starts)} wall regions in column")

    # The room should be bounded by two significant walls
    # Find the two walls that bound the main content area
    if len(wall_ends) >= 2:
        # Skip the title area (first ~100 pixels might be title)
        valid_walls = [(s, e) for s, e in zip(wall_starts, wall_ends) if s > 100 and e - s > 5]

        if len(valid_walls) >= 2:
            # Top wall of content area
            top_wall = valid_walls[0][1]  # Bottom of first wall
            # Bottom wall - need to find the last significant one
            bottom_wall = valid_walls[-1][0]  # Top of last wall

            print(f"Content walls: top at {top_wall}, bottom at {bottom_wall}")

            content_height = bottom_wall - top_wall
            print(f"Content height: {content_height}")

            # This room spans 7 grid units vertically (-3 to 4)
            # But the grid height is also 7, so this should give us scale_y
            scale_y = content_height / grid_height
            print(f"Scale Y from content: {scale_y:.4f}")

            # Calculate offset_y
            # The top of the room (-22, -3) is at grid y=-3
            # top_wall should correspond to grid y=-3
            # top_wall = (-3) * scale_y + offset_y
            offset_y = top_wall - grid_min_y * scale_y

    else:
        # Fallback: use extent with adjustment
        scale_y = scale_x  # Assume square
        content_center_y = extent_y + extent_h / 2
        grid_center_y = (grid_min_y + grid_max_y) / 2
        offset_y = content_center_y - grid_center_y * scale_y

    # Calculate offset_x
    offset_x = extent_x - grid_min_x * scale_x

    print(f"\nCalculated transformation:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # ========== Verify by checking specific rooms ==========
    print("\n--- Verification ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for room in main_rooms:
        px1 = room['x'] * scale_x + offset_x
        py1 = room['y'] * scale_y + offset_y
        px2 = (room['x'] + room['w']) * scale_x + offset_x
        py2 = (room['y'] + room['h']) * scale_y + offset_y

        # Check if this matches visible content
        # Sample a few pixels inside the predicted room
        cx = int((px1 + px2) / 2)
        cy = int((py1 + py2) / 2)

        if 0 <= cx < img_w and 0 <= cy < img_h:
            center_pixel = gray[cy, cx]
            is_floor = center_pixel > 200  # White = floor

            status = "OK (floor)" if is_floor else "CHECK (not floor)"
            print(f"Room ({room['x']},{room['y']}): center=({cx},{cy}) pixel={center_pixel} {status}")

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

    cv2.imwrite("//debug_calibrated.png", debug_img)
    print("\nSaved debug_calibrated.png")

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
