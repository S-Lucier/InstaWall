"""
Watabou Coordinate Transformation - Version 5
Focus on detecting room BOUNDARIES (thick outer walls) not internal grid
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
    print(f"Loaded {len(rects)} rectangles from JSON")

    # Main rooms only
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]
    main_rooms = sorted(main_rooms, key=lambda r: r['x'])

    print("\nMain rooms (sorted by x):")
    for i, r in enumerate(main_rooms):
        print(f"  {i}: ({r['x']}, {r['y']}) {r['w']}x{r['h']}")

    # Load image
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"\nImage size: {img_w} x {img_h}")

    # ========== Detect THICK walls only ==========
    # The room boundaries have thicker/darker walls than the internal grid

    # Find very dark pixels (main walls)
    _, walls = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to get solid wall regions
    # The wall thickness is maybe 10-20 pixels
    kernel = np.ones((5, 5), np.uint8)
    walls_clean = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("//debug_v5_walls.png", walls_clean)

    # ========== Alternative: Use contours on walls ==========
    # Find rectangular contours that represent room boundaries

    # First, let's look at the structure more carefully
    # The rooms are separated by corridors (1x1 in grid units)

    # Let's try to find the scale by measuring the internal grid spacing
    # The internal grid dots are at regular intervals

    # Find the grid spacing by looking at the projection
    print("\n--- Analyzing grid pattern ---")

    # Focus on the middle section where rooms are
    y_start = img_h // 3
    y_end = 2 * img_h // 3
    roi_gray = gray[y_start:y_end, :]

    # Find light pixels (floor)
    _, light = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)

    # Project vertically to find column spacing
    col_proj = np.sum(light, axis=0)

    # Find peaks (white columns between grid lines)
    # The spacing between peaks should be one grid unit

    from scipy.signal import find_peaks

    # Normalize
    col_proj_norm = col_proj / col_proj.max() if col_proj.max() > 0 else col_proj

    # Find valleys (dark columns = wall lines)
    valleys, _ = find_peaks(-col_proj_norm, height=-0.5, distance=30)

    print(f"Found {len(valleys)} vertical grid lines")

    if len(valleys) > 2:
        spacings = np.diff(valleys)
        median_spacing = np.median(spacings)
        print(f"Vertical spacings: min={spacings.min():.0f}, max={spacings.max():.0f}, median={median_spacing:.0f}")

        # The median spacing is likely one grid unit in pixels
        grid_unit_x = median_spacing
        print(f"Estimated grid unit (X): {grid_unit_x:.1f} pixels")

    # Do the same for horizontal
    row_proj = np.sum(light, axis=1)
    row_proj_norm = row_proj / row_proj.max() if row_proj.max() > 0 else row_proj

    h_valleys, _ = find_peaks(-row_proj_norm, height=-0.5, distance=30)

    print(f"Found {len(h_valleys)} horizontal grid lines in ROI")

    if len(h_valleys) > 2:
        h_spacings = np.diff(h_valleys)
        median_h_spacing = np.median(h_spacings)
        print(f"Horizontal spacings: min={h_spacings.min():.0f}, max={h_spacings.max():.0f}, median={median_h_spacing:.0f}")
        grid_unit_y = median_h_spacing
        print(f"Estimated grid unit (Y): {grid_unit_y:.1f} pixels")

    # ========== Find specific room features ==========
    print("\n--- Finding specific rooms ---")

    # The ending room (x=-30) has a distinctive pattern with columns
    # The main room (x=-22) has a circular feature
    # Let's find these by template matching or distinctive features

    # Instead, let's use the wall structure
    # Find contours of wall regions

    # Threshold to get all dark pixels (walls + lines)
    _, all_dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find dungeon extent
    coords = cv2.findNonZero(all_dark)
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = cv2.boundingRect(coords)
    print(f"Dungeon extent: ({dungeon_x}, {dungeon_y}) {dungeon_w}x{dungeon_h}")

    # ========== Use the known room positions ==========
    # We know from JSON that:
    # - Leftmost room: x=-30, 7x5
    # - Rightmost room: x=-3, 3x5
    # - Total grid span: -31 to 1 = 32 units (including corridors)

    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)

    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Grid span: x=[{grid_min_x}, {grid_max_x}]={grid_width}, y=[{grid_min_y}, {grid_max_y}]={grid_height}")

    # ========== Estimate scale from dungeon extent ==========
    # Assuming the dungeon pixel extent corresponds to the grid extent

    scale_x_estimate = dungeon_w / grid_width
    scale_y_estimate = dungeon_h / grid_height

    print(f"Scale from extent: x={scale_x_estimate:.2f}, y={scale_y_estimate:.2f}")

    # If we found grid unit from spacing analysis, use that
    if 'grid_unit_x' in dir():
        scale_x = grid_unit_x
    else:
        scale_x = scale_x_estimate

    if 'grid_unit_y' in dir():
        scale_y = grid_unit_y
    else:
        scale_y = scale_y_estimate

    print(f"Using scale: x={scale_x:.2f}, y={scale_y:.2f}")

    # ========== Find offset by matching specific features ==========
    print("\n--- Finding offset using specific features ---")

    # The large room at (-22, -3) with size 5x7 has a circular feature in the center
    # Let's find this circle

    # Use HoughCircles to find circles
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=30, maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Found {len(circles[0])} circles")
        for i, (cx, cy, r) in enumerate(circles[0][:5]):
            print(f"  Circle {i}: center=({cx}, {cy}), radius={r}")

        # The largest/most prominent circle should be in the room at (-22, -3)
        # This room's center in grid coords is: (-22 + 5/2, -3 + 7/2) = (-19.5, 0.5)

        # If the circle is at pixel (cx, cy), then:
        # cx = -19.5 * scale_x + offset_x  =>  offset_x = cx + 19.5 * scale_x
        # cy = 0.5 * scale_y + offset_y    =>  offset_y = cy - 0.5 * scale_y

        # Use the largest circle
        largest_circle = circles[0][np.argmax(circles[0][:, 2])]
        cx, cy, r = largest_circle
        print(f"Largest circle: ({cx}, {cy}) r={r}")

        # Calculate offset assuming circle is at center of room (-22, -3, 5x7)
        room_center_x = -22 + 5/2  # -19.5
        room_center_y = -3 + 7/2   # 0.5

        offset_x = cx - room_center_x * scale_x
        offset_y = cy - room_center_y * scale_y

        print(f"Offset from circle: x={offset_x:.2f}, y={offset_y:.2f}")

    else:
        print("No circles found, using extent-based offset")
        offset_x = dungeon_x - grid_min_x * scale_x
        offset_y = dungeon_y - grid_min_y * scale_y

    # ========== Refinement using all rooms ==========
    print("\n--- Refining transformation ---")

    # Try to find each room's approximate center by analyzing pixel regions
    # We'll use a sliding window approach

    def find_room_center_in_region(gray, approx_x, approx_y, search_radius):
        """Find the center of a room near the approximate location"""
        # Extract region
        x1 = max(0, int(approx_x - search_radius))
        x2 = min(gray.shape[1], int(approx_x + search_radius))
        y1 = max(0, int(approx_y - search_radius))
        y2 = min(gray.shape[0], int(approx_y + search_radius))

        roi = gray[y1:y2, x1:x2]

        # Find light pixels
        _, light = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

        # Find center of mass
        coords = cv2.findNonZero(light)
        if coords is not None and len(coords) > 100:
            m = cv2.moments(coords)
            if m["m00"] > 0:
                cx = m["m10"] / m["m00"] + x1
                cy = m["m01"] / m["m00"] + y1
                return cx, cy
        return None

    # For each main room, predict its center and try to find it
    correspondences = []
    for room in main_rooms:
        grid_cx = room['x'] + room['w'] / 2
        grid_cy = room['y'] + room['h'] / 2

        pred_px = grid_cx * scale_x + offset_x
        pred_py = grid_cy * scale_y + offset_y

        result = find_room_center_in_region(gray, pred_px, pred_py, scale_x * 3)

        if result:
            px_cx, px_cy = result
            correspondences.append({
                'grid_cx': grid_cx, 'grid_cy': grid_cy,
                'pixel_cx': px_cx, 'pixel_cy': px_cy,
                'room': room
            })
            print(f"Room ({room['x']},{room['y']}): predicted ({pred_px:.0f},{pred_py:.0f}), found ({px_cx:.0f},{px_cy:.0f})")

    if len(correspondences) >= 2:
        # Refine using least squares
        grid_x = np.array([c['grid_cx'] for c in correspondences])
        grid_y = np.array([c['grid_cy'] for c in correspondences])
        pixel_x = np.array([c['pixel_cx'] for c in correspondences])
        pixel_y = np.array([c['pixel_cy'] for c in correspondences])

        # Least squares: pixel = grid * scale + offset
        A_x = np.vstack([grid_x, np.ones(len(grid_x))]).T
        scale_x_refined, offset_x_refined = np.linalg.lstsq(A_x, pixel_x, rcond=None)[0]

        A_y = np.vstack([grid_y, np.ones(len(grid_y))]).T
        scale_y_refined, offset_y_refined = np.linalg.lstsq(A_y, pixel_y, rcond=None)[0]

        print(f"\nRefined parameters:")
        print(f"  scale_x = {scale_x_refined:.4f}")
        print(f"  scale_y = {scale_y_refined:.4f}")
        print(f"  offset_x = {offset_x_refined:.4f}")
        print(f"  offset_y = {offset_y_refined:.4f}")

        # Calculate residuals
        pred_x = grid_x * scale_x_refined + offset_x_refined
        pred_y = grid_y * scale_y_refined + offset_y_refined

        residuals = np.sqrt((pixel_x - pred_x)**2 + (pixel_y - pred_y)**2)
        print(f"Residuals: mean={np.mean(residuals):.1f}, max={np.max(residuals):.1f}")

        scale_x = scale_x_refined
        scale_y = scale_y_refined
        offset_x = offset_x_refined
        offset_y = offset_y_refined

    # ========== Validation ==========
    print("\n--- Validation ---")

    debug_img = img.copy()

    for i, r in enumerate(main_rooms):
        px1 = int(r['x'] * scale_x + offset_x)
        py1 = int(r['y'] * scale_y + offset_y)
        px2 = int((r['x'] + r['w']) * scale_x + offset_x)
        py2 = int((r['y'] + r['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 5, py1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for r in corridors:
        px1 = int(r['x'] * scale_x + offset_x)
        py1 = int(r['y'] * scale_y + offset_y)
        px2 = int((r['x'] + r['w']) * scale_x + offset_x)
        py2 = int((r['y'] + r['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_v5_validation.png", debug_img)
    print("Saved debug_v5_validation.png")

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
