"""
Watabou Coordinate Transformation - Final Version
Direct approach using known reference points
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

    # ========== Find the prominent circle (in room at -22, -3) ==========
    print("\n--- Finding reference features ---")

    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=50, maxRadius=200)

    circle_center = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find the largest circle
        largest = circles[0][np.argmax(circles[0][:, 2])]
        circle_center = (largest[0], largest[1])
        circle_radius = largest[2]
        print(f"Found circle at ({circle_center[0]}, {circle_center[1]}) radius={circle_radius}")

    # ========== Find dungeon extent ==========
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = cv2.boundingRect(coords)
    print(f"Dungeon bounds: ({dungeon_x}, {dungeon_y}) to ({dungeon_x+dungeon_w}, {dungeon_y+dungeon_h})")

    # ========== Grid information ==========
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)

    grid_width = grid_max_x - grid_min_x  # 32
    grid_height = grid_max_y - grid_min_y  # 7

    print(f"Grid bounds: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")

    # ========== Method 1: Use dungeon extent + aspect correction ==========
    print("\n--- Method 1: Extent-based ---")

    # Initial scale from extent
    scale_x1 = dungeon_w / grid_width
    scale_y1 = dungeon_h / grid_height

    # Offset so that grid_min maps to dungeon_min
    offset_x1 = dungeon_x - grid_min_x * scale_x1
    offset_y1 = dungeon_y - grid_min_y * scale_y1

    print(f"Scale: ({scale_x1:.2f}, {scale_y1:.2f})")
    print(f"Offset: ({offset_x1:.2f}, {offset_y1:.2f})")

    # ========== Method 2: Use circle as anchor ==========
    if circle_center:
        print("\n--- Method 2: Circle-anchored ---")

        # The circle is in room at (-22, -3) with size 5x7
        # Room center: (-22 + 2.5, -3 + 3.5) = (-19.5, 0.5)
        room_grid_cx = -22 + 5/2
        room_grid_cy = -3 + 7/2

        # Use scale from extent
        scale_x2 = scale_x1
        scale_y2 = scale_y1

        # Calculate offset from circle
        offset_x2 = circle_center[0] - room_grid_cx * scale_x2
        offset_y2 = circle_center[1] - room_grid_cy * scale_y2

        print(f"Scale: ({scale_x2:.2f}, {scale_y2:.2f})")
        print(f"Offset: ({offset_x2:.2f}, {offset_y2:.2f})")

    # ========== Method 3: Manual adjustment ==========
    print("\n--- Method 3: Iterative refinement ---")

    # Start with extent-based estimate
    scale_x = scale_x1
    scale_y = scale_y1
    offset_x = offset_x1
    offset_y = offset_y1

    # The rooms are probably not at the exact edges of the dungeon extent
    # There's likely padding/margin

    # Let's refine by checking specific rooms
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    # For each room, find the actual floor region center
    room_matches = []

    for room in main_rooms:
        # Predict pixel location
        pred_cx = (room['x'] + room['w']/2) * scale_x + offset_x
        pred_cy = (room['y'] + room['h']/2) * scale_y + offset_y

        # Extract a region around this location
        margin = int(max(room['w'], room['h']) * scale_x * 0.7)
        x1 = max(0, int(pred_cx - margin))
        x2 = min(img_w, int(pred_cx + margin))
        y1 = max(0, int(pred_cy - margin))
        y2 = min(img_h, int(pred_cy + margin))

        roi = gray[y1:y2, x1:x2]

        # Find white floor pixels
        _, floor = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY)

        # Find centroid
        M = cv2.moments(floor)
        if M["m00"] > 1000:
            actual_cx = M["m10"] / M["m00"] + x1
            actual_cy = M["m01"] / M["m00"] + y1

            room_matches.append({
                'grid_cx': room['x'] + room['w']/2,
                'grid_cy': room['y'] + room['h']/2,
                'pixel_cx': actual_cx,
                'pixel_cy': actual_cy
            })

    print(f"Found {len(room_matches)} room correspondences")

    if len(room_matches) >= 2:
        # Solve using least squares
        grid_x = np.array([m['grid_cx'] for m in room_matches])
        grid_y = np.array([m['grid_cy'] for m in room_matches])
        pixel_x = np.array([m['pixel_cx'] for m in room_matches])
        pixel_y = np.array([m['pixel_cy'] for m in room_matches])

        # pixel = grid * scale + offset
        # [grid, 1] * [scale, offset]^T = pixel

        A_x = np.column_stack([grid_x, np.ones(len(grid_x))])
        result_x, residuals_x, _, _ = np.linalg.lstsq(A_x, pixel_x, rcond=None)
        scale_x, offset_x = result_x

        A_y = np.column_stack([grid_y, np.ones(len(grid_y))])
        result_y, residuals_y, _, _ = np.linalg.lstsq(A_y, pixel_y, rcond=None)
        scale_y, offset_y = result_y

        print(f"\nLeast squares solution:")
        print(f"  scale_x = {scale_x:.4f}")
        print(f"  scale_y = {scale_y:.4f}")
        print(f"  offset_x = {offset_x:.4f}")
        print(f"  offset_y = {offset_y:.4f}")

        # Check fit
        pred_x = grid_x * scale_x + offset_x
        pred_y = grid_y * scale_y + offset_y
        errors = np.sqrt((pixel_x - pred_x)**2 + (pixel_y - pred_y)**2)
        print(f"\nFit quality: mean error = {np.mean(errors):.1f} pixels, max = {np.max(errors):.1f} pixels")

    # ========== Validation visualization ==========
    print("\n--- Creating validation image ---")

    debug_img = img.copy()

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 10, py1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Mark circle if found
    if circle_center:
        cv2.circle(debug_img, circle_center, 5, (0, 255, 0), -1)

    cv2.imwrite("//debug_final_validation.png", debug_img)
    print("Saved debug_final_validation.png")

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

    # Verify with key points
    print("\n--- Verification ---")
    print("Grid origin (0, 0) maps to pixel:", end=" ")
    print(f"({0 * scale_x + offset_x:.0f}, {0 * scale_y + offset_y:.0f})")

    print("Grid point (-30, -2) maps to pixel:", end=" ")
    print(f"({-30 * scale_x + offset_x:.0f}, {-2 * scale_y + offset_y:.0f})")

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    main()
