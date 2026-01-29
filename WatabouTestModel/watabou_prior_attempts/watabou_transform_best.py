"""
Watabou Coordinate Transformation - Best Version
Comprehensive analysis with validation scoring
"""

import json
import cv2
import numpy as np

def evaluate_transformation(gray, rects, scale_x, scale_y, offset_x, offset_y):
    """
    Score a transformation based on how well predicted rooms match actual content.
    Returns a score where lower is better.
    """
    h, w = gray.shape
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    total_score = 0
    count = 0

    for room in main_rooms:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        # Skip if out of bounds
        if px1 < 0 or py1 < 0 or px2 > w or py2 > h:
            total_score += 1000
            count += 1
            continue

        # Check that interior is mostly white (floor)
        interior = gray[py1+10:py2-10, px1+10:px2-10] if py2-py1 > 20 and px2-px1 > 20 else None
        if interior is not None and interior.size > 0:
            floor_ratio = np.sum(interior > 200) / interior.size
            # Good if > 60% is floor
            interior_score = max(0, 0.6 - floor_ratio) * 100
        else:
            interior_score = 50

        # Check that edges have walls (dark)
        edge_samples = []
        # Top edge
        if py1 > 0:
            edge_samples.extend(gray[max(0, py1-5):py1+5, px1:px2].flatten())
        # Bottom edge
        if py2 < h:
            edge_samples.extend(gray[py2-5:min(h, py2+5), px1:px2].flatten())
        # Left edge
        if px1 > 0:
            edge_samples.extend(gray[py1:py2, max(0, px1-5):px1+5].flatten())
        # Right edge
        if px2 < w:
            edge_samples.extend(gray[py1:py2, px2-5:min(w, px2+5)].flatten())

        if edge_samples:
            wall_ratio = np.sum(np.array(edge_samples) < 100) / len(edge_samples)
            edge_score = max(0, 0.3 - wall_ratio) * 100  # Want at least 30% dark at edges
        else:
            edge_score = 50

        total_score += interior_score + edge_score
        count += 1

    return total_score / count if count > 0 else float('inf')


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

    # ========== Calculate base scale from X extent ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x

    print(f"\nBase scale_x: {scale_x:.4f}")
    print(f"Base offset_x: {offset_x:.4f}")

    # ========== Try different scale_y values ==========
    print("\n--- Testing different scale_y values ---")

    best_score = float('inf')
    best_params = None

    # Try range of scale_y from 0.6x to 2.5x of scale_x
    for scale_y_factor in np.arange(0.8, 2.5, 0.05):
        scale_y = scale_x * scale_y_factor

        # Try different offset_y values
        # Center the grid vertically at different positions
        for offset_factor in np.arange(0.3, 0.7, 0.05):
            # The grid center should be at this fraction of the image height
            grid_center_y = (grid_min_y + grid_max_y) / 2
            image_center_y = img_h * offset_factor
            offset_y = image_center_y - grid_center_y * scale_y

            score = evaluate_transformation(gray, rects, scale_x, scale_y, offset_x, offset_y)

            if score < best_score:
                best_score = score
                best_params = (scale_x, scale_y, offset_x, offset_y)
                print(f"  scale_y_factor={scale_y_factor:.2f}, offset_factor={offset_factor:.2f}: score={score:.2f}")

    if best_params:
        scale_x, scale_y, offset_x, offset_y = best_params
        print(f"\nBest parameters found with score {best_score:.2f}")
    else:
        scale_y = scale_x
        grid_center_y = (grid_min_y + grid_max_y) / 2
        offset_y = img_h * 0.5 - grid_center_y * scale_y

    print(f"\nFinal transformation:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # ========== Verify room centers are on floor ==========
    print("\n--- Verification ---")

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]
    all_ok = True

    for room in main_rooms:
        cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
        cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

        if 0 <= cx < img_w and 0 <= cy < img_h:
            pixel = gray[cy, cx]
            is_floor = pixel > 200
            status = "OK" if is_floor else "FAIL"
            if not is_floor:
                all_ok = False
            print(f"Room ({room['x']},{room['y']}): center=({cx},{cy}) pixel={pixel} {status}")

    print(f"\nAll centers on floor: {all_ok}")

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

    cv2.imwrite("//debug_best.png", debug_img)
    print("\nSaved debug_best.png")

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
