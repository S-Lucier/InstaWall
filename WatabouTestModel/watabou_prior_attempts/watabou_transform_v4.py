"""
Watabou Coordinate Transformation - Version 4
More careful room detection and matching
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

    # Get grid bounds
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    grid_min_x = min(all_x)
    grid_max_x = max(all_x)
    grid_min_y = min(all_y)
    grid_max_y = max(all_y)

    grid_width = grid_max_x - grid_min_x
    grid_height = grid_max_y - grid_min_y

    print(f"Grid bounds: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # Load image
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_w} x {img_h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find dungeon extent using dark pixels
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark_mask)
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = cv2.boundingRect(coords)
    print(f"Dungeon extent: ({dungeon_x}, {dungeon_y}) size {dungeon_w}x{dungeon_h}")

    # ========== Improved Room Detection ==========
    print("\n--- Room Detection (Improved) ---")

    # Find white/light areas that are INSIDE the dungeon bounds
    # and exclude the background

    # Create a mask for the dungeon area
    dungeon_mask = np.zeros_like(gray)
    dungeon_mask[dungeon_y:dungeon_y+dungeon_h, dungeon_x:dungeon_x+dungeon_w] = 255

    # Find floor pixels (white)
    _, floor_all = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Mask to only dungeon area
    floor_dungeon = cv2.bitwise_and(floor_all, dungeon_mask)

    # Now we need to separate individual rooms
    # The rooms are connected by corridors, so we need to segment them

    # Use wall detection to create barriers
    _, walls = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Dilate walls to create stronger barriers
    kernel = np.ones((11, 11), np.uint8)
    walls_thick = cv2.dilate(walls, kernel, iterations=2)

    # Floor minus thick walls
    floor_separated = cv2.bitwise_and(floor_dungeon, cv2.bitwise_not(walls_thick))

    # Erode to separate connected areas more
    kernel_erode = np.ones((21, 21), np.uint8)
    floor_eroded = cv2.erode(floor_separated, kernel_erode, iterations=1)

    cv2.imwrite("//debug_v4_floor_eroded.png", floor_eroded)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(floor_eroded)
    print(f"Found {num_labels - 1} components")

    # Filter regions
    pixel_regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Filter: minimum size, maximum size, inside dungeon
        if area < 5000:
            continue
        if area > dungeon_w * dungeon_h * 0.5:  # Too large (background)
            continue
        if x < dungeon_x - 50 or x + w > dungeon_x + dungeon_w + 50:
            continue

        pixel_regions.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': centroids[i][0], 'cy': centroids[i][1],
            'area': area,
            'ar': w / h if h > 0 else 1.0
        })

    # Sort by x position (left to right)
    pixel_regions = sorted(pixel_regions, key=lambda r: r['x'])

    print(f"Filtered to {len(pixel_regions)} regions (excluding background)")
    for i, pr in enumerate(pixel_regions):
        print(f"  {i}: x={pr['x']:4d} y={pr['y']:4d} size={pr['w']:3d}x{pr['h']:3d} AR={pr['ar']:.2f} area={pr['area']}")

    # ========== Visualize Detected Regions ==========
    debug_img = img.copy()
    for i, pr in enumerate(pixel_regions):
        cv2.rectangle(debug_img, (pr['x'], pr['y']), (pr['x']+pr['w'], pr['y']+pr['h']), (0, 255, 0), 2)
        cv2.circle(debug_img, (int(pr['cx']), int(pr['cy'])), 5, (0, 0, 255), -1)
        cv2.putText(debug_img, str(i), (pr['x']+5, pr['y']+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("//debug_v4_detected.png", debug_img)
    print("Saved debug_v4_detected.png")

    # ========== Match Rooms ==========
    print("\n--- Room Matching ---")

    # Main rooms sorted by x (left to right in grid = most negative to least negative)
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]
    main_rooms = sorted(main_rooms, key=lambda r: r['x'])

    print("Grid rooms (left to right in grid coords):")
    for i, r in enumerate(main_rooms):
        ar = r['w'] / r['h']
        print(f"  {i}: x={r['x']:3d} y={r['y']:3d} size={r['w']}x{r['h']} AR={ar:.2f}")

    # The leftmost grid room (x=-30) should be leftmost in the image
    # The rightmost grid room (x=-3) should be rightmost in the image

    # Check if we have the right number of regions
    if len(pixel_regions) != len(main_rooms):
        print(f"Warning: {len(pixel_regions)} pixel regions vs {len(main_rooms)} grid rooms")

    # Match by position order: grid rooms sorted by x should match pixel regions sorted by x
    # (since x gets more negative going left in both coordinate systems)

    matches = []
    if len(pixel_regions) >= len(main_rooms):
        # Try to match by order
        for i, gr in enumerate(main_rooms):
            if i < len(pixel_regions):
                matches.append((gr, pixel_regions[i]))

        print("\nMatches by order:")
        for gr, pr in matches:
            print(f"  Grid({gr['x']},{gr['y']}) {gr['w']}x{gr['h']} <-> Pixel({pr['x']},{pr['y']}) {pr['w']}x{pr['h']}")

    # ========== Calculate Transformation from Matches ==========
    print("\n--- Calculate Transformation ---")

    if len(matches) >= 2:
        # Use multiple matches to estimate scale and offset
        # Collect data points: grid_x -> pixel_cx, grid_y -> pixel_cy

        grid_centers_x = []
        grid_centers_y = []
        pixel_centers_x = []
        pixel_centers_y = []

        for gr, pr in matches:
            gc_x = gr['x'] + gr['w'] / 2
            gc_y = gr['y'] + gr['h'] / 2
            grid_centers_x.append(gc_x)
            grid_centers_y.append(gc_y)
            pixel_centers_x.append(pr['cx'])
            pixel_centers_y.append(pr['cy'])

        # Linear regression: pixel = grid * scale + offset
        # For x: pixel_cx = grid_cx * scale_x + offset_x
        # For y: pixel_cy = grid_cy * scale_y + offset_y

        grid_centers_x = np.array(grid_centers_x)
        grid_centers_y = np.array(grid_centers_y)
        pixel_centers_x = np.array(pixel_centers_x)
        pixel_centers_y = np.array(pixel_centers_y)

        # Solve for scale_x, offset_x using least squares
        A_x = np.vstack([grid_centers_x, np.ones(len(grid_centers_x))]).T
        scale_x, offset_x = np.linalg.lstsq(A_x, pixel_centers_x, rcond=None)[0]

        # Solve for scale_y, offset_y
        A_y = np.vstack([grid_centers_y, np.ones(len(grid_centers_y))]).T
        scale_y, offset_y = np.linalg.lstsq(A_y, pixel_centers_y, rcond=None)[0]

        print(f"Least squares solution:")
        print(f"  scale_x = {scale_x:.4f}")
        print(f"  scale_y = {scale_y:.4f}")
        print(f"  offset_x = {offset_x:.4f}")
        print(f"  offset_y = {offset_y:.4f}")

        # Calculate residuals
        pred_x = grid_centers_x * scale_x + offset_x
        pred_y = grid_centers_y * scale_y + offset_y
        residuals_x = pixel_centers_x - pred_x
        residuals_y = pixel_centers_y - pred_y

        print(f"\nResiduals (pixels):")
        for i, (gr, pr) in enumerate(matches):
            print(f"  Room {i}: x_err={residuals_x[i]:.1f}, y_err={residuals_y[i]:.1f}")

        rmse_x = np.sqrt(np.mean(residuals_x**2))
        rmse_y = np.sqrt(np.mean(residuals_y**2))
        print(f"RMSE: x={rmse_x:.2f}, y={rmse_y:.2f}")

    else:
        # Fallback to simple calculation
        scale_x = dungeon_w / grid_width
        scale_y = dungeon_h / grid_height
        offset_x = dungeon_x - grid_min_x * scale_x
        offset_y = dungeon_y - grid_min_y * scale_y
        print("Using fallback calculation (not enough matches)")

    # ========== Validation Visualization ==========
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

        print(f"Room {i}: grid({r['x']},{r['y']}) -> pixel({px1},{py1})")

    # Corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for r in corridors:
        px1 = int(r['x'] * scale_x + offset_x)
        py1 = int(r['y'] * scale_y + offset_y)
        px2 = int((r['x'] + r['w']) * scale_x + offset_x)
        py2 = int((r['y'] + r['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_v4_validation.png", debug_img)
    print("\nSaved debug_v4_validation.png")

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
