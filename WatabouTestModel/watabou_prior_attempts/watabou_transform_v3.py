"""
Watabou Coordinate Transformation - Version 3
Simplified and more direct approach
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

    # Get grid bounds (including corridors)
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
    if img is None:
        print("Failed to load image")
        return

    img_h, img_w = img.shape[:2]
    print(f"Image size: {img_w} x {img_h}")

    # Convert to grayscale and find dungeon extent
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find dark pixels (walls, lines)
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find bounding box of dark pixels
    coords = cv2.findNonZero(dark_mask)
    if coords is None:
        print("No dark pixels found")
        return

    px, py, pw, ph = cv2.boundingRect(coords)
    print(f"Dungeon extent in pixels: x={px}, y={py}, w={pw}, h={ph}")

    # Initial estimate
    scale_x_init = pw / grid_width
    scale_y_init = ph / grid_height
    print(f"Initial scale estimate: x={scale_x_init:.2f}, y={scale_y_init:.2f}")

    # ========== Room Detection ==========
    print("\n--- Room Detection ---")

    # Find white floor areas
    _, floor_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Erode to separate connected rooms
    kernel = np.ones((15, 15), np.uint8)
    floor_eroded = cv2.erode(floor_mask, kernel, iterations=1)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(floor_eroded)

    print(f"Found {num_labels - 1} connected components")

    # Filter and store regions
    pixel_regions = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 3000:  # Filter small
            continue
        pixel_regions.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': centroids[i][0], 'cy': centroids[i][1],
            'area': area
        })

    print(f"Kept {len(pixel_regions)} significant regions")

    # Sort by area descending
    pixel_regions = sorted(pixel_regions, key=lambda r: r['area'], reverse=True)
    for i, pr in enumerate(pixel_regions[:10]):
        print(f"  Region {i}: pos=({pr['x']}, {pr['y']}) size=({pr['w']}x{pr['h']}) area={pr['area']}")

    # ========== Match Largest Rooms ==========
    print("\n--- Room Matching ---")

    # Main rooms (non-corridors)
    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]
    main_rooms = sorted(main_rooms, key=lambda r: r['w'] * r['h'], reverse=True)

    print("Main rooms by area:")
    for i, r in enumerate(main_rooms):
        ar = r['w'] / r['h']
        print(f"  Room {i}: grid=({r['x']}, {r['y']}) size=({r['w']}x{r['h']}) AR={ar:.2f}")

    # The two largest rooms are:
    # 1. (-22, -3) 5x7 area=35, AR=0.71 (tall)
    # 2. (-30, -2) 7x5 area=35, AR=1.40 (wide, ending room)

    # Find matching pixel regions by aspect ratio
    tall_grid = next(r for r in main_rooms if r['w'] < r['h'])  # 5x7
    wide_grid = next(r for r in main_rooms if r['w'] > r['h'])  # 7x5

    print(f"\nTall room (grid): ({tall_grid['x']}, {tall_grid['y']}) {tall_grid['w']}x{tall_grid['h']}")
    print(f"Wide room (grid): ({wide_grid['x']}, {wide_grid['y']}) {wide_grid['w']}x{wide_grid['h']}")

    # Find corresponding pixel regions
    tall_pixel = None
    wide_pixel = None

    for pr in pixel_regions:
        ar = pr['w'] / pr['h']
        if ar < 0.9 and tall_pixel is None:  # Tall
            tall_pixel = pr
        elif ar > 1.1 and wide_pixel is None:  # Wide
            wide_pixel = pr

    if tall_pixel:
        print(f"Tall room (pixel): ({tall_pixel['x']}, {tall_pixel['y']}) {tall_pixel['w']}x{tall_pixel['h']}")
    if wide_pixel:
        print(f"Wide room (pixel): ({wide_pixel['x']}, {wide_pixel['y']}) {wide_pixel['w']}x{wide_pixel['h']}")

    if not tall_pixel or not wide_pixel:
        print("Could not find matching rooms by aspect ratio")
        # Fall back to initial estimate
        scale_x = scale_x_init
        scale_y = scale_y_init
        offset_x = px - grid_min_x * scale_x
        offset_y = py - grid_min_y * scale_y
    else:
        # ========== Calculate Transformation ==========
        print("\n--- Calculate Transformation ---")

        # Calculate scale from both rooms
        scale_x_tall = tall_pixel['w'] / tall_grid['w']
        scale_y_tall = tall_pixel['h'] / tall_grid['h']
        scale_x_wide = wide_pixel['w'] / wide_grid['w']
        scale_y_wide = wide_pixel['h'] / wide_grid['h']

        print(f"Scale from tall room: x={scale_x_tall:.2f}, y={scale_y_tall:.2f}")
        print(f"Scale from wide room: x={scale_x_wide:.2f}, y={scale_y_wide:.2f}")

        # Average
        scale_x = (scale_x_tall + scale_x_wide) / 2
        scale_y = (scale_y_tall + scale_y_wide) / 2

        print(f"Average scale: x={scale_x:.2f}, y={scale_y:.2f}")

        # Calculate offset from tall room (more distinctive)
        # pixel_center = grid_center * scale + offset
        tall_grid_cx = tall_grid['x'] + tall_grid['w'] / 2
        tall_grid_cy = tall_grid['y'] + tall_grid['h'] / 2

        offset_x = tall_pixel['cx'] - tall_grid_cx * scale_x
        offset_y = tall_pixel['cy'] - tall_grid_cy * scale_y

        print(f"Offset calculated: x={offset_x:.2f}, y={offset_y:.2f}")

    # ========== Validation ==========
    print("\n--- Validation ---")

    debug_img = img.copy()

    errors = []
    for i, r in enumerate(main_rooms):
        # Convert grid to pixel
        px1 = int(r['x'] * scale_x + offset_x)
        py1 = int(r['y'] * scale_y + offset_y)
        px2 = int((r['x'] + r['w']) * scale_x + offset_x)
        py2 = int((r['y'] + r['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 5, py1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print(f"Room {i}: grid({r['x']},{r['y']},{r['w']}x{r['h']}) -> pixel({px1},{py1})-({px2},{py2})")

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for r in corridors:
        px1 = int(r['x'] * scale_x + offset_x)
        py1 = int(r['y'] * scale_y + offset_y)
        px2 = int((r['x'] + r['w']) * scale_x + offset_x)
        py2 = int((r['y'] + r['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    cv2.imwrite("//debug_v3_validation.png", debug_img)
    print("\nSaved debug_v3_validation.png")

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
