"""
Watabou Coordinate Transformation - Edge-based Version
Find room boundaries using edge detection
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

    print(f"Grid: x=[{grid_min_x}, {grid_max_x}]={grid_width}, y=[{grid_min_y}, {grid_max_y}]={grid_height}")
    print(f"Dungeon extent: ({extent_x}, {extent_y}) {extent_w}x{extent_h}")

    # ========== Scale X from dungeon width ==========
    scale_x = extent_w / grid_width
    offset_x = extent_x - grid_min_x * scale_x
    print(f"\nScale X: {scale_x:.4f}")

    # ========== Analyze the large room (-22, -3) which is 5x7 ==========
    # This room should be easy to identify because it's the largest

    room_22 = {'x': -22, 'y': -3, 'w': 5, 'h': 7}
    pred_cx = (room_22['x'] + room_22['w']/2) * scale_x + offset_x  # ~2379

    print(f"\nAnalyzing room at (-22,-3) predicted center x={pred_cx:.0f}")

    # Extract a vertical strip around this position
    strip_x = int(pred_cx)
    strip_hw = 100  # Half width
    strip = gray[:, max(0, strip_x-strip_hw):min(img_w, strip_x+strip_hw)]

    # Find edges
    edges = cv2.Canny(strip, 50, 150)

    # Project edges horizontally to find horizontal lines
    edge_profile = np.sum(edges, axis=1)

    # Find peaks (horizontal edges)
    from scipy.signal import find_peaks
    peaks, props = find_peaks(edge_profile, height=strip_hw*0.5, distance=30)

    print(f"Found {len(peaks)} horizontal edge peaks")
    for p in peaks[:15]:
        print(f"  y={p}: strength={edge_profile[p]:.0f}")

    # Look for the main room boundaries
    # The room is 7 grid units tall, so should span 7 * scale_y pixels

    # Find pairs of peaks that could be room top/bottom
    # Looking at the image, room heights should be substantial

    # Let's try different scale_y values and see which matches best
    print("\n--- Testing scale_y values ---")

    best_scale_y = None
    best_offset_y = None
    best_score = float('inf')

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for scale_y_ratio in np.arange(1.0, 2.5, 0.1):
        scale_y = scale_x * scale_y_ratio

        # Find best offset_y
        for offset_factor in np.arange(0.3, 0.7, 0.02):
            grid_center_y = (grid_min_y + grid_max_y) / 2
            offset_y = img_h * offset_factor - grid_center_y * scale_y

            # Score: check if room centers are on floor pixels
            score = 0
            for room in main_rooms:
                cx = int((room['x'] + room['w']/2) * scale_x + offset_x)
                cy = int((room['y'] + room['h']/2) * scale_y + offset_y)

                if 0 <= cx < img_w and 0 <= cy < img_h:
                    pixel = gray[cy, cx]
                    if pixel < 180:
                        score += 1
                else:
                    score += 2

            if score < best_score:
                best_score = score
                best_scale_y = scale_y
                best_offset_y = offset_y

    print(f"Best: scale_y={best_scale_y:.4f} (ratio={best_scale_y/scale_x:.2f}), offset_y={best_offset_y:.4f}, score={best_score}")

    scale_y = best_scale_y
    offset_y = best_offset_y

    # ========== Refine using edge alignment ==========
    # Now try to align the scale_y so room edges match actual edges

    print("\n--- Refining with edge alignment ---")

    # For each room, check how well its predicted edges match actual edges
    for room in main_rooms[:3]:
        grid_top = room['y']
        grid_bottom = room['y'] + room['h']

        pred_top = grid_top * scale_y + offset_y
        pred_bottom = grid_bottom * scale_y + offset_y

        pred_cx = (room['x'] + room['w']/2) * scale_x + offset_x

        # Check if there are edges near predicted positions
        strip_x = int(pred_cx)
        if strip_x > strip_hw and strip_x < img_w - strip_hw:
            col_edges = edges[:, strip_hw]  # Center column of strip

            # Find nearest edge to predicted top
            edge_positions = np.where(col_edges > 0)[0]
            if len(edge_positions) > 0:
                dist_to_top = np.abs(edge_positions - pred_top)
                nearest_top = edge_positions[np.argmin(dist_to_top)]
                dist_to_bottom = np.abs(edge_positions - pred_bottom)
                nearest_bottom = edge_positions[np.argmin(dist_to_bottom)]

                print(f"Room ({room['x']},{room['y']}): pred edges at y={pred_top:.0f},{pred_bottom:.0f}, nearest actual: {nearest_top},{nearest_bottom}")

    # ========== Verify ==========
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

    cv2.imwrite("//debug_edges.png", debug_img)
    print("\nSaved debug_edges.png")

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
