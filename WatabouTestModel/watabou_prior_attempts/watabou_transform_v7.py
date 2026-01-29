"""
Watabou Coordinate Transformation - Version 7
Measure grid spacing directly from the image grid pattern
"""

import json
import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

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

    # ========== Find the prominent circle ==========
    # This is our main anchor point - it's in the room at grid (-22, -3) with size 5x7
    # The circle should be roughly centered in that room

    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 100,
                               param1=50, param2=30, minRadius=80, maxRadius=250)

    if circles is None:
        print("ERROR: No circle found")
        return

    circles = np.uint16(np.around(circles))
    # Get the largest circle
    largest = circles[0][np.argmax(circles[0][:, 2])]
    circle_cx, circle_cy, circle_r = largest
    print(f"Circle found at pixel ({circle_cx}, {circle_cy}) with radius {circle_r}")

    # The circle is in room at grid (-22, -3) with size 5x7
    # Room center is at grid (-22 + 2.5, -3 + 3.5) = (-19.5, 0.5)
    circle_grid_x = -19.5
    circle_grid_y = 0.5  # Roughly center, but let's verify

    # ========== Measure grid spacing from the image ==========
    # Look at a horizontal strip through the circle to find vertical grid lines

    print("\n--- Measuring grid spacing ---")

    # Take a horizontal strip around the circle's y-position
    strip_y = circle_cy
    strip_h = 50
    strip = gray[strip_y - strip_h:strip_y + strip_h, :]

    # Find dark pixels (grid lines)
    _, dark_strip = cv2.threshold(strip, 150, 255, cv2.THRESH_BINARY_INV)

    # Project horizontally (sum along columns)
    col_profile = np.sum(dark_strip, axis=0).astype(float)

    # Smooth
    from scipy.ndimage import gaussian_filter1d
    col_smooth = gaussian_filter1d(col_profile, sigma=3)

    # Find peaks (dark vertical lines)
    peaks, props = find_peaks(col_smooth, height=strip_h * 0.3, distance=50, prominence=100)

    print(f"Found {len(peaks)} vertical lines in strip")

    if len(peaks) > 3:
        # Calculate spacings
        spacings = np.diff(peaks)
        print(f"Spacings: min={spacings.min():.0f}, max={spacings.max():.0f}, median={np.median(spacings):.0f}")

        # The most common spacing should be 1 grid unit
        # Filter out small spacings (noise) and large spacings (wall thickness variations)
        valid_spacings = spacings[(spacings > 80) & (spacings < 200)]
        if len(valid_spacings) > 0:
            grid_unit_x = np.median(valid_spacings)
            print(f"Grid unit X (from line spacing): {grid_unit_x:.1f} pixels")
        else:
            grid_unit_x = np.median(spacings)
            print(f"Grid unit X (all spacings): {grid_unit_x:.1f} pixels")
    else:
        # Fallback
        grid_unit_x = 140
        print(f"Using fallback grid unit X: {grid_unit_x}")

    # Do the same vertically
    strip_x = circle_cx
    strip_w = 50
    v_strip = gray[:, strip_x - strip_w:strip_x + strip_w]

    _, dark_v_strip = cv2.threshold(v_strip, 150, 255, cv2.THRESH_BINARY_INV)
    row_profile = np.sum(dark_v_strip, axis=1).astype(float)
    row_smooth = gaussian_filter1d(row_profile, sigma=3)

    v_peaks, _ = find_peaks(row_smooth, height=strip_w * 0.3, distance=50, prominence=100)

    print(f"Found {len(v_peaks)} horizontal lines in vertical strip")

    if len(v_peaks) > 3:
        v_spacings = np.diff(v_peaks)
        print(f"V-spacings: min={v_spacings.min():.0f}, max={v_spacings.max():.0f}, median={np.median(v_spacings):.0f}")

        valid_v_spacings = v_spacings[(v_spacings > 80) & (v_spacings < 200)]
        if len(valid_v_spacings) > 0:
            grid_unit_y = np.median(valid_v_spacings)
            print(f"Grid unit Y (from line spacing): {grid_unit_y:.1f} pixels")
        else:
            grid_unit_y = np.median(v_spacings)
            print(f"Grid unit Y (all spacings): {grid_unit_y:.1f} pixels")
    else:
        grid_unit_y = grid_unit_x  # Assume square
        print(f"Using fallback grid unit Y: {grid_unit_y}")

    # ========== Use the circle as anchor ==========
    print("\n--- Computing transformation ---")

    # The transformation is:
    # pixel_x = grid_x * scale_x + offset_x
    # pixel_y = grid_y * scale_y + offset_y

    scale_x = grid_unit_x
    scale_y = grid_unit_y

    # The circle is at grid position (-19.5, 0.5)
    # pixel = grid * scale + offset
    # circle_cx = (-19.5) * scale_x + offset_x
    # offset_x = circle_cx + 19.5 * scale_x

    offset_x = circle_cx - circle_grid_x * scale_x
    offset_y = circle_cy - circle_grid_y * scale_y

    print(f"scale_x = {scale_x:.4f}")
    print(f"scale_y = {scale_y:.4f}")
    print(f"offset_x = {offset_x:.4f}")
    print(f"offset_y = {offset_y:.4f}")

    # ========== Verify with known features ==========
    print("\n--- Verification with known features ---")

    # 1. The dungeon extent
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    dungeon_x, dungeon_y, dungeon_w, dungeon_h = cv2.boundingRect(coords)

    print(f"Dungeon extent: ({dungeon_x}, {dungeon_y}) to ({dungeon_x+dungeon_w}, {dungeon_y+dungeon_h})")

    # Grid bounds
    grid_min_x = min(r['x'] for r in rects)
    grid_max_x = max(r['x'] + r['w'] for r in rects)
    grid_min_y = min(r['y'] for r in rects)
    grid_max_y = max(r['y'] + r['h'] for r in rects)

    # Predict where grid bounds should be
    pred_left = grid_min_x * scale_x + offset_x
    pred_right = grid_max_x * scale_x + offset_x
    pred_top = grid_min_y * scale_y + offset_y
    pred_bottom = grid_max_y * scale_y + offset_y

    print(f"Grid bounds: x=[{grid_min_x}, {grid_max_x}], y=[{grid_min_y}, {grid_max_y}]")
    print(f"Predicted pixel bounds: x=[{pred_left:.0f}, {pred_right:.0f}], y=[{pred_top:.0f}, {pred_bottom:.0f}]")
    print(f"Actual dungeon bounds: x=[{dungeon_x}, {dungeon_x+dungeon_w}], y=[{dungeon_y}, {dungeon_y+dungeon_h}]")

    # Calculate error
    error_left = abs(pred_left - dungeon_x)
    error_right = abs(pred_right - (dungeon_x + dungeon_w))
    error_top = abs(pred_top - dungeon_y)
    error_bottom = abs(pred_bottom - (dungeon_y + dungeon_h))

    print(f"Errors: left={error_left:.0f}, right={error_right:.0f}, top={error_top:.0f}, bottom={error_bottom:.0f}")

    # ========== Adjust if needed ==========
    # If there's significant systematic error, adjust the offset

    total_x_error = ((pred_left - dungeon_x) + (pred_right - (dungeon_x + dungeon_w))) / 2
    total_y_error = ((pred_top - dungeon_y) + (pred_bottom - (dungeon_y + dungeon_h))) / 2

    print(f"Systematic error: x={total_x_error:.1f}, y={total_y_error:.1f}")

    # Adjust offset
    offset_x -= total_x_error
    offset_y -= total_y_error

    print(f"\nAdjusted parameters:")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    # ========== Create validation image ==========
    debug_img = img.copy()

    main_rooms = [r for r in rects if not (r['w'] == 1 and r['h'] == 1)]

    for i, room in enumerate(main_rooms):
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)

        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (px1 + 10, py1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw corridors
    corridors = [r for r in rects if r['w'] == 1 and r['h'] == 1]
    for room in corridors:
        px1 = int(room['x'] * scale_x + offset_x)
        py1 = int(room['y'] * scale_y + offset_y)
        px2 = int((room['x'] + room['w']) * scale_x + offset_x)
        py2 = int((room['y'] + room['h']) * scale_y + offset_y)
        cv2.rectangle(debug_img, (px1, py1), (px2, py2), (255, 0, 0), 2)

    # Mark circle
    cv2.circle(debug_img, (circle_cx, circle_cy), 10, (0, 255, 0), -1)

    cv2.imwrite("//debug_v7_validation.png", debug_img)
    print("\nSaved debug_v7_validation.png")

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
