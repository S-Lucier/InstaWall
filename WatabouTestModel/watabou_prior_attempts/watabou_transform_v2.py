"""
Watabou Coordinate Transformation - Version 2

Improved approach:
1. Detect room corners/intersections in the image
2. Use the fact that grid cells should be square (scale_x == scale_y)
3. Use distinctive features (largest room, specific corners) for matching
"""

import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import itertools
from scipy.optimize import minimize

@dataclass
class GridRoom:
    """Room in grid coordinates from JSON"""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self): return self.w * self.h

    @property
    def center_x(self): return self.x + self.w / 2

    @property
    def center_y(self): return self.y + self.h / 2

    @property
    def is_corridor(self): return self.w == 1 and self.h == 1


def load_json_rooms(json_path: str) -> List[GridRoom]:
    """Load rooms from Watabou JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    rooms = []
    for rect in data['rects']:
        room = GridRoom(x=rect['x'], y=rect['y'], w=rect['w'], h=rect['h'])
        rooms.append(room)

    return rooms


def find_room_corners_by_template(img_path: str, debug: bool = True):
    """
    Find room corners using line detection and intersection.
    Watabou dungeons have distinct wall patterns.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Detect lines using HoughLines
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Dilate edges to make lines more continuous
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    if debug:
        cv2.imwrite("debug_v2_edges.png", edges_dilated)

    # Find lines
    lines = cv2.HoughLinesP(edges_dilated, 1, np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=10)

    if lines is None:
        return []

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

        if angle < 10 or angle > 170:  # Horizontal
            h_lines.append((min(y1, y2), min(x1, x2), max(x1, x2)))
        elif 80 < angle < 100:  # Vertical
            v_lines.append((min(x1, x2), min(y1, y2), max(y1, y2)))

    print(f"Found {len(h_lines)} horizontal and {len(v_lines)} vertical lines")

    if debug:
        debug_img = img.copy()
        for y, x1, x2 in h_lines:
            cv2.line(debug_img, (x1, y), (x2, y), (0, 0, 255), 2)
        for x, y1, y2 in v_lines:
            cv2.line(debug_img, (x, y1), (x, y2), (0, 255, 0), 2)
        cv2.imwrite("debug_v2_lines.png", debug_img)

    return h_lines, v_lines


def detect_grid_lines(img_path: str, debug: bool = True):
    """
    Detect the regular grid pattern in the dungeon.
    The grid lines should be evenly spaced.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find dark pixels (walls)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Project to find vertical and horizontal line positions
    # Sum along columns to find vertical lines
    col_sum = np.sum(binary, axis=0)
    # Sum along rows to find horizontal lines
    row_sum = np.sum(binary, axis=1)

    if debug:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.plot(col_sum)
        ax1.set_title("Column sums (vertical line positions)")
        ax2.plot(row_sum)
        ax2.set_title("Row sums (horizontal line positions)")
        plt.savefig("debug_v2_projections.png")
        plt.close()

    # Find peaks in the projections (these are wall positions)
    # Use local maxima
    from scipy.signal import find_peaks

    # For vertical lines (peaks in column sum)
    v_peaks, _ = find_peaks(col_sum, height=h*0.1, distance=50)
    # For horizontal lines (peaks in row sum)
    h_peaks, _ = find_peaks(row_sum, height=w*0.05, distance=50)

    print(f"Found {len(v_peaks)} vertical line positions")
    print(f"Found {len(h_peaks)} horizontal line positions")

    if debug:
        debug_img = img.copy()
        for x in v_peaks:
            cv2.line(debug_img, (x, 0), (x, h), (0, 0, 255), 1)
        for y in h_peaks:
            cv2.line(debug_img, (0, y), (w, y), (0, 255, 0), 1)
        cv2.imwrite("debug_v2_gridlines.png", debug_img)

    return v_peaks, h_peaks


def estimate_grid_spacing(positions: np.ndarray) -> Tuple[float, List[int]]:
    """
    Estimate the grid spacing from detected line positions.
    Returns the most common spacing and the grid positions.
    """
    if len(positions) < 2:
        return None, []

    # Calculate all spacings
    spacings = np.diff(sorted(positions))

    # Find the mode of spacings (most common)
    # Round to nearest 5 pixels
    spacings_rounded = np.round(spacings / 5) * 5
    unique, counts = np.unique(spacings_rounded, return_counts=True)

    # Filter out very small spacings
    valid_mask = unique > 20
    unique = unique[valid_mask]
    counts = counts[valid_mask]

    if len(unique) == 0:
        return None, []

    # Most common spacing
    mode_spacing = unique[np.argmax(counts)]

    print(f"Spacing analysis: unique={unique}, counts={counts}, mode={mode_spacing}")

    return mode_spacing, positions


def match_rooms_to_grid(grid_rooms: List[GridRoom], pixel_regions, scale: float,
                        offset_x: float, offset_y: float) -> float:
    """
    Calculate error for a given transformation.
    """
    total_error = 0
    matched = 0

    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    for gr in main_rooms:
        # Predict center in pixels
        pred_cx = gr.center_x * scale + offset_x
        pred_cy = gr.center_y * scale + offset_y

        # Find closest pixel region
        min_dist = float('inf')
        for pr in pixel_regions:
            dist = np.sqrt((pred_cx - pr['cx'])**2 + (pred_cy - pr['cy'])**2)
            if dist < min_dist:
                min_dist = dist

        if min_dist < scale * 5:  # Within reasonable distance
            matched += 1
            total_error += min_dist

    return total_error / max(matched, 1), matched


def detect_rooms_by_flooding(img_path: str, debug: bool = True):
    """
    Detect individual room regions by finding enclosed areas.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find floor pixels (white/light gray)
    _, floor_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Find walls (dark pixels)
    _, wall_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Dilate walls to ensure rooms are separated
    kernel = np.ones((7, 7), np.uint8)
    wall_dilated = cv2.dilate(wall_mask, kernel, iterations=2)

    # Floor with walls removed
    floor_clean = cv2.bitwise_and(floor_mask, cv2.bitwise_not(wall_dilated))

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(floor_clean)

    regions = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        if area < 5000:  # Filter small regions
            continue

        regions.append({
            'x': x, 'y': y, 'w': cw, 'h': ch,
            'cx': centroids[i][0], 'cy': centroids[i][1],
            'area': area
        })

    print(f"Found {len(regions)} floor regions")

    if debug:
        debug_img = img.copy()
        for i, region in enumerate(sorted(regions, key=lambda r: r['area'], reverse=True)):
            cv2.rectangle(debug_img, (region['x'], region['y']),
                         (region['x']+region['w'], region['y']+region['h']), (0, 255, 0), 2)
            cv2.circle(debug_img, (int(region['cx']), int(region['cy'])), 5, (0, 0, 255), -1)
            cv2.putText(debug_img, str(i), (region['x']+5, region['y']+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite("debug_v2_rooms.png", debug_img)

    return regions


def solve_by_largest_rooms(grid_rooms: List[GridRoom], pixel_regions: List[dict],
                           debug: bool = True) -> Tuple[float, float, float]:
    """
    Match the largest rooms from grid to pixels and solve for transformation.
    Assumes scale_x == scale_y (square grid cells).
    """
    main_rooms = sorted([r for r in grid_rooms if not r.is_corridor],
                        key=lambda r: r.area, reverse=True)
    pixel_sorted = sorted(pixel_regions, key=lambda r: r['area'], reverse=True)

    print(f"\nMatching {len(main_rooms)} grid rooms to {len(pixel_sorted)} pixel regions")

    # The largest room (5x7 = 35 area) should be the biggest detected region
    # Second largest (7x5 = 35) should be second biggest

    # Try matching largest grid rooms to largest pixel regions
    best_scale = None
    best_offset_x = None
    best_offset_y = None
    best_error = float('inf')

    # Grid rooms sorted by area: first two are 5x7 and 7x5
    for i, gr1 in enumerate(main_rooms[:3]):
        for j, gr2 in enumerate(main_rooms[:3]):
            if i == j:
                continue

            for p1 in pixel_sorted[:5]:
                for p2 in pixel_sorted[:5]:
                    if p1 is p2:
                        continue

                    # Calculate scale from size ratios
                    # pixel_width / grid_width should equal pixel_height / grid_height for square cells
                    scale_from_w1 = p1['w'] / gr1.w
                    scale_from_h1 = p1['h'] / gr1.h
                    scale_from_w2 = p2['w'] / gr2.w
                    scale_from_h2 = p2['h'] / gr2.h

                    # Average scale estimate
                    scales = [scale_from_w1, scale_from_h1, scale_from_w2, scale_from_h2]
                    avg_scale = np.mean(scales)
                    scale_var = np.std(scales)

                    # Only proceed if scales are consistent
                    if scale_var > avg_scale * 0.3:
                        continue

                    # Calculate offset from first room center
                    offset_x = p1['cx'] - gr1.center_x * avg_scale
                    offset_y = p1['cy'] - gr1.center_y * avg_scale

                    # Verify with second room
                    pred_x2 = gr2.center_x * avg_scale + offset_x
                    pred_y2 = gr2.center_y * avg_scale + offset_y
                    error = np.sqrt((pred_x2 - p2['cx'])**2 + (pred_y2 - p2['cy'])**2)

                    # Calculate total error for all rooms
                    total_error, matched = match_rooms_to_grid(grid_rooms, pixel_sorted[:15],
                                                               avg_scale, offset_x, offset_y)

                    if matched >= 4 and total_error < best_error:
                        best_error = total_error
                        best_scale = avg_scale
                        best_offset_x = offset_x
                        best_offset_y = offset_y
                        print(f"Match: gr{i}->p{pixel_sorted.index(p1)}, gr{j}->p{pixel_sorted.index(p2)}")
                        print(f"  scale={avg_scale:.2f}, offset=({offset_x:.2f}, {offset_y:.2f})")
                        print(f"  error={total_error:.2f}, matched={matched}")

    return best_scale, best_offset_x, best_offset_y


def refine_transformation_optimization(grid_rooms: List[GridRoom], pixel_regions: List[dict],
                                       initial_params: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Refine transformation using optimization.
    """
    scale0, offset_x0, offset_y0 = initial_params
    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    def objective(params):
        scale, offset_x, offset_y = params
        total_error = 0
        for gr in main_rooms:
            pred_cx = gr.center_x * scale + offset_x
            pred_cy = gr.center_y * scale + offset_y

            # Find closest pixel region
            min_dist = float('inf')
            for pr in pixel_regions:
                dist = np.sqrt((pred_cx - pr['cx'])**2 + (pred_cy - pr['cy'])**2)
                if dist < min_dist:
                    min_dist = dist
            total_error += min_dist

        return total_error

    # Optimize
    result = minimize(objective, [scale0, offset_x0, offset_y0],
                     method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 0.1, 'fatol': 0.1})

    print(f"Optimization result: {result.x}, error={result.fun}")

    return tuple(result.x)


def manual_room_matching(img_path: str, grid_rooms: List[GridRoom], debug: bool = True):
    """
    Manually identify distinctive features in the image and match to grid.

    Looking at the image:
    - Large room with columns (ending room) is at grid x=-30, 7x5
    - Large central room with circular feature is at grid x=-22, 5x7
    - Small corridor room at rightmost position is at grid x=-3, 3x5
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find the ending room (leftmost large room with columns)
    # It has water/pool features at the bottom left

    # Find the rightmost room (entrance area)
    # Look for the right edge of the dungeon content

    # Find dark pixel extent
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    if coords is not None:
        px, py, pw, ph = cv2.boundingRect(coords)
        dungeon_left = px
        dungeon_right = px + pw
        dungeon_top = py
        dungeon_bottom = py + ph
        print(f"Dungeon bounds: left={dungeon_left}, right={dungeon_right}, top={dungeon_top}, bottom={dungeon_bottom}")
        print(f"Dungeon size: {pw} x {ph}")

    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    # Grid bounds
    grid_left = min(r.x for r in main_rooms)
    grid_right = max(r.x + r.w for r in main_rooms)
    grid_top = min(r.y for r in main_rooms)
    grid_bottom = max(r.y + r.h for r in main_rooms)
    grid_width = grid_right - grid_left
    grid_height = grid_bottom - grid_top

    print(f"Grid bounds: left={grid_left}, right={grid_right}, top={grid_top}, bottom={grid_bottom}")
    print(f"Grid size: {grid_width} x {grid_height}")

    # The rightmost room edge (x=0 in grid) should be at dungeon_right
    # The leftmost room edge (x=-30 in grid) should be at dungeon_left

    # Calculate scale
    # Actually, looking at corridors: they connect rooms, so the corridor at x=0
    # connects to the room at x=-3. The right edge of dungeon should be around x=0

    # Let's assume:
    # pixel_x = grid_x * scale + offset_x
    # grid_right (0) -> dungeon_right
    # grid_left (-30) -> dungeon_left

    # 0 * scale + offset_x = dungeon_right  =>  offset_x = dungeon_right
    # -30 * scale + offset_x = dungeon_left  =>  -30 * scale = dungeon_left - offset_x
    # scale = (offset_x - dungeon_left) / 30

    # But we need to account for room margins (the hatched border)
    # Looking more carefully at the image, there's a margin around each room

    # Let me try a different approach: use the actual room positions

    return pw, ph, dungeon_left, dungeon_top, grid_width, grid_height


def find_wall_intersections(img_path: str, debug: bool = True):
    """
    Find wall intersections (corners) which should be at grid points.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris corner detection on wall pixels
    _, walls = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Convert to float
    walls_float = np.float32(walls)

    # Harris corner detection
    dst = cv2.cornerHarris(walls_float, blockSize=5, ksize=3, k=0.04)

    # Dilate to enhance corners
    dst = cv2.dilate(dst, None)

    # Threshold
    corners = np.argwhere(dst > 0.01 * dst.max())

    print(f"Found {len(corners)} corner points")

    if debug:
        debug_img = img.copy()
        for y, x in corners[:500]:  # Limit for visualization
            cv2.circle(debug_img, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite("debug_v2_corners.png", debug_img)

    return corners


def analyze_with_known_reference(img_path: str, json_path: str):
    """
    Use known structural features to establish transformation.

    From the JSON:
    - Entrance corridor at (0, 0) with 1x1 size
    - First room at (-3, -2) with 3x5 size
    - Last room (ending) at (-30, -2) with 7x5 size

    The transformation must map these to their pixel locations.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    grid_rooms = load_json_rooms(json_path)
    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    # Find distinct features in the image
    # 1. The ending room has columns - find the column pattern
    # 2. The entrance is at the far right

    # Let's use the dungeon extent with padding adjustment
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(dark)
    px, py, pw, ph = cv2.boundingRect(coords)

    print(f"Dungeon pixel extent: ({px}, {py}) to ({px+pw}, {py+ph})")
    print(f"Dungeon pixel size: {pw} x {ph}")

    # Grid extent
    min_gx = min(r.x for r in grid_rooms)  # -31 (includes corridors)
    max_gx = max(r.x + r.w for r in grid_rooms)  # 1
    min_gy = min(r.y for r in grid_rooms)  # -3
    max_gy = max(r.y + r.h for r in grid_rooms)  # 4

    grid_width = max_gx - min_gx  # 32
    grid_height = max_gy - min_gy  # 7

    print(f"Grid extent: ({min_gx}, {min_gy}) to ({max_gx}, {max_gy})")
    print(f"Grid size: {grid_width} x {grid_height}")

    # Initial scale estimate assuming uniform scaling
    scale_x = pw / grid_width
    scale_y = ph / grid_height

    # Use the average since grid cells should be square
    # But the image might have different aspect ratio
    print(f"Initial scale estimate: x={scale_x:.2f}, y={scale_y:.2f}")

    # For square cells, use the smaller scale to ensure everything fits
    # Actually, let's check if they're close
    scale = (scale_x + scale_y) / 2 if abs(scale_x - scale_y) < scale_x * 0.5 else scale_x

    # Calculate offset
    # min_gx * scale + offset_x = px  =>  offset_x = px - min_gx * scale
    offset_x = px - min_gx * scale
    offset_y = py - min_gy * scale_y  # Use scale_y for vertical

    print(f"Initial parameters: scale={scale:.4f}, offset=({offset_x:.4f}, {offset_y:.4f})")

    return scale_x, scale_y, offset_x, offset_y, min_gx, min_gy


def validate_and_visualize(json_path: str, img_path: str,
                           scale_x: float, scale_y: float,
                           offset_x: float, offset_y: float,
                           output_path: str = "debug_v2_validation.png"):
    """Validate transformation visually."""
    grid_rooms = load_json_rooms(json_path)
    img = cv2.imread(img_path)

    main_rooms = [r for r in grid_rooms if not r.is_corridor]
    corridors = [r for r in grid_rooms if r.is_corridor]

    debug_img = img.copy()

    print(f"\nValidation with scale=({scale_x:.4f}, {scale_y:.4f}), offset=({offset_x:.4f}, {offset_y:.4f})")

    # Draw main rooms
    for i, room in enumerate(main_rooms):
        x1 = int(room.x * scale_x + offset_x)
        y1 = int(room.y * scale_y + offset_y)
        x2 = int((room.x + room.w) * scale_x + offset_x)
        y2 = int((room.y + room.h) * scale_y + offset_y)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw corridors
    for room in corridors:
        x1 = int(room.x * scale_x + offset_x)
        y1 = int(room.y * scale_y + offset_y)
        x2 = int((room.x + room.w) * scale_x + offset_x)
        y2 = int((room.y + room.h) * scale_y + offset_y)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite(output_path, debug_img)
    print(f"Saved {output_path}")


def iterative_refinement(json_path: str, img_path: str):
    """
    Iteratively refine the transformation by examining the results.
    """
    # Get initial estimate
    scale_x, scale_y, offset_x, offset_y, min_gx, min_gy = analyze_with_known_reference(img_path, json_path)

    validate_and_visualize(json_path, img_path, scale_x, scale_y, offset_x, offset_y,
                          "debug_v2_iter0.png")

    # Load image for measurements
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Let's manually measure some key points
    # Find the rightmost dungeon edge (should correspond to grid x=0 or x=1)

    # Looking at the validation, we need to adjust
    # Let me detect individual rooms more precisely

    # Detect rooms
    pixel_rooms = detect_rooms_by_flooding(img_path)

    # Sort rooms by x position (left to right in image = right to left in grid negative coords)
    pixel_rooms_sorted = sorted(pixel_rooms, key=lambda r: r['x'])

    print("\nPixel rooms left to right:")
    for i, pr in enumerate(pixel_rooms_sorted[:10]):
        print(f"  {i}: x={pr['x']:.0f}, y={pr['y']:.0f}, w={pr['w']:.0f}, h={pr['h']:.0f}, area={pr['area']}")

    grid_rooms = load_json_rooms(json_path)
    main_rooms = sorted([r for r in grid_rooms if not r.is_corridor], key=lambda r: r.x)

    print("\nGrid rooms left to right (most negative to least):")
    for i, gr in enumerate(main_rooms):
        print(f"  {i}: x={gr.x}, y={gr.y}, w={gr.w}, h={gr.h}, area={gr.area}")

    # Try matching by position
    # Leftmost pixel room should match leftmost grid room (x=-30)
    # Rightmost pixel room should match rightmost grid room (x=-3)

    # But we have 6 main rooms and might have detected more regions

    # Let's try using the two largest rooms which are distinct
    # Grid: (-22, -3, 5x7) area=35 and (-30, -2, 7x5) area=35
    # These should be the two largest detected regions

    largest_grid = sorted(main_rooms, key=lambda r: r.area, reverse=True)[:2]
    largest_pixel = sorted(pixel_rooms, key=lambda r: r['area'], reverse=True)[:5]

    print("\nLargest grid rooms:")
    for gr in largest_grid:
        print(f"  ({gr.x}, {gr.y}) {gr.w}x{gr.h} area={gr.area}")

    print("\nLargest pixel rooms:")
    for pr in largest_pixel:
        print(f"  ({pr['x']:.0f}, {pr['y']:.0f}) {pr['w']:.0f}x{pr['h']:.0f} area={pr['area']}")

    # Match them
    # The 7x5 room at x=-30 should have aspect ratio > 1 (wider than tall)
    # The 5x7 room at x=-22 should have aspect ratio < 1 (taller than wide)

    gr_wide = [r for r in largest_grid if r.w > r.h][0]  # 7x5
    gr_tall = [r for r in largest_grid if r.h > r.w][0]  # 5x7

    # Find matching pixel rooms
    pr_wide_candidates = [r for r in largest_pixel if r['w'] > r['h']]
    pr_tall_candidates = [r for r in largest_pixel if r['h'] > r['w']]

    print(f"\nWide grid room: ({gr_wide.x}, {gr_wide.y}) {gr_wide.w}x{gr_wide.h}")
    print(f"Wide pixel candidates:")
    for pr in pr_wide_candidates:
        print(f"  ({pr['x']:.0f}, {pr['y']:.0f}) {pr['w']:.0f}x{pr['h']:.0f}")

    print(f"\nTall grid room: ({gr_tall.x}, {gr_tall.y}) {gr_tall.w}x{gr_tall.h}")
    print(f"Tall pixel candidates:")
    for pr in pr_tall_candidates:
        print(f"  ({pr['x']:.0f}, {pr['y']:.0f}) {pr['w']:.0f}x{pr['h']:.0f}")

    # Best match: largest wide and largest tall
    if pr_wide_candidates and pr_tall_candidates:
        pr_wide = max(pr_wide_candidates, key=lambda r: r['area'])
        pr_tall = max(pr_tall_candidates, key=lambda r: r['area'])

        # Calculate scales from each room
        scale_x_wide = pr_wide['w'] / gr_wide.w
        scale_y_wide = pr_wide['h'] / gr_wide.h
        scale_x_tall = pr_tall['w'] / gr_tall.w
        scale_y_tall = pr_tall['h'] / gr_tall.h

        print(f"\nScale estimates:")
        print(f"  From wide room: x={scale_x_wide:.2f}, y={scale_y_wide:.2f}")
        print(f"  From tall room: x={scale_x_tall:.2f}, y={scale_y_tall:.2f}")

        # Average scales
        scale_x_new = (scale_x_wide + scale_x_tall) / 2
        scale_y_new = (scale_y_wide + scale_y_tall) / 2

        # Calculate offset from wide room (ending room)
        # Center of grid room -> center of pixel room
        gr_cx = gr_wide.center_x
        gr_cy = gr_wide.center_y
        pr_cx = pr_wide['cx']
        pr_cy = pr_wide['cy']

        offset_x_new = pr_cx - gr_cx * scale_x_new
        offset_y_new = pr_cy - gr_cy * scale_y_new

        print(f"\nRefined parameters:")
        print(f"  scale_x = {scale_x_new:.4f}")
        print(f"  scale_y = {scale_y_new:.4f}")
        print(f"  offset_x = {offset_x_new:.4f}")
        print(f"  offset_y = {offset_y_new:.4f}")

        validate_and_visualize(json_path, img_path, scale_x_new, scale_y_new,
                              offset_x_new, offset_y_new, "debug_v2_iter1.png")

        # Further refine by checking all rooms
        return scale_x_new, scale_y_new, offset_x_new, offset_y_new

    return scale_x, scale_y, offset_x, offset_y


if __name__ == "__main__":
    json_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.json"
    img_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png"

    print("=" * 70)
    print("Watabou Coordinate Transformation - Version 2")
    print("=" * 70)

    # Run iterative refinement
    scale_x, scale_y, offset_x, offset_y = iterative_refinement(json_path, img_path)

    print("\n" + "=" * 70)
    print("FINAL TRANSFORMATION PARAMETERS")
    print("=" * 70)
    print(f"Transformation: pixel = grid * scale + offset")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")
    print("\nTo convert grid coordinates to pixels:")
    print(f"  pixel_x = grid_x * {scale_x:.4f} + {offset_x:.4f}")
    print(f"  pixel_y = grid_y * {scale_y:.4f} + {offset_y:.4f}")
