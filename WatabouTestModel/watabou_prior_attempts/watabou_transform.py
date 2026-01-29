"""
Reverse-engineer Watabou's coordinate transformation from grid to pixel coordinates.

Strategy:
1. Load the JSON to get room bounds in grid coordinates
2. Use computer vision to detect room regions in the PNG
3. Match rooms based on size ratios and relative positions
4. Solve for the transformation: pixel = grid * scale + offset
"""

import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import itertools

@dataclass
class GridRoom:
    """Room in grid coordinates from JSON"""
    x: int
    y: int
    w: int
    h: int
    area: int
    is_corridor: bool

    @property
    def center_x(self):
        return self.x + self.w / 2

    @property
    def center_y(self):
        return self.y + self.h / 2

@dataclass
class PixelRegion:
    """Detected region in pixel coordinates"""
    x: int  # left
    y: int  # top
    w: int  # width
    h: int  # height
    area: int
    contour: np.ndarray

    @property
    def center_x(self):
        return self.x + self.w / 2

    @property
    def center_y(self):
        return self.y + self.h / 2


def load_json_rooms(json_path: str) -> List[GridRoom]:
    """Load rooms from Watabou JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    rooms = []
    for rect in data['rects']:
        room = GridRoom(
            x=rect['x'],
            y=rect['y'],
            w=rect['w'],
            h=rect['h'],
            area=rect['w'] * rect['h'],
            is_corridor=(rect['w'] == 1 and rect['h'] == 1)
        )
        rooms.append(room)

    return rooms


def detect_rooms_contours(img_path: str, debug: bool = True) -> List[PixelRegion]:
    """Detect room regions using contour detection"""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The dungeon rooms appear as lighter areas bounded by dark walls
    # Let's use thresholding to find the floor areas

    # First, find the background color (likely the beige border)
    # The rooms are white/light gray, walls are black

    # Threshold to find white/light areas (rooms)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if debug:
        cv2.imwrite("debug_thresh.png", thresh)
        print("Saved debug_thresh.png")

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:  # Filter small noise
            continue

        x, y, w, h = cv2.boundingRect(contour)
        regions.append(PixelRegion(
            x=x, y=y, w=w, h=h, area=area, contour=contour
        ))

    print(f"Found {len(regions)} regions with contour method")

    if debug:
        debug_img = img.copy()
        for i, region in enumerate(regions):
            cv2.rectangle(debug_img, (region.x, region.y),
                         (region.x + region.w, region.y + region.h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i), (region.x + 5, region.y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite("debug_contours.png", debug_img)
        print("Saved debug_contours.png")

    return regions


def detect_rooms_flood_fill(img_path: str, debug: bool = True) -> List[PixelRegion]:
    """Alternative: Use flood fill to detect connected white regions"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Threshold to find white areas
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # Use connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    regions = []
    for i in range(1, num_labels):  # Skip background (label 0)
        x, y, w_comp, h_comp, area = stats[i]
        if area < 1000:
            continue

        # Create contour from label mask
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            regions.append(PixelRegion(
                x=x, y=y, w=w_comp, h=h_comp, area=area, contour=contours[0]
            ))

    print(f"Found {len(regions)} regions with connected components")

    if debug:
        debug_img = img.copy()
        for i, region in enumerate(regions):
            cv2.rectangle(debug_img, (region.x, region.y),
                         (region.x + region.w, region.y + region.h), (255, 0, 0), 2)
            cv2.putText(debug_img, str(i), (region.x + 5, region.y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite("debug_flood.png", debug_img)
        print("Saved debug_flood.png")

    return regions


def detect_rooms_morphology(img_path: str, debug: bool = True) -> List[PixelRegion]:
    """Use morphological operations to find room boundaries"""
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find dark pixels (walls)
    _, walls = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Dilate walls to close gaps
    kernel = np.ones((5, 5), np.uint8)
    walls_dilated = cv2.dilate(walls, kernel, iterations=2)

    # Invert to get floor regions
    floors = cv2.bitwise_not(walls_dilated)

    # Erode to separate connected rooms
    floors_eroded = cv2.erode(floors, kernel, iterations=1)

    if debug:
        cv2.imwrite("debug_walls.png", walls)
        cv2.imwrite("debug_floors.png", floors)
        cv2.imwrite("debug_floors_eroded.png", floors_eroded)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(floors_eroded, connectivity=8)

    regions = []
    for i in range(1, num_labels):
        x, y, w_comp, h_comp, area = stats[i]
        if area < 500:
            continue

        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            regions.append(PixelRegion(
                x=x, y=y, w=w_comp, h=h_comp, area=area, contour=contours[0]
            ))

    print(f"Found {len(regions)} regions with morphology method")

    if debug:
        debug_img = img.copy()
        for i, region in enumerate(sorted(regions, key=lambda r: r.area, reverse=True)):
            cv2.rectangle(debug_img, (region.x, region.y),
                         (region.x + region.w, region.y + region.h), (0, 255, 255), 2)
            cv2.putText(debug_img, f"{i}:{region.area}", (region.x + 5, region.y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite("debug_morph.png", debug_img)
        print("Saved debug_morph.png")

    return regions


def solve_transformation(grid_rooms: List[GridRoom], pixel_regions: List[PixelRegion]) -> Tuple[float, float, float, float]:
    """
    Solve for transformation parameters.

    The transformation is:
    pixel_x = grid_x * scale_x + offset_x
    pixel_y = grid_y * scale_y + offset_y

    We need to find correspondences between grid rooms and pixel regions.
    """

    # Filter out corridors from grid rooms
    main_rooms = [r for r in grid_rooms if not r.is_corridor]
    print(f"\nMain rooms in JSON (non-corridor): {len(main_rooms)}")
    for i, room in enumerate(main_rooms):
        print(f"  Room {i}: grid({room.x}, {room.y}) size({room.w}x{room.h}) area={room.area}")

    # Sort pixel regions by area (descending)
    pixel_sorted = sorted(pixel_regions, key=lambda r: r.area, reverse=True)
    print(f"\nLargest pixel regions:")
    for i, region in enumerate(pixel_sorted[:10]):
        print(f"  Region {i}: pixel({region.x}, {region.y}) size({region.w}x{region.h}) area={region.area}")

    # Sort grid rooms by area (descending)
    grid_sorted = sorted(main_rooms, key=lambda r: r.area, reverse=True)

    # Try to match based on size ratios
    # The largest grid room should correspond to the largest pixel region (roughly)

    # First attempt: match the largest rooms
    if len(pixel_sorted) < 2 or len(grid_sorted) < 2:
        print("Not enough rooms to solve transformation")
        return None

    # Use the two largest rooms for initial estimation
    best_scale_x = None
    best_scale_y = None
    best_offset_x = None
    best_offset_y = None
    best_error = float('inf')

    # Try different matching combinations of the largest rooms
    for g1, g2 in itertools.combinations(grid_sorted[:6], 2):
        for p1, p2 in itertools.combinations(pixel_sorted[:min(10, len(pixel_sorted))], 2):
            # Estimate scale from the two pairs
            # scale = (pixel_delta) / (grid_delta)

            grid_dx = g2.center_x - g1.center_x
            grid_dy = g2.center_y - g1.center_y
            pixel_dx = p2.center_x - p1.center_x
            pixel_dy = p2.center_y - p1.center_y

            if abs(grid_dx) < 0.5 or abs(grid_dy) < 0.5:
                continue

            scale_x = pixel_dx / grid_dx
            scale_y = pixel_dy / grid_dy

            # Check if scale is reasonable (positive, similar in x and y)
            if scale_x <= 0 or scale_y <= 0:
                continue

            # Estimate offset using first room
            offset_x = p1.center_x - g1.center_x * scale_x
            offset_y = p1.center_y - g1.center_y * scale_y

            # Validate by computing error for all rooms
            total_error = 0
            matched = 0
            for gr in grid_sorted[:6]:
                pred_x = gr.center_x * scale_x + offset_x
                pred_y = gr.center_y * scale_y + offset_y

                # Find closest pixel region
                min_dist = float('inf')
                for pr in pixel_sorted[:10]:
                    dist = np.sqrt((pred_x - pr.center_x)**2 + (pred_y - pr.center_y)**2)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist < 200:  # Within reasonable distance
                    matched += 1
                    total_error += min_dist

            if matched >= 4 and total_error < best_error:
                best_error = total_error
                best_scale_x = scale_x
                best_scale_y = scale_y
                best_offset_x = offset_x
                best_offset_y = offset_y
                print(f"New best: scale=({scale_x:.2f}, {scale_y:.2f}) offset=({offset_x:.2f}, {offset_y:.2f}) error={total_error:.2f} matched={matched}")

    return best_scale_x, best_scale_y, best_offset_x, best_offset_y


def solve_transformation_v2(grid_rooms: List[GridRoom], img_path: str) -> Tuple[float, float, float, float]:
    """
    Alternative approach: Use the known structure of the dungeon.

    Looking at the JSON, the rooms are arranged roughly in a horizontal line.
    The x-coordinates range from about -30 to 0.

    We can try to estimate the scale by looking at the width ratios.
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    # Calculate grid bounds
    min_gx = min(r.x for r in main_rooms)
    max_gx = max(r.x + r.w for r in main_rooms)
    min_gy = min(r.y for r in main_rooms)
    max_gy = max(r.y + r.h for r in main_rooms)

    grid_width = max_gx - min_gx
    grid_height = max_gy - min_gy

    print(f"\nGrid bounds: x=[{min_gx}, {max_gx}] y=[{min_gy}, {max_gy}]")
    print(f"Grid size: {grid_width} x {grid_height}")

    # Looking at the image, there's a border/margin around the dungeon
    # The dungeon content appears to be roughly centered

    # Estimate the dungeon area in the image
    # Based on the image, the dungeon spans roughly the middle portion

    # Let's detect the actual dungeon extent by finding dark pixels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find where the dungeon is (dark lines exist)
    _, dark_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find the bounding box of all dark pixels
    coords = cv2.findNonZero(dark_mask)
    if coords is not None:
        px, py, pw, ph = cv2.boundingRect(coords)
        print(f"\nDungeon extent in pixels: x={px}, y={py}, w={pw}, h={ph}")

        # Estimate scale
        # The dungeon content should map to this pixel region
        scale_x = pw / grid_width
        scale_y = ph / grid_height

        # Estimate offset
        # min_gx should map to px
        offset_x = px - min_gx * scale_x
        offset_y = py - min_gy * scale_y

        print(f"Initial estimate: scale=({scale_x:.2f}, {scale_y:.2f}) offset=({offset_x:.2f}, {offset_y:.2f})")

        return scale_x, scale_y, offset_x, offset_y

    return None


def refine_transformation(grid_rooms: List[GridRoom], img_path: str,
                          initial_params: Tuple[float, float, float, float],
                          debug: bool = True) -> Tuple[float, float, float, float]:
    """
    Refine transformation by matching specific features.
    """
    scale_x, scale_y, offset_x, offset_y = initial_params

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    main_rooms = [r for r in grid_rooms if not r.is_corridor]

    if debug:
        debug_img = img.copy()

        # Draw predicted room locations
        for room in main_rooms:
            # Convert grid corners to pixels
            x1 = int(room.x * scale_x + offset_x)
            y1 = int(room.y * scale_y + offset_y)
            x2 = int((room.x + room.w) * scale_x + offset_x)
            y2 = int((room.y + room.h) * scale_y + offset_y)

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite("debug_initial_transform.png", debug_img)
        print("Saved debug_initial_transform.png")

    return scale_x, scale_y, offset_x, offset_y


def detect_individual_rooms(img_path: str, debug: bool = True) -> List[PixelRegion]:
    """
    More sophisticated room detection that looks for enclosed rectangular areas.
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find edges (walls are drawn as lines)
    edges = cv2.Canny(gray, 50, 150)

    # Dilate to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    if debug:
        cv2.imwrite("debug_edges.png", edges)
        cv2.imwrite("debug_edges_dilated.png", edges_dilated)

    # Invert and find connected white (floor) regions
    inv = cv2.bitwise_not(edges_dilated)

    # Find contours
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 2000 or area > w * h * 0.5:  # Filter by size
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        # Check if roughly rectangular (dungeon rooms are rectangular)
        rect_area = cw * ch
        fill_ratio = area / rect_area if rect_area > 0 else 0

        if fill_ratio > 0.5:  # At least 50% fill
            regions.append(PixelRegion(
                x=x, y=y, w=cw, h=ch, area=area, contour=contour
            ))

    print(f"Found {len(regions)} rectangular regions")

    if debug:
        debug_img = img.copy()
        for i, region in enumerate(sorted(regions, key=lambda r: r.area, reverse=True)[:20]):
            cv2.rectangle(debug_img, (region.x, region.y),
                         (region.x + region.w, region.y + region.h), (0, 255, 0), 2)
        cv2.imwrite("debug_rect_regions.png", debug_img)
        print("Saved debug_rect_regions.png")

    return regions


def analyze_watabou_structure(json_path: str, img_path: str):
    """
    Main analysis function.
    """
    print("=" * 60)
    print("Watabou Coordinate Transformation Analysis")
    print("=" * 60)

    # Load data
    grid_rooms = load_json_rooms(json_path)
    print(f"\nLoaded {len(grid_rooms)} rooms from JSON")

    # Try multiple detection methods
    print("\n--- Method 1: Contour Detection ---")
    regions1 = detect_rooms_contours(img_path, debug=True)

    print("\n--- Method 2: Morphology ---")
    regions2 = detect_rooms_morphology(img_path, debug=True)

    print("\n--- Method 3: Edge-based ---")
    regions3 = detect_individual_rooms(img_path, debug=True)

    # Try solving transformation with different region sets
    print("\n--- Solving Transformation ---")

    # Method 1: Use image bounds
    params = solve_transformation_v2(grid_rooms, img_path)
    if params:
        scale_x, scale_y, offset_x, offset_y = params
        params = refine_transformation(grid_rooms, img_path, params, debug=True)

    # Try with detected regions
    print("\n--- Trying region-based matching ---")
    for name, regions in [("contours", regions1), ("morphology", regions2), ("edges", regions3)]:
        if len(regions) >= 4:
            print(f"\nTrying {name} regions...")
            result = solve_transformation(grid_rooms, regions)
            if result and result[0] is not None:
                print(f"  Result: scale=({result[0]:.2f}, {result[1]:.2f}) offset=({result[2]:.2f}, {result[3]:.2f})")

    return params


def validate_transformation(json_path: str, img_path: str,
                           scale_x: float, scale_y: float,
                           offset_x: float, offset_y: float):
    """
    Validate the transformation by overlaying predicted room locations on the image.
    """
    grid_rooms = load_json_rooms(json_path)
    img = cv2.imread(img_path)

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    print(f"Transformation: pixel = grid * scale + offset")
    print(f"  scale_x = {scale_x:.4f}")
    print(f"  scale_y = {scale_y:.4f}")
    print(f"  offset_x = {offset_x:.4f}")
    print(f"  offset_y = {offset_y:.4f}")

    debug_img = img.copy()

    main_rooms = [r for r in grid_rooms if not r.is_corridor]
    corridors = [r for r in grid_rooms if r.is_corridor]

    print(f"\nPredicted room locations:")
    for i, room in enumerate(main_rooms):
        # Convert grid corners to pixels
        x1 = int(room.x * scale_x + offset_x)
        y1 = int(room.y * scale_y + offset_y)
        x2 = int((room.x + room.w) * scale_x + offset_x)
        y2 = int((room.y + room.h) * scale_y + offset_y)

        print(f"  Room {i}: grid({room.x},{room.y},{room.w}x{room.h}) -> pixel({x1},{y1}) to ({x2},{y2})")

        # Draw on image
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(debug_img, f"R{i}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw corridors
    for room in corridors:
        x1 = int(room.x * scale_x + offset_x)
        y1 = int(room.y * scale_y + offset_y)
        x2 = int((room.x + room.w) * scale_x + offset_x)
        y2 = int((room.y + room.h) * scale_y + offset_y)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imwrite("debug_validation.png", debug_img)
    print("\nSaved debug_validation.png")

    return debug_img


if __name__ == "__main__":
    json_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.json"
    img_path = r"C:\Users\shini\Downloads\watabou_exports\chapel_of_supremus.png"

    # Run analysis
    params = analyze_watabou_structure(json_path, img_path)

    if params:
        scale_x, scale_y, offset_x, offset_y = params
        validate_transformation(json_path, img_path, scale_x, scale_y, offset_x, offset_y)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if params:
        print(f"scale_x = {params[0]:.4f}")
        print(f"scale_y = {params[1]:.4f}")
        print(f"offset_x = {params[2]:.4f}")
        print(f"offset_y = {params[3]:.4f}")
