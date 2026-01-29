"""
Mask to Walls v2 - With TODO improvements:
1. Extend wall endpoints to junctions
2. Straighten near-horizontal/vertical walls (optional)
3. Split walls at door intersections
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from scipy.spatial import KDTree
import math


def calculate_params_from_scale(grid_size_pixels=70):
    """Auto-calculate parameters based on grid/wall scale."""
    wall_offset = max(3, int(grid_size_pixels * 0.08))
    epsilon = max(2, int(grid_size_pixels * 0.05))
    snap_distance = max(2, int(grid_size_pixels * 0.04))

    return {
        'wall_offset': wall_offset,
        'epsilon': epsilon,
        'snap_distance': snap_distance,
        'door_width_mult': 1.2,
        'min_wall_length': 5,
        'junction_snap_distance': 8,  # Distance to snap endpoints to junctions
        'straighten_threshold': 5,  # Degrees threshold for straightening
    }


def contour_to_segments(contour):
    """Convert contour points to line segments."""
    segments = []
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        segments.append({
            'x1': int(p1[0]),
            'y1': int(p1[1]),
            'x2': int(p2[0]),
            'y2': int(p2[1]),
            'type': 'wall'
        })
    return segments


def extract_walls_from_wall_mask(mask, params):
    """Extract walls from the wall mask (black pixels, value 0)."""
    wall_mask = (mask == 0).astype(np.uint8) * 255

    if not wall_mask.any():
        print("  No wall regions found")
        return []

    print(f"  Processing walls from wall mask")

    # Erode wall mask inward by wall_offset to place walls inside wall regions
    kernel_size = params['wall_offset'] * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    wall_eroded = cv2.erode(wall_mask, kernel)

    # Find contours on eroded wall boundary
    contours, _ = cv2.findContours(wall_eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    print(f"  Found {len(contours)} wall contours")

    walls = []
    for contour in contours:
        if len(contour) < 3:
            continue

        # Simplify with Douglas-Peucker
        simplified = cv2.approxPolyDP(contour, params['epsilon'], closed=True)

        # Convert to segments
        segments = contour_to_segments(simplified)
        walls.extend(segments)

    return walls


def extract_doors(mask, params):
    """Extract door segments from door mask (gray pixels, value 127)."""
    door_mask = (mask == 127).astype(np.uint8) * 255
    wall_mask = (mask == 0).astype(np.uint8) * 255

    if not door_mask.any():
        print("  No doors found")
        return []

    contours, _ = cv2.findContours(door_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doors = []
    print(f"  Found {len(contours)} door regions")

    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(contour)

        # Sample walls around door to determine orientation
        search_radius = max(w, h) + 10
        y1 = max(0, cy - search_radius)
        y2 = min(mask.shape[0], cy + search_radius)
        x1 = max(0, cx - search_radius)
        x2 = min(mask.shape[1], cx + search_radius)

        region = wall_mask[y1:y2, x1:x2]

        left_walls = region[search_radius, :search_radius].sum()
        right_walls = region[search_radius, search_radius:].sum()
        horizontal_walls = left_walls + right_walls

        top_walls = region[:search_radius, search_radius].sum()
        bottom_walls = region[search_radius:, search_radius].sum()
        vertical_walls = top_walls + bottom_walls

        door_width_mult = params['door_width_mult']

        if horizontal_walls > vertical_walls:
            # Walls on left/right -> door is horizontal
            door_width = int(w * door_width_mult)
            doors.append({
                'x1': cx - door_width // 2,
                'y1': cy,
                'x2': cx + door_width // 2,
                'y2': cy,
                'type': 'door'
            })
        else:
            # Walls on top/bottom -> door is vertical
            door_height = int(h * door_width_mult)
            doors.append({
                'x1': cx,
                'y1': cy - door_height // 2,
                'x2': cx,
                'y2': cy + door_height // 2,
                'type': 'door'
            })

    return doors


def find_junctions(segments):
    """Find all junction points where multiple segments meet."""
    endpoint_counts = {}

    for seg in segments:
        p1 = (seg['x1'], seg['y1'])
        p2 = (seg['x2'], seg['y2'])

        endpoint_counts[p1] = endpoint_counts.get(p1, 0) + 1
        endpoint_counts[p2] = endpoint_counts.get(p2, 0) + 1

    # Junctions are points where 2+ segments meet
    junctions = [point for point, count in endpoint_counts.items() if count >= 2]
    return junctions


def extend_endpoints_to_junctions(segments, params):
    """
    IMPROVEMENT 1: Extend wall endpoints to nearby junction points.
    This ensures walls connect seamlessly.
    """
    if not segments:
        return segments

    snap_dist = params['junction_snap_distance']

    # Find all junctions
    junctions = find_junctions(segments)

    if not junctions:
        print("  No junctions found")
        return segments

    print(f"  Found {len(junctions)} junction points")

    # Build KDTree for junctions
    junction_array = np.array(junctions)
    tree = KDTree(junction_array)

    # For each segment endpoint, snap to nearest junction if close enough
    extended = []
    snapped_count = 0

    for seg in segments:
        p1 = np.array([seg['x1'], seg['y1']])
        p2 = np.array([seg['x2'], seg['y2']])

        # Check if p1 is near a junction
        dist1, idx1 = tree.query(p1)
        if dist1 <= snap_dist and dist1 > 0:  # Don't snap if already at junction
            p1 = junction_array[idx1]
            snapped_count += 1

        # Check if p2 is near a junction
        dist2, idx2 = tree.query(p2)
        if dist2 <= snap_dist and dist2 > 0:
            p2 = junction_array[idx2]
            snapped_count += 1

        extended.append({
            'x1': int(p1[0]),
            'y1': int(p1[1]),
            'x2': int(p2[0]),
            'y2': int(p2[1]),
            'type': seg['type']
        })

    print(f"  Snapped {snapped_count} endpoints to junctions")
    return extended


def straighten_walls(segments, params):
    """
    IMPROVEMENT 2: Straighten near-horizontal/vertical walls.
    If a wall is within threshold degrees of axis-aligned, snap it.
    """
    threshold_deg = params['straighten_threshold']
    threshold_rad = math.radians(threshold_deg)

    straightened = []
    count = 0

    for seg in segments:
        if seg['type'] != 'wall':
            straightened.append(seg)
            continue

        dx = seg['x2'] - seg['x1']
        dy = seg['y2'] - seg['y1']

        if dx == 0 and dy == 0:
            straightened.append(seg)
            continue

        angle = math.atan2(dy, dx)
        length = math.sqrt(dx*dx + dy*dy)

        # Check if near horizontal (0 or 180 degrees)
        near_horizontal = (abs(angle) < threshold_rad or
                          abs(angle - math.pi) < threshold_rad or
                          abs(angle + math.pi) < threshold_rad)

        # Check if near vertical (90 or -90 degrees)
        near_vertical = (abs(angle - math.pi/2) < threshold_rad or
                        abs(angle + math.pi/2) < threshold_rad)

        if near_horizontal:
            # Make perfectly horizontal - adjust y2 to match y1
            mid_y = (seg['y1'] + seg['y2']) // 2
            straightened.append({
                'x1': seg['x1'],
                'y1': mid_y,
                'x2': seg['x2'],
                'y2': mid_y,
                'type': 'wall'
            })
            count += 1
        elif near_vertical:
            # Make perfectly vertical - adjust x2 to match x1
            mid_x = (seg['x1'] + seg['x2']) // 2
            straightened.append({
                'x1': mid_x,
                'y1': seg['y1'],
                'x2': mid_x,
                'y2': seg['y2'],
                'type': 'wall'
            })
            count += 1
        else:
            straightened.append(seg)

    print(f"  Straightened {count} near-axis walls")
    return straightened


def line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line segment (p1,p2) with line segment (p3,p4).
    Returns (x, y, t, u) where t and u are parameters [0,1] if intersection is within segments.
    Returns None if lines are parallel or don't intersect within segments.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both segments (with small tolerance)
    tol = 0.01
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy, t, u)

    return None


def split_walls_at_doors(segments, params):
    """
    IMPROVEMENT 3: Split wall segments where doors intersect them.
    This creates proper wall breaks at doorways.
    """
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors:
        print("  No doors to process")
        return segments

    print(f"  Processing {len(doors)} doors against {len(walls)} walls")

    new_walls = []
    splits_made = 0

    for wall in walls:
        wall_p1 = (wall['x1'], wall['y1'])
        wall_p2 = (wall['x2'], wall['y2'])

        # Collect all intersection points with doors
        intersections = []

        for door in doors:
            door_p1 = (door['x1'], door['y1'])
            door_p2 = (door['x2'], door['y2'])

            result = line_intersection(wall_p1, wall_p2, door_p1, door_p2)

            if result is not None:
                ix, iy, t, u = result
                # Only add if intersection is not at wall endpoints
                if 0.01 < t < 0.99:
                    intersections.append((t, int(round(ix)), int(round(iy))))

        if not intersections:
            # No intersections, keep wall as-is
            new_walls.append(wall)
        else:
            # Sort intersections by parameter t (distance along wall)
            intersections.sort(key=lambda x: x[0])

            # Split wall at each intersection point
            current_start = wall_p1

            for t, ix, iy in intersections:
                # Create segment from current start to intersection
                new_walls.append({
                    'x1': current_start[0],
                    'y1': current_start[1],
                    'x2': ix,
                    'y2': iy,
                    'type': 'wall'
                })
                current_start = (ix, iy)
                splits_made += 1

            # Create final segment from last intersection to wall end
            new_walls.append({
                'x1': current_start[0],
                'y1': current_start[1],
                'x2': wall_p2[0],
                'y2': wall_p2[1],
                'type': 'wall'
            })

    print(f"  Split walls at {splits_made} door intersections")
    print(f"  Wall count: {len(walls)} -> {len(new_walls)}")

    return new_walls + doors


def connect_doors_to_walls(walls, doors, snap_distance=10):
    """Snap door endpoints to nearby wall endpoints."""
    if not doors or not walls:
        return walls + doors

    print(f"  Connecting door endpoints to walls (snap distance: {snap_distance}px)")

    # Collect all wall endpoints
    wall_endpoints = []
    for wall in walls:
        wall_endpoints.append([wall['x1'], wall['y1']])
        wall_endpoints.append([wall['x2'], wall['y2']])

    if not wall_endpoints:
        return walls + doors

    wall_endpoints = np.array(wall_endpoints)
    tree = KDTree(wall_endpoints)

    # Snap door endpoints to nearest wall endpoints
    snapped_doors = []
    for door in doors:
        # Find nearest wall endpoint for each door endpoint
        p1 = np.array([door['x1'], door['y1']])
        p2 = np.array([door['x2'], door['y2']])

        dist1, idx1 = tree.query(p1)
        dist2, idx2 = tree.query(p2)

        # If within snap distance, use wall endpoint
        if dist1 < snap_distance:
            p1_snapped = wall_endpoints[idx1]
        else:
            p1_snapped = p1

        if dist2 < snap_distance:
            p2_snapped = wall_endpoints[idx2]
        else:
            p2_snapped = p2

        snapped_doors.append({
            'x1': int(p1_snapped[0]),
            'y1': int(p1_snapped[1]),
            'x2': int(p2_snapped[0]),
            'y2': int(p2_snapped[1]),
            'type': 'door'
        })

    return walls + snapped_doors


def snap_endpoints(segments, snap_distance):
    """Merge nearby endpoints."""
    if not segments:
        return segments

    print(f"  Snapping endpoints within {snap_distance}px")

    endpoints = []
    for seg in segments:
        endpoints.append([seg['x1'], seg['y1']])
        endpoints.append([seg['x2'], seg['y2']])

    endpoints = np.array(endpoints)
    tree = KDTree(endpoints)

    visited = set()
    clusters = []

    for i in range(len(endpoints)):
        if i in visited:
            continue

        indices = tree.query_ball_point(endpoints[i], snap_distance)

        if len(indices) > 1:
            cluster = [endpoints[j] for j in indices]
            clusters.append(cluster)
            visited.update(indices)
        else:
            visited.add(i)

    point_mapping = {}
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0).astype(int)
        for point in cluster:
            key = tuple(point)
            point_mapping[key] = tuple(centroid)

    snapped = []
    for seg in segments:
        p1 = (seg['x1'], seg['y1'])
        p2 = (seg['x2'], seg['y2'])

        p1_snapped = point_mapping.get(p1, p1)
        p2_snapped = point_mapping.get(p2, p2)

        snapped.append({
            'x1': p1_snapped[0],
            'y1': p1_snapped[1],
            'x2': p2_snapped[0],
            'y2': p2_snapped[1],
            'type': seg['type']
        })

    return snapped


def filter_short_segments(segments, min_length):
    """Remove very short segments."""
    filtered = []
    removed = 0

    for seg in segments:
        length = np.sqrt((seg['x2'] - seg['x1'])**2 + (seg['y2'] - seg['y1'])**2)
        if length >= min_length:
            filtered.append(seg)
        else:
            removed += 1

    if removed > 0:
        print(f"  Filtered {removed} short segments (<{min_length}px)")

    return filtered


def filter_zero_length_segments(segments):
    """Remove segments where start == end."""
    filtered = []
    removed = 0

    for seg in segments:
        if seg['x1'] == seg['x2'] and seg['y1'] == seg['y2']:
            removed += 1
        else:
            filtered.append(seg)

    if removed > 0:
        print(f"  Removed {removed} zero-length segments")

    return filtered


def visualize_walls(image, segments, output_path):
    """Create visualization."""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    endpoints = set()

    for seg in segments:
        p1 = (seg['x1'], seg['y1'])
        p2 = (seg['x2'], seg['y2'])

        endpoints.add(p1)
        endpoints.add(p2)

        if seg['type'] == 'door':
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            color = (255, 0, 0)  # Blue
            thickness = 2

        cv2.line(vis, p1, p2, color, thickness)

    for point in endpoints:
        cv2.circle(vis, point, 4, (0, 255, 255), -1)  # Yellow dots

    cv2.putText(vis, "Blue = Walls", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(vis, "Red = Doors", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "Yellow = Junctions", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imwrite(str(output_path), vis)
    print(f"\nVisualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output", type=str, default="wall_visualization_v2.png")
    parser.add_argument("--grid-size", type=int, default=70)
    parser.add_argument("--no-straighten", action="store_true", help="Disable wall straightening")

    args = parser.parse_args()
    params = calculate_params_from_scale(args.grid_size)

    print(f"Loading mask: {args.mask}")
    mask = np.array(Image.open(args.mask).convert('L'))

    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")

    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Step 1: Extract walls from wall mask
    print("\nStep 1: Extracting walls from wall mask...")
    walls = extract_walls_from_wall_mask(mask, params)
    print(f"  Extracted {len(walls)} wall segments")

    # Step 2: Extract doors from door mask
    print("\nStep 2: Extracting doors from door mask...")
    doors = extract_doors(mask, params)
    print(f"  Extracted {len(doors)} door segments")

    # Step 3: Connect doors to walls
    print("\nStep 3: Connecting doors to walls...")
    all_segments = connect_doors_to_walls(walls, doors, snap_distance=15)
    print(f"  Total segments: {len(all_segments)}")

    # Step 4: Initial filtering and snapping
    print("\nStep 4: Initial post-processing...")
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])
    all_segments = snap_endpoints(all_segments, params['snap_distance'])

    # Step 5: IMPROVEMENT 1 - Extend endpoints to junctions
    print("\nStep 5: Extending wall endpoints to junctions...")
    all_segments = extend_endpoints_to_junctions(all_segments, params)

    # Step 6: IMPROVEMENT 2 - Straighten walls (optional)
    if not args.no_straighten:
        print("\nStep 6: Straightening near-axis walls...")
        all_segments = straighten_walls(all_segments, params)
    else:
        print("\nStep 6: Skipping wall straightening (--no-straighten)")

    # Step 7: IMPROVEMENT 3 - Split walls at door intersections (do this LAST)
    print("\nStep 7: Splitting walls at door intersections...")
    all_segments = split_walls_at_doors(all_segments, params)

    # Step 8: Final cleanup
    print("\nStep 8: Final cleanup...")
    all_segments = filter_zero_length_segments(all_segments)
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    print(f"\nFinal segment count:")
    print(f"  Walls: {sum(1 for s in all_segments if s['type'] == 'wall')}")
    print(f"  Doors: {sum(1 for s in all_segments if s['type'] == 'door')}")
    print(f"  Total: {len(all_segments)}")

    visualize_walls(mask, all_segments, args.output)


if __name__ == "__main__":
    main()
