"""
Mask to Walls v3 - Fixes for v2 issues:
1. Smart corner extension (line intersection instead of junction snapping)
2. Proper endpoint merging (extend along wall direction, not centroid averaging)
3. Door-wall alignment fix (snap doors after all wall processing)
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from scipy.spatial import KDTree
import math
from collections import defaultdict


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
        'corner_merge_distance': 10,  # Distance to look for corners to merge
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

    kernel_size = params['wall_offset'] * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    wall_eroded = cv2.erode(wall_mask, kernel)

    contours, _ = cv2.findContours(wall_eroded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    print(f"  Found {len(contours)} wall contours")

    walls = []
    for contour in contours:
        if len(contour) < 3:
            continue

        simplified = cv2.approxPolyDP(contour, params['epsilon'], closed=True)
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
            door_width = int(w * door_width_mult)
            doors.append({
                'x1': cx - door_width // 2,
                'y1': cy,
                'x2': cx + door_width // 2,
                'y2': cy,
                'type': 'door'
            })
        else:
            door_height = int(h * door_width_mult)
            doors.append({
                'x1': cx,
                'y1': cy - door_height // 2,
                'x2': cx,
                'y2': cy + door_height // 2,
                'type': 'door'
            })

    return doors


def get_segment_direction(seg):
    """Get normalized direction vector of a segment."""
    dx = seg['x2'] - seg['x1']
    dy = seg['y2'] - seg['y1']
    length = math.sqrt(dx*dx + dy*dy)
    if length < 0.001:
        return (0, 0)
    return (dx / length, dy / length)


def get_segment_angle(seg):
    """Get angle of segment in radians."""
    dx = seg['x2'] - seg['x1']
    dy = seg['y2'] - seg['y1']
    return math.atan2(dy, dx)


def is_perpendicular(seg1, seg2, tolerance_deg=15):
    """Check if two segments are roughly perpendicular."""
    angle1 = get_segment_angle(seg1)
    angle2 = get_segment_angle(seg2)
    diff = abs(angle1 - angle2)
    # Normalize to 0-180
    diff = diff % math.pi
    # Check if close to 90 degrees
    tolerance_rad = math.radians(tolerance_deg)
    return abs(diff - math.pi/2) < tolerance_rad


def line_line_intersection(p1, d1, p2, d2):
    """
    Find intersection of two lines defined by point and direction.
    Returns intersection point or None if parallel.
    """
    # Line 1: p1 + t * d1
    # Line 2: p2 + s * d2
    # Solve: p1 + t*d1 = p2 + s*d2

    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-10:
        return None  # Parallel

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    t = (dx * (-d2[1]) - dy * (-d2[0])) / det

    ix = p1[0] + t * d1[0]
    iy = p1[1] + t * d1[1]

    return (ix, iy)


def extend_segment_to_point(seg, endpoint_idx, target_point):
    """
    Extend a segment so that the specified endpoint reaches target_point.
    endpoint_idx: 1 for (x1,y1), 2 for (x2,y2)
    Returns new segment.
    """
    new_seg = seg.copy()
    if endpoint_idx == 1:
        new_seg['x1'] = int(round(target_point[0]))
        new_seg['y1'] = int(round(target_point[1]))
    else:
        new_seg['x2'] = int(round(target_point[0]))
        new_seg['y2'] = int(round(target_point[1]))
    return new_seg


def smart_corner_extension(segments, params):
    """
    V3 FIX 1: Smart corner extension using line intersection.
    Find clusters of nearby endpoints and extend walls to meet properly.
    """
    if not segments:
        return segments

    merge_dist = params['corner_merge_distance']

    # Build endpoint info: (point, segment_index, endpoint_idx)
    endpoint_info = []
    for i, seg in enumerate(segments):
        endpoint_info.append(((seg['x1'], seg['y1']), i, 1))
        endpoint_info.append(((seg['x2'], seg['y2']), i, 2))

    # Build KDTree for clustering
    points = np.array([info[0] for info in endpoint_info])
    tree = KDTree(points)

    # Find clusters of nearby endpoints
    visited = set()
    clusters = []

    for i in range(len(points)):
        if i in visited:
            continue

        indices = tree.query_ball_point(points[i], merge_dist)
        if len(indices) > 1:
            cluster = [endpoint_info[j] for j in indices]
            clusters.append(cluster)
            visited.update(indices)
        else:
            visited.add(i)

    print(f"  Found {len(clusters)} endpoint clusters to merge")

    # Process each cluster
    # Map: segment_index -> {endpoint_idx -> new_point}
    adjustments = defaultdict(dict)

    corners_extended = 0

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Get the segments involved
        cluster_segs = [(segments[info[1]], info[1], info[2]) for info in cluster]

        # Check if we have perpendicular walls (L-corner)
        if len(cluster_segs) == 2:
            seg1, idx1, ep1 = cluster_segs[0]
            seg2, idx2, ep2 = cluster_segs[1]

            if seg1['type'] == 'wall' and seg2['type'] == 'wall':
                if is_perpendicular(seg1, seg2):
                    # Extend both walls to their intersection
                    d1 = get_segment_direction(seg1)
                    d2 = get_segment_direction(seg2)

                    p1 = (seg1['x1'], seg1['y1']) if ep1 == 1 else (seg1['x2'], seg1['y2'])
                    p2 = (seg2['x1'], seg2['y1']) if ep2 == 1 else (seg2['x2'], seg2['y2'])

                    intersection = line_line_intersection(p1, d1, p2, d2)

                    if intersection:
                        adjustments[idx1][ep1] = intersection
                        adjustments[idx2][ep2] = intersection
                        corners_extended += 1
                        continue

        # For non-perpendicular or >2 segments, use centroid but only if very close
        cluster_points = [info[0] for info in cluster]
        max_dist = max(
            math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            for p1 in cluster_points for p2 in cluster_points
        )

        if max_dist <= 3:  # Only centroid for very close points
            centroid = (
                sum(p[0] for p in cluster_points) / len(cluster_points),
                sum(p[1] for p in cluster_points) / len(cluster_points)
            )
            for point, seg_idx, ep_idx in cluster:
                adjustments[seg_idx][ep_idx] = centroid

    print(f"  Extended {corners_extended} L-corners via line intersection")

    # Apply adjustments
    new_segments = []
    for i, seg in enumerate(segments):
        new_seg = seg.copy()
        if i in adjustments:
            if 1 in adjustments[i]:
                new_seg['x1'] = int(round(adjustments[i][1][0]))
                new_seg['y1'] = int(round(adjustments[i][1][1]))
            if 2 in adjustments[i]:
                new_seg['x2'] = int(round(adjustments[i][2][0]))
                new_seg['y2'] = int(round(adjustments[i][2][1]))
        new_segments.append(new_seg)

    return new_segments


def straighten_walls(segments, params):
    """Straighten near-horizontal/vertical walls."""
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

        near_horizontal = (abs(angle) < threshold_rad or
                          abs(angle - math.pi) < threshold_rad or
                          abs(angle + math.pi) < threshold_rad)

        near_vertical = (abs(angle - math.pi/2) < threshold_rad or
                        abs(angle + math.pi/2) < threshold_rad)

        if near_horizontal:
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


def line_segment_intersection(p1, p2, p3, p4):
    """
    Find intersection point of line segment (p1,p2) with line segment (p3,p4).
    Returns (x, y, t, u) where t and u are parameters [0,1] if intersection is within segments.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    tol = 0.01
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy, t, u)

    return None


def split_walls_at_doors(segments, params):
    """Split wall segments where doors intersect them."""
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

        intersections = []

        for door in doors:
            door_p1 = (door['x1'], door['y1'])
            door_p2 = (door['x2'], door['y2'])

            result = line_segment_intersection(wall_p1, wall_p2, door_p1, door_p2)

            if result is not None:
                ix, iy, t, u = result
                if 0.01 < t < 0.99:
                    intersections.append((t, int(round(ix)), int(round(iy))))

        if not intersections:
            new_walls.append(wall)
        else:
            intersections.sort(key=lambda x: x[0])
            current_start = wall_p1

            for t, ix, iy in intersections:
                new_walls.append({
                    'x1': current_start[0],
                    'y1': current_start[1],
                    'x2': ix,
                    'y2': iy,
                    'type': 'wall'
                })
                current_start = (ix, iy)
                splits_made += 1

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


def snap_doors_to_wall_endpoints(segments, snap_distance=5):
    """
    V3 FIX 3: Snap door endpoints to nearest wall endpoints AFTER all wall processing.
    This ensures doors align with the final wall positions.
    """
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors or not walls:
        return segments

    print(f"  Re-snapping {len(doors)} door endpoints to final wall positions")

    # Collect all wall endpoints
    wall_endpoints = []
    for wall in walls:
        wall_endpoints.append([wall['x1'], wall['y1']])
        wall_endpoints.append([wall['x2'], wall['y2']])

    wall_endpoints = np.array(wall_endpoints)
    tree = KDTree(wall_endpoints)

    snapped_doors = []
    snapped_count = 0

    for door in doors:
        p1 = np.array([door['x1'], door['y1']])
        p2 = np.array([door['x2'], door['y2']])

        dist1, idx1 = tree.query(p1)
        dist2, idx2 = tree.query(p2)

        new_door = door.copy()

        if dist1 <= snap_distance:
            new_door['x1'] = int(wall_endpoints[idx1][0])
            new_door['y1'] = int(wall_endpoints[idx1][1])
            snapped_count += 1

        if dist2 <= snap_distance:
            new_door['x2'] = int(wall_endpoints[idx2][0])
            new_door['y2'] = int(wall_endpoints[idx2][1])
            snapped_count += 1

        snapped_doors.append(new_door)

    print(f"  Snapped {snapped_count} door endpoints")

    return walls + snapped_doors


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
    parser.add_argument("--output", type=str, default="wall_visualization_v3.png")
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

    # Step 1: Extract walls
    print("\nStep 1: Extracting walls from wall mask...")
    walls = extract_walls_from_wall_mask(mask, params)
    print(f"  Extracted {len(walls)} wall segments")

    # Step 2: Extract doors
    print("\nStep 2: Extracting doors from door mask...")
    doors = extract_doors(mask, params)
    print(f"  Extracted {len(doors)} door segments")

    # Step 3: Combine (don't snap doors to walls yet)
    all_segments = walls + doors
    print(f"\nStep 3: Combined {len(all_segments)} segments")

    # Step 4: Filter short segments
    print("\nStep 4: Initial filtering...")
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    # Step 5: Straighten walls (before corner extension)
    if not args.no_straighten:
        print("\nStep 5: Straightening near-axis walls...")
        all_segments = straighten_walls(all_segments, params)
    else:
        print("\nStep 5: Skipping wall straightening (--no-straighten)")

    # Step 6: V3 FIX - Smart corner extension (line intersection)
    print("\nStep 6: Smart corner extension (line intersection)...")
    all_segments = smart_corner_extension(all_segments, params)

    # Step 7: Split walls at door intersections
    print("\nStep 7: Splitting walls at door intersections...")
    all_segments = split_walls_at_doors(all_segments, params)

    # Step 8: V3 FIX - Re-snap doors to final wall positions
    print("\nStep 8: Snapping doors to final wall positions...")
    all_segments = snap_doors_to_wall_endpoints(all_segments, snap_distance=8)

    # Step 9: Final cleanup
    print("\nStep 9: Final cleanup...")
    all_segments = filter_zero_length_segments(all_segments)
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    print(f"\nFinal segment count:")
    print(f"  Walls: {sum(1 for s in all_segments if s['type'] == 'wall')}")
    print(f"  Doors: {sum(1 for s in all_segments if s['type'] == 'door')}")
    print(f"  Total: {len(all_segments)}")

    visualize_walls(mask, all_segments, args.output)


if __name__ == "__main__":
    main()
