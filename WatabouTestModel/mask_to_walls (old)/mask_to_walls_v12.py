"""
Mask to Walls v12 - Bug fixes:
- Fixed: Only merge dangling WALL endpoints, not door endpoints (was causing door merge issues)
- Fixed: Add perimeter walls around map edges to close off edge-touching rooms
- Dangling wall endpoint merge within 25px
- Default epsilon=7 for cleaner simplification
- All v8 features: nearest wall door extension, 10px wall_offset, pre-split door filtering
"""

import cv2
import numpy as np
from PIL import Image
import argparse
from scipy.spatial import KDTree
import math
import json
from collections import defaultdict


def calculate_params_from_scale(grid_size_pixels=70, epsilon_override=None):
    """Auto-calculate parameters based on grid/wall scale."""
    # V7: Fixed wall_offset at 10px for better wall visibility
    wall_offset = 10
    epsilon = 7  # Fixed at 7 for cleaner wall simplification
    snap_distance = max(2, int(grid_size_pixels * 0.04))

    # Allow epsilon override independent of grid size
    if epsilon_override is not None:
        epsilon = epsilon_override

    return {
        'wall_offset': wall_offset,
        'epsilon': epsilon,
        'snap_distance': snap_distance,
        'door_width_mult': 1.2,
        'min_wall_length': 5,
        # Corner-rounding segments are approximately wall_offset * sqrt(2)
        # Filter anything shorter than wall_offset * 2 to remove them
        'corner_segment_threshold': wall_offset * 2,
        'corner_merge_distance': wall_offset * 3,
        'straighten_threshold': 5,
        'door_extend_max': 80,  # Increased to handle 10px wall_offset
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


def segment_length(seg):
    """Calculate segment length."""
    return math.sqrt((seg['x2'] - seg['x1'])**2 + (seg['y2'] - seg['y1'])**2)


def remove_corner_rounding_segments(segments, threshold):
    """
    V4 FIX: Remove small corner-rounding segments created by erosion.
    These are approximately wall_offset * sqrt(2) in length.

    When we remove a segment, we need to connect its neighbors directly.
    """
    if not segments:
        return segments

    # First pass: identify which segments to remove
    to_remove = set()
    for i, seg in enumerate(segments):
        if segment_length(seg) < threshold:
            to_remove.add(i)

    if not to_remove:
        return segments

    print(f"  Removing {len(to_remove)} corner-rounding segments (< {threshold:.1f}px)")

    # Build adjacency: for closed contour, segment i connects to segment i+1
    n = len(segments)

    # Create new segments list, skipping removed ones and connecting neighbors
    new_segments = []
    i = 0
    while i < n:
        if i not in to_remove:
            new_segments.append(segments[i].copy())
            i += 1
        else:
            # Find the segment before this run of removed segments
            # and the segment after
            start_idx = i
            while i < n and i in to_remove:
                i += 1
            # Now i is the first non-removed segment after the run
            # We need to connect the segment before start_idx to the segment at i
            # The endpoint of seg[start_idx-1] should connect to start of seg[i]

            # But for a closed contour, we need to handle wrap-around
            # For now, just skip the removed segments - the corner extension
            # will handle connecting the remaining segments

    return new_segments


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
    corner_threshold = params['corner_segment_threshold']

    for contour in contours:
        if len(contour) < 3:
            continue

        simplified = cv2.approxPolyDP(contour, params['epsilon'], closed=True)
        segments = contour_to_segments(simplified)

        # V4: Filter out corner-rounding segments
        before_count = len(segments)
        segments = [s for s in segments if segment_length(s) >= corner_threshold]
        removed = before_count - len(segments)
        if removed > 0:
            print(f"    Removed {removed} corner segments from contour")

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
    diff = diff % math.pi
    tolerance_rad = math.radians(tolerance_deg)
    return abs(diff - math.pi/2) < tolerance_rad


def line_line_intersection(p1, d1, p2, d2):
    """
    Find intersection of two lines defined by point and direction.
    Returns intersection point or None if parallel.
    """
    det = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
    if abs(det) < 1e-10:
        return None

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    t = (dx * (-d2[1]) - dy * (-d2[0])) / det

    ix = p1[0] + t * d1[0]
    iy = p1[1] + t * d1[1]

    return (ix, iy)


def smart_corner_extension(segments, params):
    """
    Smart corner extension using line intersection.
    After removing corner-rounding segments, extend walls to meet.
    """
    if not segments:
        return segments

    merge_dist = params['corner_merge_distance']
    print(f"  Using corner merge distance: {merge_dist}px")

    # Build endpoint info: (point, segment_index, endpoint_idx)
    endpoint_info = []
    for i, seg in enumerate(segments):
        endpoint_info.append(((seg['x1'], seg['y1']), i, 1))
        endpoint_info.append(((seg['x2'], seg['y2']), i, 2))

    points = np.array([info[0] for info in endpoint_info])
    tree = KDTree(points)

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

    adjustments = defaultdict(dict)
    corners_extended = 0

    # Debug: track cluster sizes
    cluster_sizes = defaultdict(int)
    for cluster in clusters:
        cluster_sizes[len(cluster)] += 1
    print(f"  Cluster size distribution: {dict(cluster_sizes)}")

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        cluster_segs = [(segments[info[1]], info[1], info[2]) for info in cluster]

        # Get only wall segments
        wall_segs = [(seg, idx, ep) for seg, idx, ep in cluster_segs if seg['type'] == 'wall']

        # For exactly 2 wall segments meeting at a corner
        if len(wall_segs) == 2:
            seg1, idx1, ep1 = wall_segs[0]
            seg2, idx2, ep2 = wall_segs[1]

            if is_perpendicular(seg1, seg2):
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

        if max_dist <= 5:  # Increased from 3 to 5
            centroid = (
                sum(p[0] for p in cluster_points) / len(cluster_points),
                sum(p[1] for p in cluster_points) / len(cluster_points)
            )
            for point, seg_idx, ep_idx in cluster:
                adjustments[seg_idx][ep_idx] = centroid

    print(f"  Extended {corners_extended} L-corners via line intersection")

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


def merge_dangling_endpoints(segments, merge_distance=25):
    """
    V12: Merge dangling WALL endpoints that are close together.

    Only merges WALL endpoints (not doors) that are NOT already connected to another segment.
    Connected endpoints (sharing exact coordinates with another segment) are left alone.

    Args:
        segments: List of segment dicts
        merge_distance: Max distance in pixels to merge dangling endpoints
    """
    if not segments:
        return segments

    # Build map of all endpoints: (x, y) -> list of (segment_index, endpoint_type)
    # endpoint_type: 1 = start (x1,y1), 2 = end (x2,y2)
    endpoint_map = defaultdict(list)
    for i, seg in enumerate(segments):
        p1 = (seg['x1'], seg['y1'])
        p2 = (seg['x2'], seg['y2'])
        endpoint_map[p1].append((i, 1))
        endpoint_map[p2].append((i, 2))

    # Find "dangling" endpoints - those that only have ONE segment at that location
    # (connected endpoints have 2+ segments sharing the same point)
    # V12: Only consider WALL segments, not doors
    dangling = []
    for point, seg_list in endpoint_map.items():
        if len(seg_list) == 1:
            seg_idx, ep_type = seg_list[0]
            # V12: Skip door segments - only merge wall endpoints
            if segments[seg_idx]['type'] == 'door':
                continue
            dangling.append((point, seg_idx, ep_type))

    if not dangling:
        print(f"  No dangling endpoints found")
        return segments

    print(f"  Found {len(dangling)} dangling endpoints")

    # Build KDTree of dangling endpoints to find close pairs
    dangling_points = np.array([d[0] for d in dangling])
    tree = KDTree(dangling_points)

    # Find clusters of close dangling endpoints
    visited = set()
    adjustments = {}  # seg_idx -> {ep_type -> new_point}
    merged_count = 0

    for i in range(len(dangling)):
        if i in visited:
            continue

        # Find all dangling endpoints within merge_distance
        indices = tree.query_ball_point(dangling_points[i], merge_distance)

        if len(indices) <= 1:
            visited.add(i)
            continue

        # Get the points and segment info for this cluster
        cluster_points = [dangling[j][0] for j in indices]
        cluster_info = [dangling[j] for j in indices]

        # Calculate average position
        avg_x = sum(p[0] for p in cluster_points) / len(cluster_points)
        avg_y = sum(p[1] for p in cluster_points) / len(cluster_points)
        new_point = (int(round(avg_x)), int(round(avg_y)))

        # Record adjustments for all endpoints in cluster
        for point, seg_idx, ep_type in cluster_info:
            if seg_idx not in adjustments:
                adjustments[seg_idx] = {}
            adjustments[seg_idx][ep_type] = new_point

        visited.update(indices)
        merged_count += 1

    print(f"  Merged {merged_count} dangling endpoint clusters")

    # Apply adjustments
    new_segments = []
    for i, seg in enumerate(segments):
        new_seg = seg.copy()
        if i in adjustments:
            if 1 in adjustments[i]:
                new_seg['x1'], new_seg['y1'] = adjustments[i][1]
            if 2 in adjustments[i]:
                new_seg['x2'], new_seg['y2'] = adjustments[i][2]
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
    """Find intersection point of line segment (p1,p2) with line segment (p3,p4)."""
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


def point_to_segment_distance(point, seg):
    """Calculate minimum distance from a point to a line segment."""
    px, py = point
    x1, y1 = seg['x1'], seg['y1']
    x2, y2 = seg['x2'], seg['y2']

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def filter_doors_crossing_walls(segments, max_distance=8):
    """
    V6: Filter doors BEFORE wall splitting.
    Only keep doors where BOTH endpoints are close to a wall segment.
    This prevents invalid doors from creating orphan wall splits.
    """
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors or not walls:
        return segments

    valid_doors = []
    removed_count = 0

    for door in doors:
        p1 = (door['x1'], door['y1'])
        p2 = (door['x2'], door['y2'])

        # Find minimum distance from each endpoint to any wall segment
        min_dist1 = float('inf')
        min_dist2 = float('inf')

        for wall in walls:
            dist1 = point_to_segment_distance(p1, wall)
            dist2 = point_to_segment_distance(p2, wall)
            min_dist1 = min(min_dist1, dist1)
            min_dist2 = min(min_dist2, dist2)

        # Door is valid only if BOTH endpoints are close to walls
        if min_dist1 <= max_distance and min_dist2 <= max_distance:
            valid_doors.append(door)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"  Filtered {removed_count} doors not crossing walls on both ends")

    return walls + valid_doors


def add_perimeter_walls(segments, image_size):
    """
    V12: Add walls around the perimeter of the map.

    This closes off any dungeon rooms that touch the edge of the map,
    preventing open endpoints at the boundaries.

    Args:
        segments: List of segment dicts
        image_size: Tuple of (height, width)
    """
    height, width = image_size

    # Create 4 perimeter wall segments (just inside the edge)
    margin = 1  # 1 pixel from edge
    perimeter_walls = [
        # Top edge
        {'x1': margin, 'y1': margin, 'x2': width - margin, 'y2': margin, 'type': 'wall'},
        # Bottom edge
        {'x1': margin, 'y1': height - margin, 'x2': width - margin, 'y2': height - margin, 'type': 'wall'},
        # Left edge
        {'x1': margin, 'y1': margin, 'x2': margin, 'y2': height - margin, 'type': 'wall'},
        # Right edge
        {'x1': width - margin, 'y1': margin, 'x2': width - margin, 'y2': height - margin, 'type': 'wall'},
    ]

    print(f"  Added 4 perimeter wall segments")

    return segments + perimeter_walls


def extend_doors_to_walls(segments, params):
    """
    Extend door segments until they intersect wall segments.
    This ensures doors connect to walls even if wall_offset pushed walls away.
    """
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors or not walls:
        return segments

    max_extend = params.get('door_extend_max', 50)
    print(f"  Extending doors to intersect walls (max {max_extend}px)")

    extended_doors = []
    doors_extended = 0

    for door in doors:
        # Get door direction
        dx = door['x2'] - door['x1']
        dy = door['y2'] - door['y1']
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.001:
            extended_doors.append(door)
            continue

        ux, uy = dx / length, dy / length

        # Try to extend both endpoints
        new_door = door.copy()
        extended = False

        # Extend endpoint 1 (backwards along direction)
        best_t1 = 0
        p1 = (door['x1'], door['y1'])
        for wall in walls:
            wall_p1 = (wall['x1'], wall['y1'])
            wall_p2 = (wall['x2'], wall['y2'])

            # Check intersection of extended door ray with wall segment
            # Door ray: p1 - t * (ux, uy) for t in [0, max_extend]
            for t in range(1, max_extend + 1):
                test_p = (p1[0] - t * ux, p1[1] - t * uy)
                # Check if test_p is on the wall segment
                wall_dx = wall_p2[0] - wall_p1[0]
                wall_dy = wall_p2[1] - wall_p1[1]
                wall_len = math.sqrt(wall_dx*wall_dx + wall_dy*wall_dy)
                if wall_len < 0.001:
                    continue

                # Project test_p onto wall line
                to_test = (test_p[0] - wall_p1[0], test_p[1] - wall_p1[1])
                proj = (to_test[0] * wall_dx + to_test[1] * wall_dy) / (wall_len * wall_len)

                if 0 <= proj <= 1:
                    # Check distance from test_p to wall line
                    closest = (wall_p1[0] + proj * wall_dx, wall_p1[1] + proj * wall_dy)
                    dist = math.sqrt((test_p[0] - closest[0])**2 + (test_p[1] - closest[1])**2)
                    if dist < 2:  # Within 2 pixels of wall
                        # V8 FIX: Use NEAREST wall (smallest t), not furthest
                        if best_t1 == 0 or t < best_t1:
                            best_t1 = t
                            new_door['x1'] = int(round(closest[0]))
                            new_door['y1'] = int(round(closest[1]))
                            extended = True
                        break

        # Extend endpoint 2 (forwards along direction)
        best_t2 = 0
        p2 = (door['x2'], door['y2'])
        for wall in walls:
            wall_p1 = (wall['x1'], wall['y1'])
            wall_p2 = (wall['x2'], wall['y2'])

            for t in range(1, max_extend + 1):
                test_p = (p2[0] + t * ux, p2[1] + t * uy)
                wall_dx = wall_p2[0] - wall_p1[0]
                wall_dy = wall_p2[1] - wall_p1[1]
                wall_len = math.sqrt(wall_dx*wall_dx + wall_dy*wall_dy)
                if wall_len < 0.001:
                    continue

                to_test = (test_p[0] - wall_p1[0], test_p[1] - wall_p1[1])
                proj = (to_test[0] * wall_dx + to_test[1] * wall_dy) / (wall_len * wall_len)

                if 0 <= proj <= 1:
                    closest = (wall_p1[0] + proj * wall_dx, wall_p1[1] + proj * wall_dy)
                    dist = math.sqrt((test_p[0] - closest[0])**2 + (test_p[1] - closest[1])**2)
                    if dist < 2:
                        # V8 FIX: Use NEAREST wall (smallest t), not furthest
                        if best_t2 == 0 or t < best_t2:
                            best_t2 = t
                            new_door['x2'] = int(round(closest[0]))
                            new_door['y2'] = int(round(closest[1]))
                            extended = True
                        break

        if extended:
            doors_extended += 1
        extended_doors.append(new_door)

    print(f"  Extended {doors_extended} doors to reach walls")
    return walls + extended_doors


def snap_doors_to_wall_endpoints(segments, snap_distance=5):
    """Snap door endpoints to nearest wall endpoints AFTER all wall processing."""
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors or not walls:
        return segments

    print(f"  Re-snapping {len(doors)} door endpoints to final wall positions")

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


def remove_disconnected_doors(segments, max_distance=10):
    """
    Remove doors that don't connect to walls on both ends.
    This handles cases where partial door masks create invalid door segments.
    """
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    if not doors or not walls:
        return segments

    # Build KDTree of wall endpoints
    wall_endpoints = []
    for wall in walls:
        wall_endpoints.append([wall['x1'], wall['y1']])
        wall_endpoints.append([wall['x2'], wall['y2']])

    wall_endpoints = np.array(wall_endpoints)
    tree = KDTree(wall_endpoints)

    valid_doors = []
    removed_count = 0

    for door in doors:
        p1 = np.array([door['x1'], door['y1']])
        p2 = np.array([door['x2'], door['y2']])

        dist1, _ = tree.query(p1)
        dist2, _ = tree.query(p2)

        # Door must connect to walls on BOTH ends
        if dist1 <= max_distance and dist2 <= max_distance:
            valid_doors.append(door)
        else:
            removed_count += 1

    if removed_count > 0:
        print(f"  Removed {removed_count} disconnected doors (not touching walls on both ends)")

    return walls + valid_doors


def filter_short_segments(segments, min_length):
    """Remove very short segments."""
    filtered = []
    removed = 0

    for seg in segments:
        length = segment_length(seg)
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


def export_segments_json(segments, output_path, image_size):
    """Export segments to JSON for UVTT conversion.

    Args:
        segments: List of segment dicts with x1, y1, x2, y2, type
        output_path: Path to save JSON file
        image_size: Tuple of (height, width) from mask.shape
    """
    data = {
        'segments': segments,
        'image_width': image_size[1],
        'image_height': image_size[0]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nJSON export saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--output", type=str, default="wall_visualization_v8.png")
    parser.add_argument("--grid-size", type=int, default=70,
                        help="Scale factor for auto-calculating parameters (default: 70)")
    parser.add_argument("--epsilon", type=int, default=None,
                        help="Override Douglas-Peucker simplification tolerance (default: grid-size * 0.05)")
    parser.add_argument("--no-straighten", action="store_true", help="Disable wall straightening")
    parser.add_argument("--json-output", type=str, default=None,
                        help="Export segment data to JSON file for UVTT conversion")

    args = parser.parse_args()
    params = calculate_params_from_scale(args.grid_size, epsilon_override=args.epsilon)

    print(f"Loading mask: {args.mask}")
    mask = np.array(Image.open(args.mask).convert('L'))

    print(f"Mask shape: {mask.shape}")
    print(f"Unique values: {np.unique(mask)}")

    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Step 1: Extract walls (V4: filters out corner-rounding segments)
    print("\nStep 1: Extracting walls (filtering corner-rounding segments)...")
    walls = extract_walls_from_wall_mask(mask, params)
    print(f"  Extracted {len(walls)} wall segments")

    # Step 2: Extract doors
    print("\nStep 2: Extracting doors from door mask...")
    doors = extract_doors(mask, params)
    print(f"  Extracted {len(doors)} door segments")

    # Step 2b: V12 - Add perimeter walls to close off edge-touching rooms
    print("\nStep 2b: Adding perimeter walls...")
    walls = add_perimeter_walls(walls, mask.shape)
    # Note: perimeter walls are added to walls list only, doors unchanged

    # Step 3: Combine
    all_segments = walls + doors
    print(f"\nStep 3: Combined {len(all_segments)} segments")

    # Step 4: Straighten walls (before corner extension)
    if not args.no_straighten:
        print("\nStep 4: Straightening near-axis walls...")
        all_segments = straighten_walls(all_segments, params)
    else:
        print("\nStep 4: Skipping wall straightening (--no-straighten)")

    # Step 5: Smart corner extension (now works better without corner-rounding segments)
    print("\nStep 5: Smart corner extension (line intersection)...")
    all_segments = smart_corner_extension(all_segments, params)

    # Step 6: Extend doors to reach walls
    print("\nStep 6: Extending doors to intersect walls...")
    all_segments = extend_doors_to_walls(all_segments, params)

    # Step 7: V6 - Filter invalid doors BEFORE splitting (prevents orphan splits)
    print("\nStep 7: Filtering doors that don't cross walls...")
    all_segments = filter_doors_crossing_walls(all_segments, max_distance=8)

    # Step 8: Split walls at door intersections
    print("\nStep 8: Splitting walls at door intersections...")
    all_segments = split_walls_at_doors(all_segments, params)

    # Step 9: Re-snap doors to final wall positions
    print("\nStep 9: Snapping doors to final wall positions...")
    all_segments = snap_doors_to_wall_endpoints(all_segments, snap_distance=8)

    # Step 10: Final cleanup
    print("\nStep 10: Final cleanup...")
    all_segments = filter_zero_length_segments(all_segments)
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    # Step 11: V10 - Merge dangling endpoints
    print("\nStep 11: Merging dangling endpoints...")
    all_segments = merge_dangling_endpoints(all_segments, merge_distance=25)

    print(f"\nFinal segment count:")
    print(f"  Walls: {sum(1 for s in all_segments if s['type'] == 'wall')}")
    print(f"  Doors: {sum(1 for s in all_segments if s['type'] == 'door')}")
    print(f"  Total: {len(all_segments)}")

    visualize_walls(mask, all_segments, args.output)

    # Export JSON if requested
    if args.json_output:
        export_segments_json(all_segments, args.json_output, mask.shape)


if __name__ == "__main__":
    main()
