"""
Walls to Universal VTT Converter

Converts mask_to_walls (old) JSON output to Universal VTT format (.uvtt/.dd2vtt).

Usage:
    # From JSON (two-step pipeline):
    python walls_to_uvtt.py --json walls.json --image map.png --grid-size 70 --output map.uvtt

    # Direct from mask (single command):
    python walls_to_uvtt.py --mask input_mask.png --image map.png --grid-size 70 --output map.uvtt
"""

import json
import base64
import math
import argparse
from pathlib import Path
from collections import defaultdict


def load_segments_json(path):
    """Load segment data from mask_to_walls (old) JSON output."""
    with open(path, 'r') as f:
        return json.load(f)


def pixels_to_grid(x, y, grid_size):
    """Convert pixel coordinates to grid units."""
    return x / grid_size, y / grid_size


def segment_to_grid(seg, grid_size):
    """Convert a segment's coordinates to grid units."""
    x1, y1 = pixels_to_grid(seg['x1'], seg['y1'], grid_size)
    x2, y2 = pixels_to_grid(seg['x2'], seg['y2'], grid_size)
    return {
        'x1': x1, 'y1': y1,
        'x2': x2, 'y2': y2,
        'type': seg['type']
    }


def points_equal(p1, p2, tolerance=0.001):
    """Check if two points are equal within tolerance."""
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance


def build_line_of_sight(walls, grid_size):
    """
    Group connected wall segments into polylines for line_of_sight.

    Returns list of polylines, where each polyline is a list of {x, y} points.
    """
    if not walls:
        return []

    # Convert walls to grid coordinates
    grid_walls = [segment_to_grid(w, grid_size) for w in walls]

    # Build adjacency map: endpoint -> list of (segment_index, endpoint_type)
    # endpoint_type: 1 = start point, 2 = end point
    endpoint_map = defaultdict(list)

    for i, wall in enumerate(grid_walls):
        p1 = (round(wall['x1'], 4), round(wall['y1'], 4))
        p2 = (round(wall['x2'], 4), round(wall['y2'], 4))
        endpoint_map[p1].append((i, 1))
        endpoint_map[p2].append((i, 2))

    # Find segments that can be grouped into polylines
    used = set()
    polylines = []

    for start_idx in range(len(grid_walls)):
        if start_idx in used:
            continue

        # Start a new polyline
        wall = grid_walls[start_idx]
        used.add(start_idx)

        # Build polyline by following connections
        points = [(wall['x1'], wall['y1']), (wall['x2'], wall['y2'])]

        # Try to extend from the end
        while True:
            end_point = (round(points[-1][0], 4), round(points[-1][1], 4))
            connections = endpoint_map.get(end_point, [])

            found_next = False
            for seg_idx, ep_type in connections:
                if seg_idx in used:
                    continue

                next_wall = grid_walls[seg_idx]
                used.add(seg_idx)

                # Add the other endpoint of this segment
                if ep_type == 1:
                    # We matched the start, so add the end
                    points.append((next_wall['x2'], next_wall['y2']))
                else:
                    # We matched the end, so add the start
                    points.append((next_wall['x1'], next_wall['y1']))

                found_next = True
                break

            if not found_next:
                break

        # Try to extend from the start (in reverse)
        while True:
            start_point = (round(points[0][0], 4), round(points[0][1], 4))
            connections = endpoint_map.get(start_point, [])

            found_prev = False
            for seg_idx, ep_type in connections:
                if seg_idx in used:
                    continue

                prev_wall = grid_walls[seg_idx]
                used.add(seg_idx)

                # Add the other endpoint at the beginning
                if ep_type == 1:
                    points.insert(0, (prev_wall['x2'], prev_wall['y2']))
                else:
                    points.insert(0, (prev_wall['x1'], prev_wall['y1']))

                found_prev = True
                break

            if not found_prev:
                break

        # Convert to UVTT format
        polyline = [{"x": p[0], "y": p[1]} for p in points]
        polylines.append(polyline)

    return polylines


def build_portals(doors, grid_size):
    """
    Convert door segments to UVTT portal format.

    Each portal has:
    - position: midpoint of the door
    - bounds: the two endpoints
    - rotation: angle of the door line
    - closed: true (doors start closed)
    - freestanding: false
    """
    portals = []

    for door in doors:
        x1, y1 = pixels_to_grid(door['x1'], door['y1'], grid_size)
        x2, y2 = pixels_to_grid(door['x2'], door['y2'], grid_size)

        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Calculate rotation in radians (per UVTT spec)
        dx = x2 - x1
        dy = y2 - y1
        rotation = math.atan2(dy, dx)

        portal = {
            "position": {"x": mid_x, "y": mid_y},
            "bounds": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y2}
            ],
            "rotation": rotation,
            "closed": True,
            "freestanding": False
        }
        portals.append(portal)

    return portals


def encode_image_base64(image_path):
    """Read an image file and return base64 encoded string."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_uvtt(segments_data, map_image_path, grid_size, output_path):
    """
    Create a Universal VTT file from segment data and map image.

    Args:
        segments_data: Dict with 'segments', 'image_width', 'image_height'
        map_image_path: Path to the map image (PNG)
        grid_size: Pixels per grid square
        output_path: Path to save the .uvtt file
    """
    segments = segments_data['segments']
    image_width = segments_data['image_width']
    image_height = segments_data['image_height']

    # Separate walls and doors
    walls = [s for s in segments if s['type'] == 'wall']
    doors = [s for s in segments if s['type'] == 'door']

    print(f"Processing {len(walls)} walls and {len(doors)} doors...")

    # Calculate map size in grid units
    map_width = image_width / grid_size
    map_height = image_height / grid_size

    # Build line of sight (walls as polylines)
    line_of_sight = build_line_of_sight(walls, grid_size)
    print(f"Created {len(line_of_sight)} line-of-sight polylines")

    # Build portals (doors)
    portals = build_portals(doors, grid_size)
    print(f"Created {len(portals)} portals")

    # Encode map image
    print(f"Encoding map image: {map_image_path}")
    image_data = encode_image_base64(map_image_path)

    # Build UVTT structure
    uvtt = {
        "format": 0.3,
        "resolution": {
            "map_origin": {"x": 0, "y": 0},
            "map_size": {"x": map_width, "y": map_height},
            "pixels_per_grid": grid_size
        },
        "line_of_sight": line_of_sight,
        "portals": portals,
        "lights": [],
        "image": image_data
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(uvtt, f, indent=2)

    print(f"\nUniversal VTT file saved to: {output_path}")
    print(f"  Map size: {map_width:.1f} x {map_height:.1f} grid squares")
    print(f"  Line of sight polylines: {len(line_of_sight)}")
    print(f"  Portals (doors): {len(portals)}")


def run_mask_to_walls(mask_path, grid_size, epsilon=None):
    """
    Run mask_to_walls (old) extraction internally and return segment data.

    This allows single-command conversion from mask to UVTT.
    """
    # Import mask_to_walls (old) functions (v13: L-shaped door merge fix)
    from mask_to_walls_v13 import (
        calculate_params_from_scale,
        extract_walls_from_wall_mask,
        extract_doors,
        add_perimeter_walls,
        straighten_walls,
        smart_corner_extension,
        extend_doors_to_walls,
        filter_doors_crossing_walls,
        split_walls_at_doors,
        snap_doors_to_wall_endpoints,
        filter_zero_length_segments,
        filter_short_segments,
        merge_dangling_endpoints
    )
    import numpy as np
    from PIL import Image

    params = calculate_params_from_scale(grid_size, epsilon_override=epsilon)

    print(f"Loading mask: {mask_path}")
    mask = np.array(Image.open(mask_path).convert('L'))

    print(f"Mask shape: {mask.shape}")

    # Run the full pipeline (v12)
    print("\nExtracting walls...")
    walls = extract_walls_from_wall_mask(mask, params)

    print("\nExtracting doors...")
    doors = extract_doors(mask, params)

    # V12: Add perimeter walls to close off edge-touching rooms
    print("\nAdding perimeter walls...")
    walls = add_perimeter_walls(walls, mask.shape)

    all_segments = walls + doors

    print("\nProcessing segments...")
    all_segments = straighten_walls(all_segments, params)
    all_segments = smart_corner_extension(all_segments, params)
    all_segments = extend_doors_to_walls(all_segments, params)
    all_segments = filter_doors_crossing_walls(all_segments, max_distance=8)
    all_segments = split_walls_at_doors(all_segments, params)
    all_segments = snap_doors_to_wall_endpoints(all_segments, snap_distance=8)
    all_segments = filter_zero_length_segments(all_segments)
    all_segments = filter_short_segments(all_segments, params['min_wall_length'])

    # V12: Merge dangling wall endpoints
    print("\nMerging dangling endpoints...")
    all_segments = merge_dangling_endpoints(all_segments, merge_distance=25)

    return {
        'segments': all_segments,
        'image_width': mask.shape[1],
        'image_height': mask.shape[0]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert mask_to_walls (old) output to Universal VTT format"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--json", type=str,
                             help="JSON file from mask_to_walls (old) --json-output")
    input_group.add_argument("--mask", type=str,
                             help="Mask image (runs mask_to_walls (old) internally)")

    # Required arguments
    parser.add_argument("--image", type=str, required=True,
                        help="Map image file to embed in UVTT")
    parser.add_argument("--grid-size", type=int, required=True,
                        help="Pixels per grid square")
    parser.add_argument("--output", type=str, required=True,
                        help="Output UVTT file path")

    # Optional arguments
    parser.add_argument("--epsilon", type=int, default=None,
                        help="Douglas-Peucker epsilon (only with --mask)")

    args = parser.parse_args()

    # Get segment data
    if args.json:
        print(f"Loading segments from JSON: {args.json}")
        segments_data = load_segments_json(args.json)
    else:
        print(f"Running mask_to_walls (old) on: {args.mask}")
        segments_data = run_mask_to_walls(args.mask, args.grid_size, args.epsilon)

    # Create UVTT file
    create_uvtt(segments_data, args.image, args.grid_size, args.output)


if __name__ == "__main__":
    main()
