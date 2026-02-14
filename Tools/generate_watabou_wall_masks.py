"""
Generate wall line masks from Watabou dungeon exports.

Two methods:
- edge: Walls placed at room edges (where floor meets wall), no offset
- recessed: Walls offset 0.25 grid units outward into wall area
            (toward center of visual wall between rooms)

Class values (from MASK_TRAINING_STRATEGY.md):
- 0 = Background (unmarked pixels)
- 50 = Wall (room boundaries, secret doors, barred passages)
- 150 = Door (types 1, 5, 7 only)

Door type handling (see Text_Files/watabou_door_types.txt):
- Types 1, 5, 7: Normal doors -> Door class at center of tile
- Type 6: Secret door -> Wall class at center of tile + half-wall line
- Types 4, 8: Barred/portcullis -> Wall class at center of tile
- Types 0, 2, 3, 9: Open passages/stairs -> no line

Based on coordinate transform from: https://github.com/TarkanAl-Kazily/one-page-parser
"""

import json
import math
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


GRID_SIZE = 70  # Watabou's default export grid size
CENTER_WALL_OFFSET = 0.25  # Grid units outward for recessed method

# Class values
CLASS_BG = 0
CLASS_WALL = 50
CLASS_DOOR = 150

# Door type classification (see Text_Files/watabou_door_types.txt)
DOOR_RENDER_AS_DOOR = {1, 4, 5, 7, 8}  # Normal doors + barred/portcullis -> Door class
DOOR_RENDER_AS_WALL = {6}              # Secret doors -> Wall class
# Types 0, 2, 3, 9 = open passages / open frames / stairs -> no line


def generate_wall_mask(json_path, png_path, method='edge', circular=True,
                       wall_thickness=None):
    """
    Generate wall line mask from Watabou JSON data.

    Args:
        json_path: Path to Watabou JSON
        png_path: Path to Watabou PNG (for dimensions and coordinate alignment)
        method: 'edge' or 'recessed'
        circular: Whether to render circular rooms (rotunda)
        wall_thickness: Line thickness in pixels (default: GRID_SIZE * 0.05)

    Returns:
        mask: single-channel numpy array with class values
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rects = data.get('rects', [])
    doors = data.get('doors', [])

    img = Image.open(png_path)

    if wall_thickness is None:
        wall_thickness = max(2, int(GRID_SIZE * 0.05))

    # Edge: walls at room boundary (offset=0)
    # Recessed: walls pushed outward into wall area (offset=0.25 toward void)
    wall_offset = CENTER_WALL_OFFSET if method == 'recessed' else 0.0

    # --- Coordinate transform (same as floor mask script) ---
    min_x = min(r['x'] for r in rects)
    min_y = min(r['y'] for r in rects)

    # Edge offset calculation (Foundry module logic)
    x_edge_has_tile = any(d['x'] == min_x for d in doors)
    y_edge_has_tile = any(d['y'] == min_y for d in doors)

    x_px_offset = -0.25 * GRID_SIZE if x_edge_has_tile else 0.75 * GRID_SIZE
    y_px_offset = -0.25 * GRID_SIZE if y_edge_has_tile else 0.75 * GRID_SIZE

    # Empirical correction for PNG alignment
    x_px_offset += 18
    y_px_offset += 18

    # --- Build occupancy grid ---
    occupied = set()
    rotunda_cells = set()
    rotunda_rects = []

    for rect in rects:
        is_rotunda = circular and rect.get('rotunda', False)
        if is_rotunda:
            rotunda_rects.append(rect)
        for x in range(rect['x'], rect['x'] + rect['w']):
            for y in range(rect['y'], rect['y'] + rect['h']):
                occupied.add((x, y))
                if is_rotunda:
                    rotunda_cells.add((x, y))

    # --- Build rotunda cell-to-rect mapping (for ellipse intersection) ---
    rotunda_cell_to_rect = {}
    for rect in rotunda_rects:
        for x in range(rect['x'], rect['x'] + rect['w']):
            for y in range(rect['y'], rect['y'] + rect['h']):
                rotunda_cell_to_rect[(x, y)] = rect

    # --- Helper functions ---
    def grid_to_pixel(gx, gy):
        """Convert grid coordinate (float) to pixel coordinate."""
        px = gx * GRID_SIZE - min_x * GRID_SIZE + x_px_offset
        py = gy * GRID_SIZE - min_y * GRID_SIZE + y_px_offset
        return int(round(px)), int(round(py))

    def clamp_point(p):
        """Clamp a pixel point to image bounds."""
        return (max(0, min(img.width - 1, p[0])),
                max(0, min(img.height - 1, p[1])))

    def draw_line(p1, p2, color, thickness):
        """Draw a line on the mask, clamped to image bounds."""
        p1 = clamp_point(p1)
        p2 = clamp_point(p2)
        cv2.line(mask, p1, p2, int(color), thickness)

    def ellipse_x_at_y(rect, y_grid):
        """Get (x_left, x_right) where rotunda ellipse intersects y_grid."""
        cx = rect['x'] + rect['w'] / 2.0
        cy = rect['y'] + rect['h'] / 2.0
        rx = rect['w'] / 2.0 + (CENTER_WALL_OFFSET if method == 'recessed' else 0)
        ry = rect['h'] / 2.0 + (CENTER_WALL_OFFSET if method == 'recessed' else 0)
        y_norm = (y_grid - cy) / ry
        if abs(y_norm) >= 1:
            return None
        dx = rx * math.sqrt(1 - y_norm ** 2)
        return (cx - dx, cx + dx)

    def ellipse_y_at_x(rect, x_grid):
        """Get (y_top, y_bottom) where rotunda ellipse intersects x_grid."""
        cx = rect['x'] + rect['w'] / 2.0
        cy = rect['y'] + rect['h'] / 2.0
        rx = rect['w'] / 2.0 + (CENTER_WALL_OFFSET if method == 'recessed' else 0)
        ry = rect['h'] / 2.0 + (CENTER_WALL_OFFSET if method == 'recessed' else 0)
        x_norm = (x_grid - cx) / rx
        if abs(x_norm) >= 1:
            return None
        dy = ry * math.sqrt(1 - x_norm ** 2)
        return (cy - dy, cy + dy)

    # --- Create mask ---
    mask = np.zeros((img.height, img.width), dtype=np.uint8)

    # --- Build secret door cell lookup (for Pass 1 wall shortening) ---
    secret_door_info = {}  # (cx, cy) -> (ddx, ddy)
    for door in doors:
        if door.get('type', 0) == 6:
            secret_door_info[(door['x'], door['y'])] = (
                door['dir']['x'], door['dir']['y']
            )

    # --- Build exit edge set (type 3 stairs = dungeon exits) ---
    # dir points toward room interior; exit is opposite dir
    exit_edges = set()  # (cx, cy, dx, dy) = cell + void direction
    for door in doors:
        if door.get('type', 0) == 3:
            sx, sy = door['x'], door['y']
            ddx, ddy = door['dir']['x'], door['dir']['y']
            exit_edges.add((sx, sy, -ddx, -ddy))

    # =============================================
    # PASS 1: Wall lines on void-facing edges
    # =============================================
    # Only draw walls where occupied cells border void (unoccupied) space.
    # Interior edges between occupied cells get NO wall lines here.
    # Doors/secret doors/barred passages are handled in Pass 2.
    #
    # For secret door tiles (edge method only): passage walls are shortened
    # to half-length, removing the half toward the thick side (opposite dir).

    for (cx, cy) in occupied:
        if (cx, cy) in rotunda_cells:
            continue

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy

            # Skip if neighbor is occupied (interior edge) or rotunda
            if (nx, ny) in occupied or (nx, ny) in rotunda_cells:
                continue

            # Exit wall handling (type 3 stairs = dungeon exits)
            if (cx, cy, dx, dy) in exit_edges:
                if method == 'edge':
                    continue  # Remove exit wall entirely
                else:
                    # Recessed: draw only stubs at tile edges that connect
                    # to the perpendicular recessed passage walls
                    if dx != 0:  # Vertical exit wall
                        edge_x = cx + (1 if dx > 0 else 0)
                        edge_x_f = edge_x + (dx * wall_offset)
                        # Top stub: tile edge outward to recessed passage wall
                        p1 = grid_to_pixel(edge_x_f, cy)
                        p2 = grid_to_pixel(edge_x_f, cy - wall_offset)
                        draw_line(p1, p2, CLASS_WALL, wall_thickness)
                        # Bottom stub: tile edge outward to recessed passage wall
                        p1 = grid_to_pixel(edge_x_f, cy + 1)
                        p2 = grid_to_pixel(edge_x_f, cy + 1 + wall_offset)
                        draw_line(p1, p2, CLASS_WALL, wall_thickness)
                    else:  # Horizontal exit wall
                        edge_y = cy + (1 if dy > 0 else 0)
                        edge_y_f = edge_y + (dy * wall_offset)
                        # Left stub: tile edge outward to recessed passage wall
                        p1 = grid_to_pixel(cx, edge_y_f)
                        p2 = grid_to_pixel(cx - wall_offset, edge_y_f)
                        draw_line(p1, p2, CLASS_WALL, wall_thickness)
                        # Right stub: tile edge outward to recessed passage wall
                        p1 = grid_to_pixel(cx + 1, edge_y_f)
                        p2 = grid_to_pixel(cx + 1 + wall_offset, edge_y_f)
                        draw_line(p1, p2, CLASS_WALL, wall_thickness)
                    continue

            # Void edge: draw wall line
            # Offset direction: +dx/+dy pushes OUTWARD toward void (for center method)
            if dx != 0:  # Vertical edge (left or right)
                edge_x = cx + (1 if dx > 0 else 0)
                edge_x_f = edge_x + (dx * wall_offset)
                y_start = cy
                y_end = cy + 1

                # Shorten passage walls for secret doors (edge method only)
                if method == 'edge' and (cx, cy) in secret_door_info:
                    sd_ddx, sd_ddy = secret_door_info[(cx, cy)]
                    # Only shorten if this is a passage wall (perpendicular to
                    # passage direction). Passage runs in sd_ddx/sd_ddy direction,
                    # so passage walls are the edges where dx/dy is perpendicular.
                    if sd_ddy != 0:  # Vertical passage -> left/right are passage walls
                        # Thick side is opposite dir: y = cy if ddy>0, y = cy+1 if ddy<0
                        if sd_ddy > 0:
                            # Thick side at y=cy (top), remove top half
                            y_start = cy + 0.5
                        else:
                            # Thick side at y=cy+1 (bottom), remove bottom half
                            y_end = cy + 0.5

                # Corner adjustment for recessed method:
                # At each endpoint, check if wall continues straight, or if
                # it's an outside corner (extend) or inside corner (trim).
                # Skip if endpoint is adjacent to a rotunda (handled below).
                if method == 'recessed':
                    # Top endpoint
                    if (cx, cy - 1) not in rotunda_cell_to_rect:
                        top_occ = (cx, cy - 1) in occupied
                        top_diag = (cx + dx, cy - 1) in occupied
                        top_continues = top_occ and not top_diag
                        if not top_continues:
                            if top_diag:
                                y_start += wall_offset  # Inside corner: trim
                            else:
                                y_start -= wall_offset  # Outside corner: extend
                    # Bottom endpoint
                    if (cx, cy + 1) not in rotunda_cell_to_rect:
                        bot_occ = (cx, cy + 1) in occupied
                        bot_diag = (cx + dx, cy + 1) in occupied
                        bot_continues = bot_occ and not bot_diag
                        if not bot_continues:
                            if bot_diag:
                                y_end -= wall_offset  # Inside corner: trim
                            else:
                                y_end += wall_offset  # Outside corner: extend

                # Set wall endpoint to rotunda ellipse intersection
                if (cx, cy - 1) in rotunda_cell_to_rect:
                    r = rotunda_cell_to_rect[(cx, cy - 1)]
                    result = ellipse_y_at_x(r, edge_x_f)
                    if result:
                        y_start = result[1]  # Ellipse bottom at this x
                if (cx, cy + 1) in rotunda_cell_to_rect:
                    r = rotunda_cell_to_rect[(cx, cy + 1)]
                    result = ellipse_y_at_x(r, edge_x_f)
                    if result:
                        y_end = result[0]  # Ellipse top at this x

                p1 = grid_to_pixel(edge_x_f, y_start)
                p2 = grid_to_pixel(edge_x_f, y_end)

            else:  # Horizontal edge (top or bottom)
                edge_y = cy + (1 if dy > 0 else 0)
                edge_y_f = edge_y + (dy * wall_offset)
                x_start = cx
                x_end = cx + 1

                # Shorten passage walls for secret doors (edge method only)
                if method == 'edge' and (cx, cy) in secret_door_info:
                    sd_ddx, sd_ddy = secret_door_info[(cx, cy)]
                    # Only shorten if this is a passage wall (perpendicular to
                    # passage direction). Passage runs in sd_ddx/sd_ddy direction,
                    # so passage walls are the edges where dx/dy is perpendicular.
                    if sd_ddx != 0:  # Horizontal passage -> top/bottom are passage walls
                        # Thick side is opposite dir: x = cx if ddx>0, x = cx+1 if ddx<0
                        if sd_ddx > 0:
                            # Thick side at x=cx (left), remove left half
                            x_start = cx + 0.5
                        else:
                            # Thick side at x=cx+1 (right), remove right half
                            x_end = cx + 0.5

                # Corner adjustment for recessed method
                # Skip if endpoint is adjacent to a rotunda (handled below).
                if method == 'recessed':
                    # Left endpoint
                    if (cx - 1, cy) not in rotunda_cell_to_rect:
                        left_occ = (cx - 1, cy) in occupied
                        left_diag = (cx - 1, cy + dy) in occupied
                        left_continues = left_occ and not left_diag
                        if not left_continues:
                            if left_diag:
                                x_start += wall_offset  # Inside corner: trim
                            else:
                                x_start -= wall_offset  # Outside corner: extend
                    # Right endpoint
                    if (cx + 1, cy) not in rotunda_cell_to_rect:
                        right_occ = (cx + 1, cy) in occupied
                        right_diag = (cx + 1, cy + dy) in occupied
                        right_continues = right_occ and not right_diag
                        if not right_continues:
                            if right_diag:
                                x_end -= wall_offset  # Inside corner: trim
                            else:
                                x_end += wall_offset  # Outside corner: extend

                # Set wall endpoint to rotunda ellipse intersection
                if (cx - 1, cy) in rotunda_cell_to_rect:
                    r = rotunda_cell_to_rect[(cx - 1, cy)]
                    result = ellipse_x_at_y(r, edge_y_f)
                    if result:
                        x_start = result[1]  # Ellipse right at this y
                if (cx + 1, cy) in rotunda_cell_to_rect:
                    r = rotunda_cell_to_rect[(cx + 1, cy)]
                    result = ellipse_x_at_y(r, edge_y_f)
                    if result:
                        x_end = result[0]  # Ellipse left at this y

                p1 = grid_to_pixel(x_start, edge_y_f)
                p2 = grid_to_pixel(x_end, edge_y_f)

            draw_line(p1, p2, CLASS_WALL, wall_thickness)

    # =============================================
    # PASS 2: Door/secret door/barred lines at tile centers
    # =============================================
    # Doors are drawn at the CENTER of their 1x1 tile,
    # perpendicular to the door direction (crossing the passage).
    #
    # - Types 1, 5, 7: Door class (150)
    # - Type 6 (secret): Wall class (50) - visually a half-wall
    # - Types 4, 8 (barred): Wall class (50)
    # - Types 0, 2, 3, 9: No line (open passage/stairs)

    for door in doors:
        door_type = door.get('type', 0)

        if door_type in DOOR_RENDER_AS_DOOR:
            color = CLASS_DOOR
        elif door_type in DOOR_RENDER_AS_WALL:
            color = CLASS_WALL
        else:
            continue  # Open passage, stairs, etc. - no line

        cx, cy = door['x'], door['y']
        ddx, ddy = door['dir']['x'], door['dir']['y']

        # Shift: moves line along dir axis (multiplied by ddx or ddy)
        # - Secret doors in recessed mode: -0.25 toward wall end
        # - Types 7, 8: PNG door graphic is offset from tile center
        #   (measured via detect_door_offset.py across 105 maps)
        # Center offset: constant shift along the line axis (independent of dir sign)
        if door_type == 6 and method == 'recessed':
            shift = -0.25  # toward opposite-of-dir (wall end)
            center_offset = 0.0
        elif door_type == 7:
            shift = -0.25  # ±0.25 grid opposite to dir
            center_offset = 0.0
        elif door_type == 8:
            shift = -0.20  # ±0.20 grid opposite to dir
            center_offset = 0.06  # +0.06 grid constant
        else:
            shift = 0.0
            center_offset = 0.0

        # Door line is perpendicular to direction, at center of tile
        if ddx != 0:
            # Passage runs horizontally -> door line is vertical
            line_x = cx + 0.5 + center_offset + (ddx * shift)
            y_start = cy
            y_end = cy + 1
            # Extend to meet recessed passage walls
            if method == 'recessed':
                if (cx, cy - 1) not in occupied:
                    y_start -= wall_offset
                if (cx, cy + 1) not in occupied:
                    y_end += wall_offset
            p1 = grid_to_pixel(line_x, y_start)
            p2 = grid_to_pixel(line_x, y_end)
        else:
            # Passage runs vertically -> door line is horizontal
            line_y = cy + 0.5 + center_offset + (ddy * shift)
            x_start = cx
            x_end = cx + 1
            # Extend to meet recessed passage walls
            if method == 'recessed':
                if (cx - 1, cy) not in occupied:
                    x_start -= wall_offset
                if (cx + 1, cy) not in occupied:
                    x_end += wall_offset
            p1 = grid_to_pixel(x_start, line_y)
            p2 = grid_to_pixel(x_end, line_y)

        draw_line(p1, p2, color, wall_thickness)

    # =============================================
    # PASS 2b: Secret door half-wall lines (edge method only)
    # =============================================
    # Secret doors (type 6) visually show a half-wall: one side of the
    # passage has a wall section flush with the room wall, extending from
    # the passage wall to the center line. This creates an L-shape.
    #
    # We draw an additional wall line along the room-facing (interior) edge,
    # from one passage wall (void side) to the center of the tile.
    # Only for edge method - recessed method shifts the center line instead.

    for door in doors:
        if door.get('type', 0) != 6:
            continue
        if method == 'recessed':
            continue  # Recessed mode shifts center line instead

        # cx, cy = grid coordinates of the 1x1 secret door tile
        # The tile occupies the square from (cx, cy) to (cx+1, cy+1)
        cx, cy = door['x'], door['y']

        # ddx, ddy = direction the door faces (points toward connecting room)
        # e.g. ddx=1,ddy=0 means door faces RIGHT, so passage runs left-right
        # e.g. ddx=0,ddy=1 means door faces DOWN, so passage runs up-down
        ddx, ddy = door['dir']['x'], door['dir']['y']

        # The passage runs in the direction of (ddx, ddy).
        # The passage WALLS (void-facing edges) are perpendicular to this.
        # We need to pick which of the two perpendicular sides has the
        # half-wall. We try CCW 90° rotation of dir first, then CW.
        #
        # CCW 90° of (dx, dy) = (-dy, dx)
        # Example: dir=(0,1) DOWN -> CCW -> (-1, 0) LEFT
        # Example: dir=(1,0) RIGHT -> CCW -> (0, 1) DOWN
        preferred = (-ddy, ddx)
        alternate = (ddy, -ddx)

        # Check which perpendicular neighbor is void (passage wall side).
        # The half-wall is flush with a passage wall, so that side must be void.
        if (cx + preferred[0], cy + preferred[1]) not in occupied:
            half_wall_perp = preferred
        elif (cx + alternate[0], cy + alternate[1]) not in occupied:
            half_wall_perp = alternate
        else:
            half_wall_perp = None  # Both sides occupied, can't place half-wall

        if half_wall_perp is None:
            continue

        # half_wall_perp = direction from door tile toward the void side
        # where the half-wall is. NOT CURRENTLY USED in drawing below,
        # but determines which side was chosen.
        pdx, pdy = half_wall_perp

        # NOW DRAW THE HALF-WALL LINE.
        # The half-wall runs along the interior edge that faces the room
        # in the dir direction. It spans the full cell.
        #
        # For a HORIZONTAL passage (ddx != 0):
        #   The interior edge is VERTICAL, at x = cx (if dir points left)
        #   or x = cx+1 (if dir points right).
        #   The line runs from y=cy to y=cy+1 (full cell height).
        #
        # For a VERTICAL passage (ddy != 0):
        #   The interior edge is HORIZONTAL, at y = cy (if dir points up)
        #   or y = cy+1 (if dir points down).
        #   The line runs from x=cx to x=cx+1 (full cell width).
        #
        # NOTE: This line is ALWAYS on the dir-facing edge, regardless
        # of which perpendicular side the half-wall is on. The perpendicular
        # choice (half_wall_perp) is computed but doesn't affect the
        # line position at all - the line is always in the same place.

        if ddx != 0:  # Horizontal passage (dir points left or right)
            # edge_x = x-coordinate of the interior edge OPPOSITE to dir
            # If dir=(1,0) RIGHT: opposite edge is at x = cx (LEFT side)
            # If dir=(-1,0) LEFT: opposite edge is at x = cx+1 (RIGHT side)
            edge_x = cx + (0 if ddx > 0 else 1)
            p1 = grid_to_pixel(edge_x, cy)
            p2 = grid_to_pixel(edge_x, cy + 1)
        else:  # Vertical passage (dir points up or down)
            # edge_y = y-coordinate of the interior edge OPPOSITE to dir
            # If dir=(0,1) DOWN: opposite edge is at y = cy (TOP side)
            # If dir=(0,-1) UP: opposite edge is at y = cy+1 (BOTTOM side)
            edge_y = cy + (0 if ddy > 0 else 1)
            p1 = grid_to_pixel(cx, edge_y)
            p2 = grid_to_pixel(cx + 1, edge_y)

        draw_line(p1, p2, CLASS_WALL, wall_thickness)

    # =============================================
    # PASS 3: Rotunda rooms - elliptical outlines with openings
    # =============================================
    # Draw full ellipse on a temp mask, then erase where corridors/rooms
    # connect to the rotunda. Merge onto main mask with np.maximum.

    for rect in rotunda_rects:
        center_gx = rect['x'] + rect['w'] / 2.0
        center_gy = rect['y'] + rect['h'] / 2.0

        if method == 'recessed':
            rx_grid = rect['w'] / 2.0 + CENTER_WALL_OFFSET
            ry_grid = rect['h'] / 2.0 + CENTER_WALL_OFFSET
        else:
            rx_grid = rect['w'] / 2.0
            ry_grid = rect['h'] / 2.0

        center_px = grid_to_pixel(center_gx, center_gy)
        rx_px = int(round(rx_grid * GRID_SIZE))
        ry_px = int(round(ry_grid * GRID_SIZE))

        if rx_px <= 0 or ry_px <= 0:
            continue

        # Draw full ellipse on temp mask
        temp = np.zeros_like(mask)
        cv2.ellipse(temp, center_px, (rx_px, ry_px),
                    0, 0, 360, CLASS_WALL, wall_thickness)

        # Erase openings where non-rotunda occupied cells connect
        margin = 0.5  # grid units - enough to cover ellipse line width
        # For recessed, corridor walls are offset by wall_offset beyond cell
        # boundary, so erase rects must extend further in the perpendicular dir
        perp_ext = wall_offset  # 0.25 for recessed, 0 for edge
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']

        for y in range(ry, ry + rh):
            # Left side
            if (rx - 1, y) in occupied and (rx - 1, y) not in rotunda_cells:
                p1 = grid_to_pixel(rx - margin, y - perp_ext)
                p2 = grid_to_pixel(rx + margin, y + 1 + perp_ext)
                cv2.rectangle(temp, p1, p2, 0, -1)
            # Right side
            if (rx + rw, y) in occupied and (rx + rw, y) not in rotunda_cells:
                p1 = grid_to_pixel(rx + rw - margin, y - perp_ext)
                p2 = grid_to_pixel(rx + rw + margin, y + 1 + perp_ext)
                cv2.rectangle(temp, p1, p2, 0, -1)

        for x in range(rx, rx + rw):
            # Top side
            if (x, ry - 1) in occupied and (x, ry - 1) not in rotunda_cells:
                p1 = grid_to_pixel(x - perp_ext, ry - margin)
                p2 = grid_to_pixel(x + 1 + perp_ext, ry + margin)
                cv2.rectangle(temp, p1, p2, 0, -1)
            # Bottom side
            if (x, ry + rh) in occupied and (x, ry + rh) not in rotunda_cells:
                p1 = grid_to_pixel(x - perp_ext, ry + rh - margin)
                p2 = grid_to_pixel(x + 1 + perp_ext, ry + rh + margin)
                cv2.rectangle(temp, p1, p2, 0, -1)

        # Merge onto main mask (keeps higher class values like doors)
        mask = np.maximum(mask, temp)

    return mask


def batch_process(json_dir, image_dir, output_dir, method='edge',
                  circular=True, debug=False, wall_thickness=None):
    """Process all Watabou exports in a directory."""
    json_path = Path(json_dir)
    image_path = Path(image_dir)
    output_path = Path(output_dir)

    json_files = sorted(json_path.glob("*.json"))

    print(f"Processing {len(json_files)} dungeons")
    print(f"  Method: {method}")
    print(f"  Grid size: {GRID_SIZE}px")
    print(f"  Wall offset: {CENTER_WALL_OFFSET if method == 'recessed' else 0.0} grid units outward")
    print(f"  Wall thickness: {wall_thickness or max(2, int(GRID_SIZE * 0.05))}px")
    print(f"  Circular rooms: {circular}")
    print()

    output_path.mkdir(parents=True, exist_ok=True)

    if debug:
        debug_dir = output_path / "debug"
        debug_dir.mkdir(exist_ok=True)

    processed = 0
    for json_file in json_files:
        png_file = image_path / f"{json_file.stem}.png"

        if not png_file.exists():
            print(f"  Warning: No PNG for {json_file.name}, skipping")
            continue

        try:
            base_name = json_file.stem

            mask = generate_wall_mask(
                json_file, png_file, method, circular, wall_thickness
            )

            # Save mask
            mask_output = output_path / f"{base_name}.png"
            Image.fromarray(mask).save(mask_output)

            # Debug overlay
            if debug:
                img_cv = cv2.imread(str(png_file))
                overlay = img_cv.copy()
                # Red for walls, yellow for doors
                overlay[mask == CLASS_WALL] = [0, 0, 255]
                overlay[mask == CLASS_DOOR] = [0, 255, 255]
                blended = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
                cv2.imwrite(
                    str(debug_dir / f"{base_name}_overlay.png"), blended
                )

            print(f"  {base_name}")
            processed += 1

        except Exception as e:
            print(f"  Error: {json_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print()
    print(f"Processed {processed}/{len(json_files)} dungeons -> {output_dir}")
    print()
    print("Class values:")
    print(f"  0 = Background")
    print(f"  {CLASS_WALL} = Wall (boundaries + secret doors + barred)")
    print(f"  {CLASS_DOOR} = Door (types 1, 4, 5, 7, 8)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate wall line masks from Watabou dungeon exports"
    )
    parser.add_argument("--json-dir", required=True,
                        help="Directory with Watabou JSON files")
    parser.add_argument("--image-dir", required=True,
                        help="Directory with Watabou PNG files")
    parser.add_argument("--output", required=True,
                        help="Output directory for masks")
    parser.add_argument("--method", choices=['edge', 'recessed'], default='edge',
                        help="Wall placement method (default: edge)")
    parser.add_argument("--circular", action='store_true', default=True,
                        help="Render circular rooms (default: True)")
    parser.add_argument("--no-circular", dest='circular', action='store_false',
                        help="Disable circular room detection")
    parser.add_argument("--wall-thickness", type=int, default=None,
                        help=f"Line thickness in pixels (default: {max(2, int(GRID_SIZE * 0.05))})")
    parser.add_argument("--debug", action="store_true",
                        help="Save debug overlay images")

    args = parser.parse_args()

    batch_process(
        args.json_dir, args.image_dir, args.output,
        args.method, args.circular, args.debug, args.wall_thickness
    )
