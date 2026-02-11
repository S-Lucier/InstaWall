"""
Foundry VTT to Mask Converter - Hybrid Mode (with Floor Detection)

Converts Foundry VTT scene exports (JSON) into segmentation masks with
explicit floor class detected via flood fill.

Usage:
    python foundry_to_mask_hybrid.py scene.json
    python foundry_to_mask_hybrid.py scene.json --viz
    python foundry_to_mask_hybrid.py ./exports/ --batch --viz
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from PIL import Image, ImageDraw
import cv2


# =============================================================================
# Class Definitions
# =============================================================================

# Scale factor for visually distinguishable mask values
# Training code should divide by this: class_idx = mask_value // CLASS_SCALE
CLASS_SCALE = 50


class ClassID(IntEnum):
    """Class IDs for hybrid mode (with floor detection)

    Mask values are multiplied by CLASS_SCALE (50) for visibility.
    Training: class_index = mask_value // 50

    Note: Secret doors are drawn as walls (visually indistinguishable).
    Windows are converted to walls.
    """
    VOID = 0 * CLASS_SCALE         # 0 - Exterior/outside map
    FLOOR = 1 * CLASS_SCALE        # 50 - Playable interior space
    WALL = 2 * CLASS_SCALE         # 100
    TERRAIN = 3 * CLASS_SCALE      # 150
    DOOR = 4 * CLASS_SCALE         # 200


# Visualization colors (RGB)
VIZ_COLORS = {
    'void': (0, 0, 0),          # Black
    'floor': (100, 100, 100),   # Dark gray
    'wall': (255, 0, 0),        # Red
    'terrain': (0, 255, 255),   # Cyan
    'door': (0, 255, 0),        # Green
}


@dataclass
class WallSegment:
    """Represents a single wall segment from Foundry"""
    id: str
    x1: int
    y1: int
    x2: int
    y2: int
    move: int
    sight: int
    light: int
    door: int

    @property
    def is_window(self) -> bool:
        """Check if this is a window (blocks movement, allows vision).

        Window types in Foundry:
        - Normal window: light=0, sight=0, move=20
        - Proximity window: light=30, sight=30, move=20 (has threshold data)
        """
        if self.door != 0:
            return False
        # Normal window
        if self.light == 0 and self.sight == 0 and self.move == 20:
            return True
        # Proximity window
        if self.light == 30 and self.sight == 30 and self.move == 20:
            return True
        return False

    @property
    def is_door_for_connectivity(self) -> bool:
        """Check if this wall acts as a door for room connectivity purposes.

        Both normal doors AND secret doors count - a room behind a secret door
        is still playable space, not enclosed wall.
        """
        return self.door in (1, 2)

    @property
    def wall_type(self) -> str:
        """Classify wall type based on Foundry values.

        Note: Secret doors are drawn as walls (visually indistinguishable).
        Windows are converted to walls.
        """
        # Windows -> walls (model can't distinguish from wall gaps)
        if self.is_window:
            return 'wall'
        # Secret doors -> walls for drawing (look like walls)
        if self.door == 2:
            return 'wall'
        elif self.door == 1:
            return 'door'
        elif self.sight == 10 and self.light == 10:
            return 'terrain'
        else:
            return 'wall'


@dataclass
class FoundryScene:
    """Parsed Foundry scene data"""
    name: str
    width: int
    height: int
    padding: float
    grid_size: int
    background_src: str
    walls: List[WallSegment]

    @classmethod
    def from_json(cls, json_path: str) -> 'FoundryScene':
        """Load scene from Foundry JSON export"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        walls = []
        for w in data.get('walls', []):
            walls.append(WallSegment(
                id=w['_id'],
                x1=w['c'][0],
                y1=w['c'][1],
                x2=w['c'][2],
                y2=w['c'][3],
                move=w.get('move', 20),
                sight=w.get('sight', 20),
                light=w.get('light', 20),
                door=w.get('door', 0),
            ))

        return cls(
            name=data.get('name', 'Unknown'),
            width=data['width'],
            height=data['height'],
            padding=data.get('padding', 0),
            grid_size=data.get('grid', {}).get('size', 100),
            background_src=data.get('background', {}).get('src', ''),
            walls=walls,
        )

    def calculate_offset(self) -> Tuple[int, int]:
        """Calculate grid-aligned coordinate offset for padding"""
        if self.padding == 0:
            return (0, 0)

        offset_x = round(self.width * self.padding / self.grid_size) * self.grid_size
        offset_y = round(self.height * self.padding / self.grid_size) * self.grid_size
        return (int(offset_x), int(offset_y))

    def get_wall_thickness(self, custom_thickness: Optional[int] = None) -> int:
        """Get wall thickness in pixels (default 5% of grid)"""
        if custom_thickness is not None:
            return custom_thickness
        return max(1, int(self.grid_size * 0.05))

    def get_stats(self) -> Dict[str, int]:
        """Get wall type statistics.

        Note: Secret doors are counted as walls (visually indistinguishable).
        Windows are counted as walls.
        """
        stats = {'wall': 0, 'terrain': 0, 'door': 0, 'secret_door': 0, 'window': 0}
        for wall in self.walls:
            # Count original type for stats (before conversion)
            if wall.is_window:
                stats['window'] += 1
            elif wall.door == 2:
                stats['secret_door'] += 1
            elif wall.door == 1:
                stats['door'] += 1
            elif wall.sight == 10 and wall.light == 10:
                stats['terrain'] += 1
            else:
                stats['wall'] += 1
        return stats

    def remove_edge_walls(self, tolerance: int = 15) -> int:
        """Remove walls that run along the edges of the output mask image.

        The mask is created at full scene dimensions (width x height).
        Walls are drawn at (x - offset_x, y - offset_y).

        So mask edges in canvas coordinates are:
        - left = offset_x (mask x=0)
        - right = width + offset_x (mask x=width)
        - top = offset_y (mask y=0)
        - bottom = height + offset_y (mask y=height)

        For maps without padding (offset=0), edges are at 0 and width/height.

        Args:
            tolerance: Pixels from edge to consider as "on the edge"

        Returns:
            Number of walls removed
        """
        offset_x, offset_y = self.calculate_offset()

        # Mask edges in canvas coordinates
        left = offset_x
        right = self.width + offset_x
        top = offset_y
        bottom = self.height + offset_y

        # For no padding (offset=0), this gives 0 to width/height which is correct

        def wall_runs_along_edge(wall):
            x1, y1 = wall.x1, wall.y1
            x2, y2 = wall.x2, wall.y2

            # Left edge: both x coords near left
            if abs(x1 - left) <= tolerance and abs(x2 - left) <= tolerance:
                return True
            # Right edge: both x coords near right
            if abs(x1 - right) <= tolerance and abs(x2 - right) <= tolerance:
                return True
            # Top edge: both y coords near top
            if abs(y1 - top) <= tolerance and abs(y2 - top) <= tolerance:
                return True
            # Bottom edge: both y coords near bottom
            if abs(y1 - bottom) <= tolerance and abs(y2 - bottom) <= tolerance:
                return True
            return False

        original_count = len(self.walls)
        self.walls = [w for w in self.walls if not wall_runs_along_edge(w)]
        return original_count - len(self.walls)


# =============================================================================
# Mask Generation
# =============================================================================

def generate_mask_hybrid(
    scene: FoundryScene,
    wall_thickness: Optional[int] = None,
    min_region_size: Optional[int] = None,
    background_image: Optional[np.ndarray] = None,
    void_brightness_threshold: int = 100,  # Increased from 30 - exterior backgrounds are often gray
    flood_from_corners_only: bool = True,  # Only flood fill from corners (safer - entrances rarely in corners)
) -> np.ndarray:
    """
    Generate mask using hybrid mode with floor detection.

    Classes (mask values):
        0 = Void/Exterior
        50 = Floor
        100 = Wall
        150 = Terrain
        200 = Door
        250 = Secret Door

    Process:
        1. Draw walls/terrain/doors as thick lines
        2. Flood fill from edges to find exterior void
        3. Remaining unmarked pixels = floor
        4. Small landlocked regions = terrain (solid obstacles)
    """
    mask = np.zeros((scene.height, scene.width), dtype=np.uint8)
    offset_x, offset_y = scene.calculate_offset()
    thickness = scene.get_wall_thickness(wall_thickness)

    # Default min region size: 1 grid cell squared
    if min_region_size is None:
        min_region_size = scene.grid_size * scene.grid_size

    # Create PIL image for drawing
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)

    type_to_class = {
        'wall': ClassID.WALL,
        'terrain': ClassID.TERRAIN,
        'door': ClassID.DOOR,
    }

    # Draw walls (leaving 0 as unmarked for now)
    # Order: terrain < wall < door (later draws on top)
    # Note: secret doors return wall_type='wall', windows return wall_type='wall'
    for wall_type in ['terrain', 'wall', 'door']:
        class_id = type_to_class[wall_type]
        for wall in scene.walls:
            if wall.wall_type == wall_type:
                x1 = wall.x1 - offset_x
                y1 = wall.y1 - offset_y
                x2 = wall.x2 - offset_x
                y2 = wall.y2 - offset_y
                draw.line([(x1, y1), (x2, y2)], fill=int(class_id), width=thickness)

    # Create door connectivity mask (includes both normal doors AND secret doors)
    # This is separate from the visual mask - used for room connectivity check
    door_connectivity_img = Image.new('L', (scene.width, scene.height), 0)
    door_connectivity_draw = ImageDraw.Draw(door_connectivity_img)
    for wall in scene.walls:
        if wall.is_door_for_connectivity:
            x1 = wall.x1 - offset_x
            y1 = wall.y1 - offset_y
            x2 = wall.x2 - offset_x
            y2 = wall.y2 - offset_y
            door_connectivity_draw.line([(x1, y1), (x2, y2)], fill=255, width=thickness)
    door_connectivity_mask = np.array(door_connectivity_img) > 0

    mask = np.array(mask_img)

    # Flood fill from edges to find exterior void
    # Use a temporary marker value (255) for flood fill
    temp_mask = mask.copy()
    flood_value = 255

    h, w = temp_mask.shape

    # Flood fill from all four corners (always done - corners are rarely entrances)
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for cx, cy in corners:
        if temp_mask[cy, cx] == 0:
            cv2.floodFill(temp_mask, None, (cx, cy), flood_value)

    # Optionally also flood fill from all edges (may leak into entrance rooms)
    if not flood_from_corners_only:
        # Top and bottom edges
        for x in range(0, w, max(1, scene.grid_size // 2)):
            if temp_mask[0, x] == 0:
                cv2.floodFill(temp_mask, None, (x, 0), flood_value)
            if temp_mask[h-1, x] == 0:
                cv2.floodFill(temp_mask, None, (x, h-1), flood_value)
        # Left and right edges
        for y in range(0, h, max(1, scene.grid_size // 2)):
            if temp_mask[y, 0] == 0:
                cv2.floodFill(temp_mask, None, (0, y), flood_value)
            if temp_mask[y, w-1] == 0:
                cv2.floodFill(temp_mask, None, (w-1, y), flood_value)

    # Now: 255 = exterior void, 0 = interior (potential floor), other = walls/terrain/doors
    # Create final mask
    final_mask = mask.copy()

    # Color-assisted void detection: only mark as void if actually dark in source image
    flood_filled_mask = (temp_mask == flood_value)

    if background_image is not None:
        # Convert to grayscale if needed
        if len(background_image.shape) == 3:
            gray_bg = cv2.cvtColor(background_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_bg = background_image

        # Find connected components of flood-filled regions
        num_flood_labels, flood_labels = cv2.connectedComponents(flood_filled_mask.astype(np.uint8))

        for label_id in range(1, num_flood_labels):
            region_mask = (flood_labels == label_id)
            # Check average brightness of this region in source image
            avg_brightness = np.mean(gray_bg[region_mask])

            if avg_brightness < void_brightness_threshold:
                # Dark region - mark as void
                final_mask[region_mask] = int(ClassID.VOID)
            else:
                # Bright region - this is actually floor (room connecting to edge)
                final_mask[region_mask] = int(ClassID.FLOOR)
    else:
        # No background image - mark all flood-filled as void (original behavior)
        final_mask[flood_filled_mask] = int(ClassID.VOID)

    # Find remaining unmarked regions (potential floor or landlocked)
    remaining_unmarked = (mask == 0) & (~flood_filled_mask)

    # Find connected components of remaining unmarked regions
    num_labels, labels = cv2.connectedComponents(remaining_unmarked.astype(np.uint8))

    # Check if the map has ANY doors at all
    has_any_doors = np.any(door_connectivity_mask)

    if has_any_doors:
        # First pass: identify which regions have doors (including secret doors)
        # A region "has a door" if door connectivity pixels touch the region boundary
        # Note: door_connectivity_mask was created earlier and includes both doors and secret doors
        regions_with_doors = set()

        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)

            # Dilate the region by 1 and check overlap with door connectivity pixels
            region_dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            door_touches_region = np.any(door_connectivity_mask & region_dilated.astype(bool))

            if door_touches_region:
                regions_with_doors.add(label_id)

        # Second pass: find regions that touch door-connected regions (transitive)
        # Build adjacency by dilating each region and checking overlap
        region_adjacency = {i: set() for i in range(1, num_labels)}

        for label_id in range(1, num_labels):
            region_mask = (labels == label_id)
            # Dilate region to find neighbors
            dilated = cv2.dilate(region_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            # Find which other labels this touches
            neighbor_labels = np.unique(labels[dilated.astype(bool) & (labels != label_id) & (labels != 0)])
            region_adjacency[label_id].update(neighbor_labels)

        # BFS to find all regions connected to door-regions
        floor_regions = set(regions_with_doors)
        queue = list(regions_with_doors)
        while queue:
            current = queue.pop(0)
            for neighbor in region_adjacency[current]:
                if neighbor not in floor_regions:
                    floor_regions.add(neighbor)
                    queue.append(neighbor)
    else:
        # No doors on the map - treat all regions as floor (skip door connectivity check)
        floor_regions = set(range(1, num_labels))

    # Now classify regions
    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_size = np.sum(region_mask)

        if region_size < min_region_size:
            # Too small - mark as terrain regardless (tiny enclosed areas)
            final_mask[region_mask] = int(ClassID.TERRAIN)
        elif label_id in floor_regions:
            # Connected to a door (directly or transitively), or no doors on map - mark as floor
            final_mask[region_mask] = int(ClassID.FLOOR)
        else:
            # NOT connected to any door - mark as terrain
            # Even large regions without door access are likely decorative/solid
            final_mask[region_mask] = int(ClassID.TERRAIN)

    return final_mask


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(
    mask: np.ndarray,
    background_image: Optional[Image.Image] = None,
) -> Image.Image:
    """Create colored visualization of the mask overlaid on background.

    Floor and terrain areas are semi-transparent to show the map underneath.
    Walls and doors are opaque colored overlays.
    """
    h, w = mask.shape

    if background_image is not None:
        # Resize background if needed
        bg = background_image.convert('RGB')
        if bg.size != (w, h):
            bg = bg.resize((w, h), Image.Resampling.LANCZOS)
        viz = np.array(bg, dtype=np.float32)
    else:
        viz = np.zeros((h, w, 3), dtype=np.float32)

    # Semi-transparent overlays for floor/terrain (alpha blend)
    # Floor: slight darkening to show it's marked
    floor_mask = (mask == int(ClassID.FLOOR))
    viz[floor_mask] = viz[floor_mask] * 0.7 + np.array([50, 50, 50]) * 0.3

    # Terrain: cyan tint, semi-transparent
    terrain_mask = (mask == int(ClassID.TERRAIN))
    viz[terrain_mask] = viz[terrain_mask] * 0.4 + np.array(VIZ_COLORS['terrain']) * 0.6

    # Void: darken significantly
    void_mask = (mask == int(ClassID.VOID))
    viz[void_mask] = viz[void_mask] * 0.3

    # Opaque overlays for walls and doors (important to see clearly)
    # Note: Secret doors are drawn as walls, so they appear red like walls
    wall_mask = (mask == int(ClassID.WALL))
    viz[wall_mask] = VIZ_COLORS['wall']

    door_mask = (mask == int(ClassID.DOOR))
    viz[door_mask] = VIZ_COLORS['door']

    return Image.fromarray(viz.astype(np.uint8))


# =============================================================================
# File Handling
# =============================================================================

def find_matching_image(json_path: str, scene: FoundryScene) -> Optional[str]:
    """Find the background image file matching the scene"""
    json_dir = Path(json_path).parent

    # Try exact match from background.src
    if scene.background_src:
        from urllib.parse import unquote
        src_filename = unquote(Path(scene.background_src).name)
        candidate = json_dir / src_filename
        if candidate.exists():
            return str(candidate)

    # Try matching by scene name
    scene_name_clean = scene.name.replace(' ', '_').replace('-', '_')

    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        patterns = [
            json_dir / f"{scene.name}{ext}",
            json_dir / f"{scene_name_clean}{ext}",
        ]
        for pattern in patterns:
            if pattern.exists():
                return str(pattern)

    # Look for any image with similar name
    for img_file in json_dir.glob('*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            img_name = img_file.stem.lower().replace(' ', '').replace('-', '').replace('_', '')
            scene_name = scene.name.lower().replace(' ', '').replace('-', '').replace('_', '')
            if scene_name in img_name or img_name in scene_name:
                return str(img_file)

    return None


def floor_to_wall_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a floor mask to a wall mask by inverting floor pixels to wall.

    - FLOOR (50) → WALL (100)
    - TERRAIN stays TERRAIN (150)
    - Everything else unchanged
    """
    wall_mask = mask.copy()
    wall_mask[mask == int(ClassID.FLOOR)] = int(ClassID.WALL)
    return wall_mask


# Default output directories relative to project root
DEFAULT_OUTPUT_DIRS = {
    'floor': 'data/foundry_to_mask/floor_masks',
    'wall': 'data/foundry_to_mask/wall_masks',
}


def get_project_root() -> Path:
    """Get the project root (parent of Tools directory)"""
    return Path(__file__).parent.parent


def process_single_file(
    json_path: str,
    output_dir: Optional[str] = None,
    wall_thickness: Optional[int] = None,
    min_region_size: Optional[int] = None,
    create_viz: bool = False,
    image_path: Optional[str] = None,
    void_threshold: int = 100,
    corners_only: bool = True,
    mask_mode: str = 'floor',
    remove_edge_walls: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Process a single Foundry JSON file.

    Returns:
        Tuple of (mask_path, viz_path or None)
    """
    # Load scene
    scene = FoundryScene.from_json(json_path)
    print(f"Processing: {scene.name}")
    print(f"  Dimensions: {scene.width}x{scene.height}, Grid: {scene.grid_size}px, Padding: {scene.padding}")

    # Remove edge walls if requested
    if remove_edge_walls:
        removed = scene.remove_edge_walls()
        if removed > 0:
            print(f"  Removed {removed} edge wall(s)")

    stats = scene.get_stats()
    print(f"  Walls: {stats['wall']}, Terrain: {stats['terrain']}, Doors: {stats['door']}, Secret: {stats['secret_door']}, Windows: {stats['window']}")

    # Find background image for color-assisted void detection
    if image_path is None:
        image_path = find_matching_image(str(json_path), scene)

    bg_array = None
    if image_path:
        try:
            bg_img = Image.open(image_path)
            # Resize if dimensions don't match
            if bg_img.size != (scene.width, scene.height):
                bg_img = bg_img.resize((scene.width, scene.height), Image.Resampling.LANCZOS)
            bg_array = np.array(bg_img.convert('RGB'))
            print(f"  Using background for color-assisted void detection: {Path(image_path).name}")
        except Exception as e:
            print(f"  Warning: Could not load background image: {e}")

    # Generate mask
    mask = generate_mask_hybrid(
        scene,
        wall_thickness,
        min_region_size,
        bg_array,
        void_brightness_threshold=void_threshold,
        flood_from_corners_only=corners_only,
    )

    # Convert to wall mask if requested
    if mask_mode == 'wall':
        mask = floor_to_wall_mask(mask)
        print("  Converted floor mask to wall mask (floor -> wall)")

    # Count class pixels
    unique, counts = np.unique(mask, return_counts=True)
    print("  Class distribution:")
    class_names = {0: 'Void', 50: 'Floor', 100: 'Wall', 150: 'Terrain', 200: 'Door', 250: 'Secret'}
    for val, cnt in zip(unique, counts):
        pct = cnt / mask.size * 100
        name = class_names.get(val, '?')
        print(f"    {name}: {pct:.1f}%")

    # Determine output paths
    json_path = Path(json_path)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Use default output directory based on mode
        out_dir = get_project_root() / DEFAULT_OUTPUT_DIRS[mask_mode]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from scene name
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in scene.name)
    suffix = 'floor' if mask_mode == 'floor' else 'wall'
    mask_path = out_dir / f"{safe_name}_mask_{suffix}.png"

    # Save mask
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)
    print(f"  Saved mask: {mask_path}")

    # Create visualization if requested
    viz_path = None
    if create_viz:
        bg_image = None
        if bg_array is not None:
            bg_image = Image.fromarray(bg_array)

        viz = create_visualization(mask, bg_image)
        viz_path = out_dir / f"{safe_name}_viz_{suffix}.jpg"
        viz.save(viz_path, quality=90)
        print(f"  Saved viz: {viz_path}")

    return str(mask_path), str(viz_path) if viz_path else None


def process_batch(
    input_dir: str,
    output_dir: Optional[str] = None,
    wall_thickness: Optional[int] = None,
    min_region_size: Optional[int] = None,
    create_viz: bool = False,
    void_threshold: int = 100,
    corners_only: bool = True,
    mask_mode: str = 'floor',
    remove_edge_walls: bool = False,
) -> List[Tuple[str, Optional[str]]]:
    """Process all JSON files in a directory"""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))
    json_files = [f for f in json_files if f.name.startswith('fvtt-') or 'scene' in f.name.lower()]

    if not json_files:
        print(f"No Foundry JSON files found in {input_dir}")
        return []

    print(f"Found {len(json_files)} Foundry scene files")
    print(f"Mode: {mask_mode} mask")
    print("=" * 60)

    results = []
    for json_file in json_files:
        try:
            result = process_single_file(
                str(json_file),
                output_dir=output_dir,
                wall_thickness=wall_thickness,
                min_region_size=min_region_size,
                create_viz=create_viz,
                void_threshold=void_threshold,
                corners_only=corners_only,
                mask_mode=mask_mode,
                remove_edge_walls=remove_edge_walls,
            )
            results.append(result)
        except Exception as e:
            print(f"  Error processing {json_file.name}: {e}")
        print()

    print("=" * 60)
    print(f"Processed {len(results)}/{len(json_files)} files successfully")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert Foundry VTT scene exports to segmentation masks (hybrid mode with floor detection)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  floor - Interior marked as floor (class 50)
  wall  - Interior marked as wall (class 100), terrain stays terrain

Class values (divide by 50 for class index):
  0   = Void (exterior)
  50  = Floor (interior) - only in floor mode
  100 = Wall
  150 = Terrain
  200 = Door
  250 = Secret Door

Default output directories:
  floor mode: data/foundry_to_mask/floor_masks
  wall mode:  data/foundry_to_mask/wall_masks

Examples:
  python foundry_to_mask_hybrid.py scene.json --mode floor --viz
  python foundry_to_mask_hybrid.py scene.json --mode wall --viz
  python foundry_to_mask_hybrid.py ./exports/ --batch --mode floor
        """
    )

    parser.add_argument('input', help='JSON file or directory (with --batch)')
    parser.add_argument('--mode', choices=['floor', 'wall'], default='floor',
                        help='Mask mode: floor (interior=floor) or wall (interior=wall)')
    parser.add_argument('--batch', action='store_true',
                        help='Process all JSON files in directory')
    parser.add_argument('--output', '-o', help='Output directory (default: mode-specific dir)')
    parser.add_argument('--wall-thickness', type=int,
                        help='Wall thickness in pixels (default: 5%% of grid)')
    parser.add_argument('--min-region', type=int,
                        help='Min region size for floor (default: 1 grid cell squared)')
    parser.add_argument('--viz', action='store_true',
                        help='Create visualization overlay')
    parser.add_argument('--image', help='Background image path (auto-detected if not specified)')
    parser.add_argument('--void-threshold', type=int, default=100,
                        help='Brightness threshold for void detection (default: 100)')
    parser.add_argument('--flood-edges', action='store_true',
                        help='Flood fill from all edges, not just corners (may leak into entrance rooms)')
    parser.add_argument('--remove-edge-walls', action='store_true',
                        help='Remove walls running along map image edges')

    args = parser.parse_args()

    if args.batch:
        process_batch(
            args.input,
            output_dir=args.output,
            wall_thickness=args.wall_thickness,
            min_region_size=args.min_region,
            create_viz=args.viz,
            void_threshold=args.void_threshold,
            corners_only=not args.flood_edges,
            mask_mode=args.mode,
            remove_edge_walls=args.remove_edge_walls,
        )
    else:
        process_single_file(
            args.input,
            output_dir=args.output,
            wall_thickness=args.wall_thickness,
            min_region_size=args.min_region,
            create_viz=args.viz,
            image_path=args.image,
            void_threshold=args.void_threshold,
            corners_only=not args.flood_edges,
            mask_mode=args.mode,
            remove_edge_walls=args.remove_edge_walls,
        )


if __name__ == '__main__':
    main()
