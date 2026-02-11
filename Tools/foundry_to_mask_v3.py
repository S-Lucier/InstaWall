"""
Foundry VTT to Mask Converter

Converts Foundry VTT scene exports (JSON) into segmentation masks for training.

Usage:
    python foundry_to_mask.py scene.json --mode lines
    python foundry_to_mask.py scene.json --mode hybrid --viz
    python foundry_to_mask.py ./exports/ --batch --mode lines
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
# Training code should divide by this to get class indices: class_idx = mask_value // CLASS_SCALE
CLASS_SCALE = 50

class ClassID_LinesOnly(IntEnum):
    """Class IDs for lines-only mode (no floor detection)

    Mask values are multiplied by CLASS_SCALE (50) for visibility.
    Training: class_index = mask_value // 50

    Note: Secret doors are drawn as walls (visually indistinguishable).
    Windows are converted to walls.
    """
    BACKGROUND = 0 * CLASS_SCALE   # 0 - Unmarked
    WALL = 1 * CLASS_SCALE         # 50
    TERRAIN = 2 * CLASS_SCALE      # 100
    DOOR = 3 * CLASS_SCALE         # 150


class ClassID_Hybrid(IntEnum):
    """Class IDs for hybrid mode (with floor detection)

    Mask values are multiplied by CLASS_SCALE (50) for visibility.
    Training: class_index = mask_value // 50

    Note: Secret doors are drawn as walls (visually indistinguishable).
    Windows are converted to walls.
    """
    VOID = 0 * CLASS_SCALE         # 0 - Exterior
    FLOOR = 1 * CLASS_SCALE        # 50 - Interior
    WALL = 2 * CLASS_SCALE         # 100
    TERRAIN = 3 * CLASS_SCALE      # 150
    DOOR = 4 * CLASS_SCALE         # 200


# Visualization colors (RGB)
VIZ_COLORS = {
    'wall': (255, 0, 0),        # Red
    'terrain': (0, 255, 255),   # Cyan
    'door': (0, 255, 0),        # Green
    'floor': (100, 100, 100),   # Dark gray
    'void': (0, 0, 0),          # Black
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
        """Calculate grid-aligned coordinate offset for padding.

        Foundry uses Math.ceil to snap padding to grid boundaries.
        """
        if self.padding == 0:
            return (0, 0)

        offset_x = math.ceil(self.width * self.padding / self.grid_size) * self.grid_size
        offset_y = math.ceil(self.height * self.padding / self.grid_size) * self.grid_size
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

def generate_mask_lines_only(
    scene: FoundryScene,
    wall_thickness: Optional[int] = None
) -> np.ndarray:
    """
    Generate mask using lines-only mode (Mode A).

    Classes:
        0 = Background (unmarked)
        1 = Wall (includes secret doors and windows)
        2 = Terrain
        3 = Door
    """
    mask = np.zeros((scene.height, scene.width), dtype=np.uint8)
    offset_x, offset_y = scene.calculate_offset()
    thickness = scene.get_wall_thickness(wall_thickness)

    # Create PIL image for drawing (easier anti-aliased lines)
    mask_img = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask_img)

    # Draw walls by type (order matters for overlaps)
    # Note: secret doors return wall_type='wall', windows return wall_type='wall'
    type_to_class = {
        'wall': ClassID_LinesOnly.WALL,
        'terrain': ClassID_LinesOnly.TERRAIN,
        'door': ClassID_LinesOnly.DOOR,
    }

    # Draw in order: terrain < wall < door (later draws on top)
    for wall_type in ['terrain', 'wall', 'door']:
        class_id = type_to_class[wall_type]
        for wall in scene.walls:
            if wall.wall_type == wall_type:
                x1 = wall.x1 - offset_x
                y1 = wall.y1 - offset_y
                x2 = wall.x2 - offset_x
                y2 = wall.y2 - offset_y
                draw.line([(x1, y1), (x2, y2)], fill=int(class_id), width=thickness)

    return np.array(mask_img)


def generate_mask_hybrid(
    scene: FoundryScene,
    wall_thickness: Optional[int] = None,
    min_region_size: Optional[int] = None
) -> np.ndarray:
    """
    Generate mask using hybrid mode (Mode C).

    Classes:
        0 = Void/Exterior
        1 = Floor
        2 = Wall (includes secret doors and windows)
        3 = Terrain
        4 = Door
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

    # Note: secret doors return wall_type='wall', windows return wall_type='wall'
    type_to_class = {
        'wall': ClassID_Hybrid.WALL,
        'terrain': ClassID_Hybrid.TERRAIN,
        'door': ClassID_Hybrid.DOOR,
    }

    # Draw walls (leaving 0 as unmarked for now)
    # Order: terrain < wall < door (later draws on top)
    for wall_type in ['terrain', 'wall', 'door']:
        class_id = type_to_class[wall_type]
        for wall in scene.walls:
            if wall.wall_type == wall_type:
                x1 = wall.x1 - offset_x
                y1 = wall.y1 - offset_y
                x2 = wall.x2 - offset_x
                y2 = wall.y2 - offset_y
                draw.line([(x1, y1), (x2, y2)], fill=int(class_id), width=thickness)

    mask = np.array(mask_img)

    # Flood fill from corners to find exterior void
    # Use a temporary marker value (255) for flood fill
    temp_mask = mask.copy()
    flood_value = 255

    # Flood fill from all four corners
    h, w = temp_mask.shape
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]

    for cx, cy in corners:
        if temp_mask[cy, cx] == 0:  # Only fill if unmarked
            cv2.floodFill(temp_mask, None, (cx, cy), flood_value)

    # Also flood fill from edges (in case corners are blocked)
    # Top and bottom edges
    for x in range(0, w, scene.grid_size):
        if temp_mask[0, x] == 0:
            cv2.floodFill(temp_mask, None, (x, 0), flood_value)
        if temp_mask[h-1, x] == 0:
            cv2.floodFill(temp_mask, None, (x, h-1), flood_value)
    # Left and right edges
    for y in range(0, h, scene.grid_size):
        if temp_mask[y, 0] == 0:
            cv2.floodFill(temp_mask, None, (0, y), flood_value)
        if temp_mask[y, w-1] == 0:
            cv2.floodFill(temp_mask, None, (w-1, y), flood_value)

    # Now: 255 = exterior void, 0 = interior (floor), other = walls/terrain/doors
    # Create final mask
    final_mask = mask.copy()

    # Mark exterior void as class 0
    final_mask[temp_mask == flood_value] = ClassID_Hybrid.VOID

    # Mark remaining unmarked (0 in original) as floor
    # But first handle landlocked regions
    remaining_unmarked = (mask == 0) & (temp_mask != flood_value)

    # Find connected components of remaining unmarked regions
    num_labels, labels = cv2.connectedComponents(remaining_unmarked.astype(np.uint8))

    for label_id in range(1, num_labels):
        region_mask = (labels == label_id)
        region_size = np.sum(region_mask)

        if region_size < min_region_size:
            # Small landlocked region - mark as terrain (solid obstacle)
            final_mask[region_mask] = ClassID_Hybrid.TERRAIN
        else:
            # Large region - mark as floor
            final_mask[region_mask] = ClassID_Hybrid.FLOOR

    return final_mask


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(
    mask: np.ndarray,
    background_image: Optional[Image.Image] = None,
    mode: str = 'lines'
) -> Image.Image:
    """Create colored visualization of the mask"""
    h, w = mask.shape

    if background_image is not None:
        # Resize background if needed
        bg = background_image.convert('RGB')
        if bg.size != (w, h):
            bg = bg.resize((w, h), Image.Resampling.LANCZOS)
        viz = np.array(bg)
    else:
        viz = np.zeros((h, w, 3), dtype=np.uint8)

    # Overlay colors based on mode (using scaled class values)
    # Note: Secret doors are drawn as walls, so they appear red like walls
    if mode == 'lines':
        class_colors = {
            int(ClassID_LinesOnly.WALL): VIZ_COLORS['wall'],
            int(ClassID_LinesOnly.TERRAIN): VIZ_COLORS['terrain'],
            int(ClassID_LinesOnly.DOOR): VIZ_COLORS['door'],
        }
    else:  # hybrid
        class_colors = {
            int(ClassID_Hybrid.WALL): VIZ_COLORS['wall'],
            int(ClassID_Hybrid.TERRAIN): VIZ_COLORS['terrain'],
            int(ClassID_Hybrid.DOOR): VIZ_COLORS['door'],
            int(ClassID_Hybrid.FLOOR): VIZ_COLORS['floor'],
        }

    # Apply colors
    for class_id, color in class_colors.items():
        class_mask = (mask == class_id)
        viz[class_mask] = color

    return Image.fromarray(viz)


# =============================================================================
# File Handling
# =============================================================================

def find_matching_image(json_path: str, scene: FoundryScene) -> Optional[str]:
    """Find the background image file matching the scene"""
    json_dir = Path(json_path).parent

    # Try exact match from background.src
    if scene.background_src:
        # Extract filename from path (may have URL encoding)
        from urllib.parse import unquote
        src_filename = unquote(Path(scene.background_src).name)

        # Look for it in same directory
        candidate = json_dir / src_filename
        if candidate.exists():
            return str(candidate)

    # Try matching by scene name
    scene_name_clean = scene.name.replace(' ', '_').replace('-', '_')

    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        # Try various name patterns
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


# Default output directory for lines-only mode
DEFAULT_OUTPUT_DIR = 'data/foundry_to_mask/line_masks'


def get_project_root() -> Path:
    """Get the project root (parent of Tools directory)"""
    return Path(__file__).parent.parent


def process_single_file(
    json_path: str,
    mode: str = 'lines',
    output_dir: Optional[str] = None,
    wall_thickness: Optional[int] = None,
    create_viz: bool = False,
    image_path: Optional[str] = None,
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

    # Generate mask
    if mode == 'lines':
        mask = generate_mask_lines_only(scene, wall_thickness)
    else:
        mask = generate_mask_hybrid(scene, wall_thickness)

    # Determine output paths
    json_path = Path(json_path)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Use default output directory
        out_dir = get_project_root() / DEFAULT_OUTPUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from scene name
    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in scene.name)
    mask_path = out_dir / f"{safe_name}_mask_lines.png"

    # Save mask
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)
    print(f"  Saved mask: {mask_path}")

    # Create visualization if requested
    viz_path = None
    if create_viz:
        # Find background image
        if image_path is None:
            image_path = find_matching_image(str(json_path), scene)

        bg_image = None
        if image_path:
            try:
                bg_image = Image.open(image_path)
                print(f"  Using background: {Path(image_path).name}")
            except Exception as e:
                print(f"  Warning: Could not load background image: {e}")

        viz = create_visualization(mask, bg_image, mode)
        viz_path = out_dir / f"{safe_name}_viz_lines.jpg"
        viz.save(viz_path, quality=90)
        print(f"  Saved viz: {viz_path}")

    return str(mask_path), str(viz_path) if viz_path else None


def process_batch(
    input_dir: str,
    mode: str = 'lines',
    output_dir: Optional[str] = None,
    wall_thickness: Optional[int] = None,
    create_viz: bool = False,
    remove_edge_walls: bool = False,
) -> List[Tuple[str, Optional[str]]]:
    """Process all JSON files in a directory"""
    input_path = Path(input_dir)
    json_files = list(input_path.glob('*.json'))

    # Filter to Foundry exports (they start with 'fvtt-')
    json_files = [f for f in json_files if f.name.startswith('fvtt-') or 'scene' in f.name.lower()]

    if not json_files:
        print(f"No Foundry JSON files found in {input_dir}")
        return []

    print(f"Found {len(json_files)} Foundry scene files")
    print("=" * 60)

    results = []
    for json_file in json_files:
        try:
            result = process_single_file(
                str(json_file),
                mode=mode,
                output_dir=output_dir,
                wall_thickness=wall_thickness,
                create_viz=create_viz,
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
        description='Convert Foundry VTT scene exports to segmentation masks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Default output directory: data/foundry_to_mask/line_masks

Examples:
  python foundry_to_mask.py scene.json
  python foundry_to_mask.py scene.json --viz
  python foundry_to_mask.py ./exports/ --batch --viz
        """
    )

    parser.add_argument('input', help='JSON file or directory (with --batch)')
    parser.add_argument('--mode', choices=['lines', 'hybrid'], default='lines',
                        help='Mask generation mode (default: lines). Note: use foundry_to_mask_hybrid.py for floor/wall modes')
    parser.add_argument('--batch', action='store_true',
                        help='Process all JSON files in directory')
    parser.add_argument('--output', '-o', help='Output directory (default: data/foundry_to_mask/line_masks)')
    parser.add_argument('--wall-thickness', type=int,
                        help='Wall thickness in pixels (default: 5%% of grid)')
    parser.add_argument('--viz', action='store_true',
                        help='Create visualization overlay')
    parser.add_argument('--image', help='Background image path (auto-detected if not specified)')
    parser.add_argument('--remove-edge-walls', action='store_true',
                        help='Remove walls running along map image edges')

    args = parser.parse_args()

    if args.batch:
        process_batch(
            args.input,
            mode=args.mode,
            output_dir=args.output,
            wall_thickness=args.wall_thickness,
            create_viz=args.viz,
            remove_edge_walls=args.remove_edge_walls,
        )
    else:
        process_single_file(
            args.input,
            mode=args.mode,
            output_dir=args.output,
            wall_thickness=args.wall_thickness,
            create_viz=args.viz,
            image_path=args.image,
            remove_edge_walls=args.remove_edge_walls,
        )


if __name__ == '__main__':
    main()
