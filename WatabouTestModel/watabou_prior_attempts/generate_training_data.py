"""
Generate training data from Watabou One Page Dungeon exports.

Workflow:
1. Generate dungeons at https://watabou.itch.io/one-page-dungeon
2. Export both PNG and JSON for each
3. Run this script to create training masks
4. Use PNG as input image, generated mask as ground truth
"""

import json
import numpy as np
from PIL import Image
import cv2
import os
from pathlib import Path


class MatrixMap:
    """Recreate the matrix map logic from the parser."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.uint8)

    def fill_rect(self, x, y, w, h):
        """Fill a rectangular region (represents walls/solid areas)."""
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)

        # Clamp to grid bounds
        x1, x2 = max(0, x1), min(self.width, x2)
        y1, y2 = max(0, y1), min(self.height, y2)

        self.grid[y1:y2, x1:x2] = 255  # Fill with white (wall)

    def get_boundaries(self):
        """
        Extract wall boundaries between filled and empty tiles.
        This creates a mask where walls border playable space.
        """
        # Walls are at edges between filled (255) and empty (0)
        kernel = np.ones((3, 3), np.uint8)

        # Get the boundary of filled regions
        eroded = cv2.erode(self.grid, kernel, iterations=1)
        boundary = self.grid - eroded

        return boundary

    def invert_for_playable(self):
        """
        Invert the grid so playable floor is white, walls/exterior is black.
        This matches Option A labeling: playable=black, walls=white
        """
        return 255 - self.grid


def json_to_mask(json_path, output_size=(512, 512)):
    """
    Convert Watabou JSON to training mask.

    Watabou's rects represent ROOMS (playable space), not walls.
    We create a mask where:
    - Rooms (playable) = BLACK (0)
    - Walls/exterior = WHITE (255)

    Args:
        json_path: Path to exported JSON file
        output_size: Size of output mask (should match PNG size)

    Returns:
        numpy array: Binary mask for training
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Find bounds of the dungeon (handling negative coordinates)
    rects = data.get('rects', [])
    if not rects:
        raise ValueError("No rectangles found in JSON")

    # Find min/max coordinates to handle negatives
    all_x = [r['x'] for r in rects] + [r['x'] + r['w'] for r in rects]
    all_y = [r['y'] for r in rects] + [r['y'] + r['h'] for r in rects]

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    padding = 5
    min_x -= padding
    min_y -= padding
    max_x += padding
    max_y += padding

    width = int(max_x - min_x)
    height = int(max_y - min_y)

    # Start with all WHITE (walls/exterior)
    mask = np.ones((height, width), dtype=np.uint8) * 255

    # Fill ROOMS with BLACK (playable space)
    for rect in rects:
        # Offset coordinates to handle negatives
        x1 = int(rect['x'] - min_x)
        y1 = int(rect['y'] - min_y)
        x2 = int(x1 + rect['w'])
        y2 = int(y1 + rect['h'])

        # Clamp to bounds
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)

        # Fill room with black (playable)
        mask[y1:y2, x1:x2] = 0

    # Doors are already part of room rects, no need to handle separately

    # Resize to match PNG dimensions
    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

    return mask_resized


def process_dungeon_pair(png_path, json_path, output_dir):
    """
    Process a PNG/JSON pair from Watabou generator.

    Args:
        png_path: Path to dungeon PNG image
        json_path: Path to dungeon JSON data
        output_dir: Where to save processed files
    """
    # Load PNG to get dimensions
    img = Image.open(png_path)
    img_array = np.array(img.convert('RGB'))

    # Generate mask from JSON
    mask = json_to_mask(json_path, output_size=(img.width, img.height))

    # Create output filenames
    base_name = Path(png_path).stem

    img_output = Path(output_dir) / "images" / f"{base_name}.png"
    mask_output = Path(output_dir) / "masks" / f"{base_name}.png"

    # Create directories
    img_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)

    # Save processed files
    Image.fromarray(img_array).save(img_output)
    Image.fromarray(mask).save(mask_output)

    print(f"Processed: {base_name}")
    print(f"  Image: {img_output}")
    print(f"  Mask:  {mask_output}")

    return img_output, mask_output


def batch_process(input_dir, output_dir="data/watabou_training"):
    """
    Process all PNG/JSON pairs in a directory.

    Directory structure expected:
    input_dir/
        dungeon1.png
        dungeon1.json
        dungeon2.png
        dungeon2.json
        ...
    """
    input_path = Path(input_dir)

    # Find all JSON files
    json_files = list(input_path.glob("*.json"))

    print(f"Found {len(json_files)} dungeon JSON files")

    processed = 0
    for json_file in json_files:
        # Find corresponding PNG
        png_file = json_file.with_suffix('.png')

        if not png_file.exists():
            print(f"Warning: No PNG found for {json_file.name}, skipping")
            continue

        try:
            process_dungeon_pair(png_file, json_file, output_dir)
            processed += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print(f"\nProcessed {processed}/{len(json_files)} dungeons successfully")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate training data from Watabou dungeons")
    parser.add_argument("input_dir", help="Directory containing PNG/JSON pairs")
    parser.add_argument("--output", default="data/watabou_training", help="Output directory")

    args = parser.parse_args()

    batch_process(args.input_dir, args.output)

    print("\nNext steps:")
    print("1. Split data into train/val: python prepare_dataset.py --split")
    print("2. Train model: python train.py")
